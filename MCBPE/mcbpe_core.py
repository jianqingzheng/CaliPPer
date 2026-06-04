"""Core M-CBPE implementation for immune receptor binding prediction.

Implements the four-step M-CBPE algorithm:
1. Density Ratio Estimation (DRE) via logistic regression on features
2. Importance-weighted isotonic calibration
3. Apply calibrator to production (test) predictions
4. Estimate performance metrics from calibrated probabilities

Integration with S2DD:
- Uses S2DD distance as the primary feature for DRE
- Optionally includes per-chain distances and model confidence
- Supports per-bin calibration for hybrid distance-performance analysis
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import pearsonr, spearmanr


# ── Step 1: Density Ratio Estimation ────────────────────────────────────

def estimate_density_ratios(ref_features, prod_features, method='logistic'):
    """Estimate importance weights w(x) = p_prod(x) / p_ref(x).

    Uses a binary classifier trained to distinguish reference (z=0) from
    production (z=1) samples. The density ratio is:
        w(x) = (n_ref / n_prod) * p(z=1|x) / p(z=0|x)

    Args:
        ref_features: (n_ref, d) array of reference (training) features
        prod_features: (n_prod, d) array of production (test) features
        method: 'logistic' (default) or 'rf'

    Returns:
        ref_weights: (n_ref,) importance weights for reference samples
        prod_weights: (n_prod,) importance weights for production samples
    """
    n_ref = len(ref_features)
    n_prod = len(prod_features)

    # Stack features and create domain labels
    X = np.vstack([ref_features, prod_features])
    z = np.concatenate([np.zeros(n_ref), np.ones(n_prod)])

    if method == 'logistic':
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    elif method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown DRE method: {method}")

    clf.fit(X, z)
    probs = clf.predict_proba(X)  # [p(z=0|x), p(z=1|x)]
    p_prod = np.clip(probs[:, 1], 1e-8, 1 - 1e-8)
    p_ref = 1 - p_prod

    # Density ratio: w(x) = (n_ref / n_prod) * p(z=1|x) / p(z=0|x)
    ratio = (n_ref / n_prod) * (p_prod / p_ref)

    # Clip extreme weights
    ratio = np.clip(ratio, 0.01, 100.0)

    ref_weights = ratio[:n_ref]
    prod_weights = ratio[n_ref:]

    return ref_weights, prod_weights


# ── Step 2: Importance-Weighted Calibration ─────────────────────────────

def fit_weighted_calibrator(ref_predictions, ref_labels, ref_weights):
    """Fit isotonic regression calibrator with importance weights.

    Maps uncalibrated model scores → calibrated P(y=1|x) under the
    production distribution, using importance-weighted reference data.

    Args:
        ref_predictions: (n_ref,) model predicted probabilities on reference
        ref_labels: (n_ref,) ground truth binary labels on reference
        ref_weights: (n_ref,) importance weights from DRE

    Returns:
        calibrator: fitted IsotonicRegression object
    """
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds='clip'
    )
    calibrator.fit(ref_predictions, ref_labels, sample_weight=ref_weights)
    return calibrator


# ── Step 3: Apply Calibrator ────────────────────────────────────────────

def calibrate_predictions(calibrator, predictions):
    """Apply fitted calibrator to production predictions.

    Args:
        calibrator: fitted IsotonicRegression
        predictions: (n,) uncalibrated model probabilities

    Returns:
        calibrated: (n,) calibrated probabilities
    """
    return calibrator.predict(predictions)


# ── Step 4: Estimate Metrics ────────────────────────────────────────────

def estimate_metric_from_calibrated(calibrated_probs, metric_name,
                                     threshold=0.5):
    """Estimate aggregate performance metric from calibrated probabilities.

    Uses the M-CBPE formula:
        m_hat = mean[ c(f(x)) * m(rho(x), 1) + (1-c(f(x))) * m(rho(x), 0) ]

    where c(f(x)) is the calibrated probability and rho(x) is the predicted
    class at the given threshold.

    Args:
        calibrated_probs: (n,) calibrated P(y=1|x)
        metric_name: 'aucroc', 'ap', 'f1', 'acc', 'mcc'
        threshold: decision threshold for binary predictions

    Returns:
        estimated_value: scalar estimated metric value
    """
    c = calibrated_probs
    pred_class = (c >= threshold).astype(int)

    if metric_name == 'acc':
        # E[I(pred == y)] = c*I(pred==1) + (1-c)*I(pred==0)
        return np.mean(c * pred_class + (1 - c) * (1 - pred_class))

    elif metric_name == 'f1':
        # Expected TP, FP, FN
        e_tp = np.sum(c * pred_class)
        e_fp = np.sum((1 - c) * pred_class)
        e_fn = np.sum(c * (1 - pred_class))
        precision = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0
        recall = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    elif metric_name == 'aucroc':
        # For AUROC, use the calibrated probabilities as both the score
        # and the expected label. This is equivalent to the Wilcoxon-Mann-Whitney
        # estimator with expected labels.
        return _estimate_auroc_from_calibrated(c)

    elif metric_name == 'ap':
        # For AP, we use a threshold-sweep approximation
        return _estimate_ap_from_calibrated(c)

    elif metric_name == 'mcc':
        e_tp = np.sum(c * pred_class)
        e_fp = np.sum((1 - c) * pred_class)
        e_fn = np.sum(c * (1 - pred_class))
        e_tn = np.sum((1 - c) * (1 - pred_class))
        numer = e_tp * e_tn - e_fp * e_fn
        denom = np.sqrt((e_tp + e_fp) * (e_tp + e_fn) *
                        (e_tn + e_fp) * (e_tn + e_fn))
        if denom == 0:
            return 0.0
        return numer / denom

    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _estimate_auroc_from_calibrated(calibrated_probs):
    """Estimate AUROC using calibrated probs as expected labels.

    For each pair (i, j) where c_i > c_j, the expected contribution to
    AUROC is c_i * (1-c_j) * I(score_i > score_j).
    Since calibrated_probs ARE the scores, this simplifies.

    BUG FIX 2026-04-27: was sorted descending → inverted result.
    Now sorted ascending so positive samples (high c) accumulate
    all negative samples (low c) ranked below them.
    """
    c = np.sort(calibrated_probs)  # ascending: negatives first
    n = len(c)
    if n < 2:
        return 0.5

    # Expected positive rate and negative rate per sample
    # Approximate via sorted calibrated probs
    e_pos = np.sum(c)
    e_neg = n - e_pos
    if e_pos == 0 or e_neg == 0:
        return 0.5

    # Mann-Whitney U statistic with expected labels
    # U = sum over all pairs: P(score_pos > score_neg)
    # Using calibrated probs as both scores and expected labels
    rank_sum = 0.0
    cum_neg = 0.0
    for i in range(n):
        neg_weight = 1 - c[i]
        rank_sum += c[i] * cum_neg
        cum_neg += neg_weight

    if e_pos * e_neg == 0:
        return 0.5
    return rank_sum / (e_pos * e_neg)


def _estimate_ap_from_calibrated(calibrated_probs, n_thresholds=100):
    """Estimate AP from calibrated probabilities.

    Uses each sample's calibrated probability c_i as a soft label.
    Sorts by c_i descending (same as sorting by predicted score),
    then sweeps through computing precision@k and recall@k.

    BUG FIX 2026-04-28: old version used threshold sweep which produced
    near-constant recall (e_tp/e_total_pos barely changed) → AP ≈ 0.
    New version uses rank-based sweep: at each rank k, precision@k =
    sum(c[:k]) / k, recall@k = sum(c[:k]) / sum(c).
    """
    c = calibrated_probs
    n = len(c)
    if n < 2:
        return 0.0

    e_total_pos = np.sum(c)
    if e_total_pos == 0:
        return 0.0

    # Sort by calibrated prob descending (highest confidence first)
    order = np.argsort(c)[::-1]
    c_sorted = c[order]

    # Compute precision@k and recall@k at each rank
    cum_tp = np.cumsum(c_sorted)          # expected TP up to rank k
    precision_at_k = cum_tp / np.arange(1, n + 1)  # precision = TP / (TP+FP)
    recall_at_k = cum_tp / e_total_pos    # recall = TP / total_pos

    # AP = sum of precision@k * delta_recall@k (step function)
    delta_recall = np.diff(recall_at_k, prepend=0)
    ap = np.sum(precision_at_k * delta_recall)

    return float(np.clip(ap, 0, 1))


# ── Full M-CBPE Pipeline ───────────────────────────────────────────────

def mcbpe_estimate(ref_predictions, ref_labels, prod_predictions,
                   ref_features, prod_features,
                   metrics=('aucroc', 'ap', 'f1'),
                   dre_method='logistic'):
    """Run the full M-CBPE pipeline.

    Args:
        ref_predictions: model probs on reference (training) data
        ref_labels: ground truth on reference data
        prod_predictions: model probs on production (test) data
        ref_features: feature matrix for reference (e.g., S2DD distances)
        prod_features: feature matrix for production
        metrics: tuple of metric names to estimate
        dre_method: DRE classifier type

    Returns:
        dict with:
            'estimated': {metric: value} estimated metrics
            'calibrated_probs': calibrated production probabilities
            'ref_weights': importance weights on reference data
            'prod_weights': importance weights on production data
    """
    # Step 1: DRE
    ref_weights, prod_weights = estimate_density_ratios(
        ref_features, prod_features, method=dre_method)

    # Step 2: Weighted calibration
    calibrator = fit_weighted_calibrator(
        ref_predictions, ref_labels, ref_weights)

    # Step 3: Calibrate production predictions
    calibrated = calibrate_predictions(calibrator, prod_predictions)

    # Step 4: Estimate metrics
    estimated = {}
    for m in metrics:
        estimated[m] = estimate_metric_from_calibrated(calibrated, m)

    return {
        'estimated': estimated,
        'calibrated_probs': calibrated,
        'ref_weights': ref_weights,
        'prod_weights': prod_weights,
        'calibrator': calibrator,
    }


# ── Hybrid: S2DD + M-CBPE ──────────────────────────────────────────────

def build_features(distances, predictions=None, per_chain_dists=None):
    """Build feature matrix for DRE from S2DD distances and optionally
    model predictions and per-chain distances.

    Args:
        distances: (n,) S2DD aggregate distances
        predictions: (n,) model predicted probabilities (optional)
        per_chain_dists: (n, C) per-chain distances (optional)

    Returns:
        features: (n, d) feature matrix
    """
    features = [distances.reshape(-1, 1)]
    if predictions is not None:
        features.append(predictions.reshape(-1, 1))
    if per_chain_dists is not None:
        features.append(per_chain_dists)
    return np.hstack(features)


def hybrid_binned_evaluation(eval_df, calibrated_probs, metric, bin_num):
    """Evaluate per-bin metrics using both raw and calibrated predictions.

    Args:
        eval_df: DataFrame with 'label', 'pred', 'distance' columns
        calibrated_probs: calibrated probabilities from M-CBPE
        metric: metric name
        bin_num: number of distance bins

    Returns:
        dict with bin_dists, raw_perfs, calibrated_perfs, raw_r, calib_r
    """
    from General_Eval.general_evaluator import safe_metric

    sorted_idx = eval_df['distance'].argsort().values
    sorted_df = eval_df.iloc[sorted_idx].reset_index(drop=True)
    sorted_calib = calibrated_probs[sorted_idx]

    bs = len(sorted_df) // bin_num
    bin_dists = []
    raw_perfs = []
    calib_perfs = []

    for i in range(bin_num):
        s = i * bs
        e = len(sorted_df) if i == bin_num - 1 else (i + 1) * bs
        bd = sorted_df.iloc[s:e]
        bc = sorted_calib[s:e]

        bin_dists.append(bd['distance'].mean())
        raw_perfs.append(safe_metric(metric, bd['label'].values, bd['pred'].values))
        calib_perfs.append(safe_metric(metric, bd['label'].values, bc))

    bin_dists = np.array(bin_dists)
    raw_perfs = np.array(raw_perfs)
    calib_perfs = np.array(calib_perfs)

    # Correlations (filter NaN)
    valid_raw = ~np.isnan(raw_perfs)
    valid_cal = ~np.isnan(calib_perfs)

    raw_r = raw_p = np.nan
    calib_r = calib_p = np.nan

    if valid_raw.sum() >= 3:
        raw_r, raw_p = pearsonr(bin_dists[valid_raw], raw_perfs[valid_raw])
    if valid_cal.sum() >= 3:
        calib_r, calib_p = pearsonr(bin_dists[valid_cal], calib_perfs[valid_cal])

    return {
        'bin_dists': bin_dists,
        'raw_perfs': raw_perfs,
        'calibrated_perfs': calib_perfs,
        'raw_r': raw_r, 'raw_p': raw_p,
        'calib_r': calib_r, 'calib_p': calib_p,
    }
