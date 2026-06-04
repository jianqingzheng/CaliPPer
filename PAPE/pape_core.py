"""PAPE: Probabilistic Adaptive Performance Estimation.

Faithful reimplementation of Białek, Kivimäki, Kuberski, Perrakis.
"Estimating Model Performance Under Covariate Shift Without Labels."
arXiv:2401.08348v5 (NeurIPS 2025).

Core algorithm (Section 3.3, Eqs. 3-4):
  1. DRE — fit binary classifier h*(x) on (source=0, target=1) pooled features,
     compute importance weights  w_{s→t}(x) = (n_s/n_t) · h*(x)/(1-h*(x)).
  2. Weighted calibration — fit regressor c(f(x)) on (source predictions,
     source labels) using w_{s→t}(x) as sample weights. This learns the
     calibration mapping UNDER the target distribution. c is not constrained
     to be monotonic — calibrated scores are used only for performance
     estimation, not for re-ranking.
  3. Performance estimate — for any composable metric m,
        m_hat = E_{p_t(x)}[ c(f(x)) · m(y_hat(x), 1)
                          + (1-c(f(x))) · m(y_hat(x), 0) ]
     approximated by the sample mean.

PAPE differs from M-CBPE (Białek & Białek 2024, arXiv:2410.11538, which our
existing MCBPE/mcbpe_core.py implements) in two choices:
  - DRE classifier: gradient-boosted (GBM) instead of logistic regression
  - Calibration model: GBM regressor instead of isotonic regression
The paper specifies LightGBM with default hyperparameters; we use sklearn's
GradientBoostingClassifier/Regressor as a drop-in substitute when LightGBM
is unavailable.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# ── Optional LightGBM backend ────────────────────────────────────────────

def _make_dre_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(verbose=-1, random_state=42), 'lightgbm'
    except ImportError:
        return (GradientBoostingClassifier(random_state=42, n_estimators=100),
                'sklearn_gbm')


def _make_calibration_regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(verbose=-1, random_state=42), 'lightgbm'
    except ImportError:
        return (GradientBoostingRegressor(random_state=42, n_estimators=100),
                'sklearn_gbm')


# ── Step 1: Density Ratio Estimation ────────────────────────────────────

def estimate_importance_weights(X_src, X_tgt):
    """Fit DRE classifier h* and return per-source-sample weights.

    Per Eq. 3:  w_{s→t}(x_i^s) = (n_s / n_t) · h*(x_i^s) / (1 - h*(x_i^s))

    Args:
        X_src: (n_s, d) source (reference) features
        X_tgt: (n_t, d) target (production) features

    Returns:
        w_src: (n_s,) importance weights for source samples
        dre_model: fitted binary classifier (for inspection)
        backend: 'lightgbm' or 'sklearn_gbm'
    """
    n_s, n_t = len(X_src), len(X_tgt)
    X = np.vstack([X_src, X_tgt])
    z = np.concatenate([np.zeros(n_s), np.ones(n_t)]).astype(int)

    model, backend = _make_dre_classifier()
    model.fit(X, z)
    p_tgt = model.predict_proba(X_src)[:, 1]
    p_tgt = np.clip(p_tgt, 1e-6, 1 - 1e-6)
    w_src = (n_s / n_t) * p_tgt / (1.0 - p_tgt)
    return w_src, model, backend


# ── Step 2: Weighted Calibration ────────────────────────────────────────

def fit_weighted_calibration(preds_src, y_src, weights_src):
    """Fit calibration mapping c: [0,1] → [0,1] as GBM regressor.

    Unlike isotonic regression (M-CBPE), the paper specifies LGBM
    regression. Calibrated scores are not required to be monotonic in the
    raw scores because they are used only for performance estimation, not
    for re-ranking predictions (Section 3.3, last paragraph).

    Args:
        preds_src: (n_s,) source model predictions in [0, 1]
        y_src: (n_s,) source binary labels
        weights_src: (n_s,) importance weights from DRE

    Returns:
        calibrator: fitted regressor
    """
    X = preds_src.reshape(-1, 1)
    model, _backend = _make_calibration_regressor()
    model.fit(X, y_src, sample_weight=weights_src)
    return model


def apply_calibration(calibrator, preds):
    """Apply fitted calibrator; clip to [0, 1]."""
    c = calibrator.predict(preds.reshape(-1, 1))
    return np.clip(c, 0.0, 1.0)


# ── Step 3: Metric estimation from calibrated probabilities ─────────────

def estimate_metric(calibrated_probs, raw_preds, metric_name, threshold=0.5):
    """Estimate metric m̂ = E[c·m(ŷ,1) + (1-c)·m(ŷ,0)] (Eq. 4).

    For AUROC (non-composable), we use the expected-label Mann-Whitney U:
        AUROC ≈ Σ_{i,j} c_i·(1-c_j)·I(f_i > f_j) / Σ_{i,j} c_i·(1-c_j)
    using raw predictions f as the ranking scores. This is the expected
    AUROC under the posterior c_i = P(y_i=1 | x_i).

    Args:
        calibrated_probs: (n,) c(f(x)) values in [0, 1]
        raw_preds: (n,) raw model scores (used for AUROC ranking and
            threshold-based metrics)
        metric_name: 'aucroc', 'ap', 'f1', 'acc', 'mcc'
        threshold: decision threshold (default 0.5 per PAPE paper § 4)

    Returns:
        scalar estimate
    """
    c = np.asarray(calibrated_probs)
    f = np.asarray(raw_preds)

    if metric_name == 'aucroc':
        return _expected_auroc(c, f)
    if metric_name == 'ap':
        return _expected_ap(c, f)
    if metric_name == 'acc':
        y_hat = (f >= threshold).astype(int)
        # E[I(ŷ=y)] = c·I(ŷ=1) + (1-c)·I(ŷ=0)
        return float(np.mean(c * y_hat + (1.0 - c) * (1 - y_hat)))
    if metric_name == 'f1':
        y_hat = (f >= threshold).astype(int)
        e_tp = float(np.sum(c * y_hat))
        e_fp = float(np.sum((1.0 - c) * y_hat))
        e_fn = float(np.sum(c * (1 - y_hat)))
        prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
        return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
    if metric_name == 'mcc':
        y_hat = (f >= threshold).astype(int)
        e_tp = float(np.sum(c * y_hat))
        e_fp = float(np.sum((1.0 - c) * y_hat))
        e_fn = float(np.sum(c * (1 - y_hat)))
        e_tn = float(np.sum((1.0 - c) * (1 - y_hat)))
        num = e_tp * e_tn - e_fp * e_fn
        den = np.sqrt((e_tp + e_fp) * (e_tp + e_fn) *
                      (e_tn + e_fp) * (e_tn + e_fn))
        return float(num / den) if den > 0 else 0.0
    if metric_name == 'brier':
        # E[Brier] = E[(f - y)^2] = mean(c*(f-1)^2 + (1-c)*f^2)
        return float(np.mean(c * (f - 1.0)**2 + (1.0 - c) * f**2))
    if metric_name == 'bss':
        brier = float(np.mean(c * (f - 1.0)**2 + (1.0 - c) * f**2))
        prev = float(c.mean())
        brier_clim = prev * (1 - prev)
        return float(1 - brier / brier_clim) if brier_clim > 0 else np.nan
    if metric_name == 'ppv' or metric_name == 'prec':
        y_hat = (f >= threshold).astype(int)
        e_tp = float(np.sum(c * y_hat))
        e_fp = float(np.sum((1.0 - c) * y_hat))
        return float(e_tp / (e_tp + e_fp)) if (e_tp + e_fp) > 0 else 0.0
    if metric_name == 'npv':
        y_hat = (f >= threshold).astype(int)
        e_tn = float(np.sum((1.0 - c) * (1 - y_hat)))
        e_fn = float(np.sum(c * (1 - y_hat)))
        return float(e_tn / (e_tn + e_fn)) if (e_tn + e_fn) > 0 else 0.0
    raise ValueError(f"Unknown metric: {metric_name}")


def _expected_auroc(c, f):
    """Expected AUROC with soft labels c and ranking by raw scores f.

    AUROC = Σ_{i,j} c_i · (1-c_j) · I(f_i > f_j) / (E[#pos] · E[#neg])

    Iterating descending by f: at each sample, the samples already seen
    have STRICTLY HIGHER f. Treat current as a soft negative; it is
    "beaten" by expected positives among those seen. Accumulate
    (1-c_current) · cum_pos to the numerator, then add c_current to
    cum_pos for the next iteration.
    """
    if len(c) < 2:
        return 0.5
    order = np.argsort(-f)  # descending by raw score
    c_sorted = c[order]
    num = 0.0
    cum_pos = 0.0  # expected positives with higher score so far
    for ci in c_sorted:
        num += (1.0 - ci) * cum_pos
        cum_pos += ci
    e_pos = float(c.sum())
    e_neg = float((1.0 - c).sum())
    if e_pos == 0.0 or e_neg == 0.0:
        return 0.5
    return num / (e_pos * e_neg)


def _expected_ap(c, f, n_thresholds=100):
    """Expected AP via threshold sweep over raw scores f with soft labels c."""
    if len(c) < 2:
        return 0.0
    e_pos_total = float(c.sum())
    if e_pos_total == 0.0:
        return 0.0
    thresholds = np.unique(
        np.quantile(f, np.linspace(0, 1, n_thresholds + 1)[1:-1]))
    precisions, recalls = [1.0], [0.0]
    for t in sorted(thresholds, reverse=True):
        pos = f >= t
        e_tp = float(c[pos].sum())
        e_fp = float((1.0 - c[pos]).sum())
        prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        rec = e_tp / e_pos_total
        precisions.append(prec)
        recalls.append(rec)
    precisions.append(0.0)
    recalls.append(1.0)
    order = np.argsort(recalls)
    recalls = np.array(recalls)[order]
    precisions = np.array(precisions)[order]
    return float(np.clip(np.trapz(precisions, recalls), 0.0, 1.0))


# ── Full PAPE Pipeline ─────────────────────────────────────────────────

def pape_estimate(ref_predictions, ref_labels, prod_predictions,
                  ref_features, prod_features,
                  metrics=('aucroc', 'ap', 'f1'),
                  threshold=0.5):
    """Run the full PAPE pipeline (Białek et al. 2024, arxiv 2401.08348).

    Args:
        ref_predictions: (n_s,) model probs on source (reference) data
        ref_labels: (n_s,) ground truth on source data
        prod_predictions: (n_t,) model probs on target (production) data
        ref_features: (n_s, d) source covariates for DRE
        prod_features: (n_t, d) target covariates for DRE
        metrics: tuple of metric names to estimate
        threshold: decision threshold for threshold-based metrics

    Returns:
        dict with keys:
            'estimated': {metric: scalar}
            'calibrated_probs': (n_t,) c(f(x)) on target
            'ref_weights': (n_s,) importance weights
            'backend': which backend was used ('lightgbm' or 'sklearn_gbm')
    """
    ref_predictions = np.asarray(ref_predictions, dtype=float)
    prod_predictions = np.asarray(prod_predictions, dtype=float)
    ref_labels = np.asarray(ref_labels, dtype=int)
    ref_features = np.asarray(ref_features, dtype=float)
    prod_features = np.asarray(prod_features, dtype=float)

    # Step 1: DRE → importance weights
    w_ref, dre_model, backend = estimate_importance_weights(
        ref_features, prod_features)

    # Step 2: Weighted calibration on source
    calibrator = fit_weighted_calibration(ref_predictions, ref_labels, w_ref)

    # Step 3: Apply to target
    c_prod = apply_calibration(calibrator, prod_predictions)

    # Step 4: Metric estimation on target
    estimated = {}
    for m in metrics:
        estimated[m] = estimate_metric(c_prod, prod_predictions, m,
                                        threshold=threshold)

    return {
        'estimated': estimated,
        'calibrated_probs': c_prod,
        'ref_weights': w_ref,
        'dre_model': dre_model,
        'calibrator': calibrator,
        'backend': backend,
    }
