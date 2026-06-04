"""S2DD v2.7 — Extended v2.6 with dual curve fitting + adaptive bin_num.

NEW in v2.7 (vs v2.6):
    1. Right-side Gaussian curve as an alternative fit for convex decay patterns:
            y = a·exp(−(d − d0)² / (2σ²)) + c + β·mp
       Fitted alongside the exponential-decay curve, and the better-R² one is
       selected automatically per (test set, metric) pair.

    2. Adaptive bin_num in predict_metric() and fit_recalibration():
            n_bins = max(4, min(8, n_minority // 8))
       where n_minority = min(n_pos_cal, n_neg_cal). min_per_bin=8 ensures ≥8
       minority-class samples per bin (matches manuscript Methods main.tex
       L376/L453/L498/L511/L531 + supplementary.tex L406). Restores Fix 1
       (commit 3b6946d0, 2026-05-11) after accidental revert in commit
       50b7b16d (2026-05-14). apply_recalibration also uses adaptive n_bins.

    3. Joint curve + β vbias correction in predict_metric():
       Residual (actual − PAPE) is fitted with fit_best_curve (exp or Gaussian)
       jointly optimizing curve params + β·mp, with λ only on β.
       Replaces v2.6's separate linear residual regression which applied λ to
       all params and lost the curve shape.

Unchanged from v2.6:
    - Bayesian recalibration (PPV/NPV sigmoid) — same protocol
    - DRE-weighted calibrator, per-test-set binning, PAPE Eq.4
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Sequence
from scipy.optimize import minimize

from .general_evaluator import safe_metric
from .combine_first_helpers import (
    fit_ridge_vbias,
    predict_ridge_vbias,
    CALIBRATION_LAM,
    PERFORMANCE_PREDICTION_LAM,
    get_s2dd_config,
    compute_combine_first_distances,
)
from PAPE.pape_core import (
    estimate_importance_weights,
    fit_weighted_calibration,
    apply_calibration,
    estimate_metric as pape_eq4,
)

eps = 1e-7
_logit = lambda x: np.log(np.clip(x, eps, 1-eps) / np.clip(1-x, eps, 1-eps))
_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

VBIAS_BETA_LAM = 0.05  # L2 on β only; safe for both dataset-level and subset-level
MIN_BIN_SAMPLES = 30   # minimum samples per bin/subset for curve fitting

# ══════════════════════════════════════════════════════════════════════
# NEW: Right-side Gaussian curve fitting
# ══════════════════════════════════════════════════════════════════════

RIGHT_GAUSSIAN_BOUNDS = (
    [-2.0, -10.0, 0.1, -1.0, -5.0],   # [a_lo, d0_lo, sigma_lo, c_lo, beta_lo]
    [ 2.0,  10.0, 20.0, 1.0,  5.0],
)
RIGHT_GAUSSIAN_INIT_GRID = [
    (0.3,  -4.0,  1.0, 0.5, 0.0),
    (0.5,  -3.0,  2.0, 0.3, 0.0),
    (0.4,  -5.0,  3.0, 0.4, 0.0),
    (0.2,  -2.0,  0.5, 0.6, 0.0),
    (0.6,  -6.0,  5.0, 0.2, 0.0),
]


def fit_right_gaussian(d, mp, y, lam=0.0):
    """[v2.7] Fit right-side Gaussian curve:
        y = a · exp(−(d − d0)² / (2σ²)) + c + β·mp

    The Gaussian peak at d0 defines a plateau; decay for d > d0 matches
    convex/sigmoid-like performance drops (performance stays high then
    sharply drops at some distance threshold). This shape cannot be
    captured by a monotonic exp-decay.

    Args:
        d, mp, y: per-bin distance / mean-prediction / metric arrays
        lam: L2 regularisation on β (default 0.0, matches PERFORMANCE_PREDICTION_LAM)

    Returns:
        params: tuple(a, d0, sigma, c, beta) or None if fit fails
    """
    d = np.asarray(d, dtype=float)
    mp = np.asarray(mp, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y) & ~np.isnan(d) & ~np.isnan(mp)
    if valid.sum() < 4:
        return None
    dv, mpv, yv = d[valid], mp[valid], y[valid]

    def loss(params):
        a, d0, sigma, c, beta = params
        pred = a * np.exp(-((dv - d0) ** 2) / (2 * sigma ** 2)) + c + beta * mpv
        return float(np.mean((yv - pred) ** 2) + lam * beta ** 2)

    lo, hi = RIGHT_GAUSSIAN_BOUNDS
    bounds = list(zip(lo, hi))

    best_params = None
    best_loss = np.inf
    for x0 in RIGHT_GAUSSIAN_INIT_GRID:
        try:
            res = minimize(loss, x0=list(x0), method='L-BFGS-B', bounds=bounds)
            if res.success and res.fun < best_loss:
                best_params = tuple(res.x)
                best_loss = res.fun
        except Exception:
            continue
    return best_params


def predict_right_gaussian(params, d, mp, lo=0.0, hi=1.0):
    """Apply right-Gaussian prediction. Returns NaN-filled array if params is None."""
    if params is None:
        return np.full_like(np.asarray(d, dtype=float), np.nan, dtype=float)
    a, d0, sigma, c, beta = params
    d = np.asarray(d, dtype=float)
    mp = np.asarray(mp, dtype=float)
    return np.clip(a * np.exp(-((d - d0) ** 2) / (2 * sigma ** 2)) + c + beta * mp,
                    lo, hi)


# ══════════════════════════════════════════════════════════════════════
# NEW: Dual-curve fitting with automatic selection
# ══════════════════════════════════════════════════════════════════════

def fit_best_curve(d, mp, y, lam=0.0):
    """[v2.7] Fit BOTH exp-decay and right-Gaussian curves; return the better-R² one.

    Concave decay (typical): exp-decay wins
    Convex decay (plateau-then-drop): right-Gaussian wins

    Returns:
        dict with keys:
            kind : 'exp' or 'gauss'
            params : tuple (fit parameters)
            r2 : float (R² of selected curve)
            r2_exp, r2_gauss : individual R² for both fits (for audit)
    """
    d = np.asarray(d, dtype=float)
    mp = np.asarray(mp, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y) & ~np.isnan(d) & ~np.isnan(mp)
    if valid.sum() < 4:
        return {'kind': None, 'params': None, 'r2': np.nan,
                'r2_exp': np.nan, 'r2_gauss': np.nan}

    dv, mpv, yv = d[valid], mp[valid], y[valid]

    # Try both fits
    exp_params = fit_ridge_vbias(dv, mpv, yv, lam=lam)
    gauss_params = fit_right_gaussian(dv, mpv, yv, lam=lam)

    def _r2(pred):
        ss_res = np.sum((yv - pred) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot

    if exp_params is not None:
        r2_exp = _r2(predict_ridge_vbias(exp_params, dv, mpv))
    else:
        r2_exp = -np.inf

    if gauss_params is not None:
        r2_gauss = _r2(predict_right_gaussian(gauss_params, dv, mpv))
    else:
        r2_gauss = -np.inf

    # Parsimony: prefer exp-decay (4 params: a, b, c, β) over Gaussian (5 params: a, d0, σ, c, β)
    # unless Gaussian's R² is meaningfully higher (Δ > 0.02, ~2% variance explained).
    # This prevents Gaussian overfitting on monotonic data where exp-decay is correct.
    R2_IMPROVEMENT_THRESHOLD = 0.02

    # Monotonicity guard: a right-side Gaussian must have d0 ≤ d_min
    # (peak at or left of data) so that only the right-side decay is visible.
    # Any d0 inside the data range shows the left-side rise → non-monotonic.
    gauss_monotonic = True
    if gauss_params is not None:
        _, d0, _, _, _ = gauss_params
        d_min = dv.min()
        if d0 > d_min:
            gauss_monotonic = False  # peak inside data → left-side rise visible

    if (gauss_monotonic and r2_gauss > r2_exp + R2_IMPROVEMENT_THRESHOLD):
        return {'kind': 'gauss', 'params': gauss_params, 'r2': r2_gauss,
                'r2_exp': r2_exp, 'r2_gauss': r2_gauss}
    else:
        return {'kind': 'exp', 'params': exp_params, 'r2': r2_exp,
                'r2_exp': r2_exp, 'r2_gauss': r2_gauss}


def predict_best_curve(fit_result, d, mp, lo=0.0, hi=1.0):
    """Apply selected curve from fit_best_curve() output."""
    if fit_result['kind'] == 'exp':
        return predict_ridge_vbias(fit_result['params'], d, mp, lo=lo, hi=hi)
    elif fit_result['kind'] == 'gauss':
        return predict_right_gaussian(fit_result['params'], d, mp, lo=lo, hi=hi)
    else:
        return np.full_like(np.asarray(d, dtype=float), np.nan, dtype=float)


# ══════════════════════════════════════════════════════════════════════
# NEW: Adaptive bin_num
# ══════════════════════════════════════════════════════════════════════

def adaptive_n_bins(n_pos, n_neg, max_bins=8, min_bins=4, min_per_bin=8):
    """[v2.7] Compute adaptive bin count from class balance.

    Formula:
        n_minority = min(n_pos, n_neg)
        n_bins = max(min_bins, min(max_bins, n_minority // min_per_bin))

    Rationale: ensures ≥`min_per_bin` minority-class samples per bin for
    stable metric estimation and curve fitting. With min_per_bin=8 (matches
    manuscript Methods L376/L453/L498/L511/L531 + supplementary L406):
    AntibioticsAI cross-dataset (33 positives) → 4 bins → 8 pos/bin →
    no NaN bins, AP error drops from 0.094 to 0.010.
    Large datasets (TCR/BCR with 1000+ positives) are unaffected (capped at 8).
    """
    n_minority = min(int(n_pos), int(n_neg))
    return max(min_bins, min(max_bins, n_minority // min_per_bin))


# ══════════════════════════════════════════════════════════════════════
# PREDICT_METRIC — v2.7 with adaptive bins + dual curve
# ══════════════════════════════════════════════════════════════════════

def predict_metric(cal_data: Dict[str, Tuple],
                    test_p: np.ndarray,
                    test_d: np.ndarray,
                    metrics: Sequence[str] = ('aucroc', 'ap', 'f1'),
                    n_bins: int | None = None,
                    threshold: float = 0.5,
                    adaptive_bins: bool = True,
                    train_anchor: dict | None = None) -> dict:
    """[S2DD v2.7] Predict aggregate performance metrics on an unlabelled test set.

    Extends v2.6 with:
        1. Adaptive n_bins from class balance (when adaptive_bins=True, default)
        2. Dual curve fitting (exp-decay + right-Gaussian) for vanilla curve
           output; the better-R² curve is auto-selected per metric

    The main PAPE+vbias prediction path is unchanged from v2.6 (still uses
    ridge residual correction); the change is in (a) how many bins the cal
    data is divided into, and (b) the vanilla-curve baseline output.

    Args:
        cal_data: dict{name: (y, pred, distance)} — calibration test sets
        test_p, test_d: unlabelled test set predictions + distances
        metrics: metrics to estimate
        n_bins: explicit bin count; if None and adaptive_bins=True, computed
        threshold: binarisation threshold (default 0.5)
        adaptive_bins: if True, compute n_bins from n_minority in cal (default True)
        train_anchor: optional dict with keys {metrics: {metric: value}, mp, distance}.
            Injects the training set's known performance as one fixed data point
            in the vbias curve fitting. Same concept as fit_recalibration's
            train_anchor but for performance prediction instead of PPV/NPV.
            Compute from full training set:
                anchor = dict(
                    metrics = {m: safe_metric(m, train_y, train_p) for m in metrics},
                    mp = train_p.mean(),
                    distance = (0 - mu_chain) / sigma_chain  # z-normed self-distance
                )

    Returns:
        dict with:
            'estimated': {m: v2.7 PAPE+vbias prediction}
            'estimated_vanilla': {m: dual-curve vanilla prediction}
            'curve_info': {m: {kind, r2_exp, r2_gauss, n_bins_used}}
    """
    cal_y = np.concatenate([v[0] for v in cal_data.values()])
    cal_p = np.concatenate([v[1] for v in cal_data.values()])
    cal_d = np.concatenate([v[2] for v in cal_data.values()])

    # Adaptive bin count
    if adaptive_bins and n_bins is None:
        n_pos = int((cal_y == 1).sum())
        n_neg = int((cal_y == 0).sum())
        n_bins_used = adaptive_n_bins(n_pos, n_neg)
    elif n_bins is None:
        n_bins_used = 8
    else:
        n_bins_used = int(n_bins)

    # DRE calibrator (v2.6 Step 1-2)
    cal_feats = np.stack([cal_d, cal_p], axis=1)
    test_feats = np.stack([test_d, test_p], axis=1)
    w_dre, _, _ = estimate_importance_weights(cal_feats, test_feats)
    cal_model = fit_weighted_calibration(cal_p, cal_y, w_dre)
    c_cal = apply_calibration(cal_model, cal_p)
    c_test = apply_calibration(cal_model, test_p)

    # Per-cal-test-set bins (v2.6 Step 3)
    cal_set_ranges = []
    offset = 0
    for name, (y_s, p_s, d_s) in cal_data.items():
        cal_set_ranges.append((offset, offset + len(y_s), name))
        offset += len(y_s)

    # Match v2.6 min_samples=30 threshold to guarantee strict backward
    # compatibility.  Adaptive bins can only *help* (when bs ≥ 30 at the
    # chosen n_bins) — never enables vbias correction on datasets too
    # small for stable per-bin metrics.

    estimated = {}
    estimated_vanilla = {}
    curve_info = {}

    for m in metrics:
        bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []

        # Inject training anchor as one fixed data point (before cal bins)
        # Same concept as fit_recalibration's train_anchor
        if train_anchor is not None and m in train_anchor.get('metrics', {}):
            anchor_actual = float(train_anchor['metrics'][m])
            # At training distance, PAPE should also predict near-perfect → residual ≈ 0
            # Use actual as pape estimate for the anchor (conservative: no correction needed)
            bin_d.append(float(train_anchor['distance']))
            bin_mp.append(float(train_anchor['mp']))
            bin_actual.append(anchor_actual)
            bin_pape.append(anchor_actual)  # residual = 0 at anchor

        for start, end, ts_name in cal_set_ranges:
            yi = cal_y[start:end]
            fi = cal_p[start:end]
            di = cal_d[start:end]
            ci = c_cal[start:end]

            si = np.argsort(di)
            bs = max(len(si) // n_bins_used, 1)
            if bs < MIN_BIN_SAMPLES:
                continue

            for i in range(n_bins_used):
                s = i * bs
                e = len(si) if i == n_bins_used - 1 else (i + 1) * bs
                idx = si[s:e]
                actual_m = safe_metric(m, yi[idx], fi[idx])
                pape_m = pape_eq4(ci[idx], fi[idx], m, threshold=threshold)
                if np.isnan(actual_m) or np.isnan(pape_m):
                    continue
                bin_d.append(di[idx].mean())
                bin_mp.append(fi[idx].mean())
                bin_actual.append(actual_m)
                bin_pape.append(pape_m)

        bin_d = np.array(bin_d)
        bin_mp = np.array(bin_mp)
        bin_actual = np.array(bin_actual)
        bin_pape = np.array(bin_pape)

        # Vbias correction (v2.7): joint curve + β regression on PAPE residuals.
        # Fits residual = a·f(d) + c + β·mp with λ only on β (not on curve params).
        # Uses fit_best_curve (dual exp/Gaussian) for the curve shape.
        residual = bin_actual - bin_pape
        pape_test = pape_eq4(c_test, test_p, m, threshold=threshold)
        mp_test = test_p.mean()
        d_test = test_d.mean()

        if len(residual) >= 4:
            fit_res = fit_best_curve(bin_d, bin_mp, residual,
                                      lam=VBIAS_BETA_LAM)
            if fit_res['params'] is not None:
                correction = float(predict_best_curve(
                    fit_res, np.array([d_test]), np.array([mp_test]))[0])
            else:
                correction = 0.0
        else:
            correction = 0.0
        estimated[m] = float(np.clip(pape_test + correction, 0, 1))

        # v2.7 NEW: Dual-curve vanilla output
        valid = ~np.isnan(bin_actual)
        if valid.sum() >= 4:
            fit_result = fit_best_curve(bin_d[valid], bin_mp[valid], bin_actual[valid],
                                         lam=PERFORMANCE_PREDICTION_LAM)
            if fit_result['params'] is not None:
                # Apply selected curve to test bins (weighted by bin size)
                si_t = np.argsort(test_d)
                bs_t = max(len(si_t) // n_bins_used, 1)
                wsum, ntot = 0.0, 0
                for i in range(n_bins_used):
                    s = i * bs_t
                    e = len(si_t) if i == n_bins_used - 1 else (i + 1) * bs_t
                    idx = si_t[s:e]
                    pred = float(predict_best_curve(
                        fit_result,
                        np.array([test_d[idx].mean()]),
                        np.array([test_p[idx].mean()]),
                    )[0])
                    wsum += len(idx) * pred
                    ntot += len(idx)
                estimated_vanilla[m] = wsum / ntot if ntot > 0 else np.nan
            else:
                estimated_vanilla[m] = np.nan
            curve_info[m] = {
                'kind': fit_result['kind'],
                'r2': fit_result['r2'],
                'r2_exp': fit_result['r2_exp'],
                'r2_gauss': fit_result['r2_gauss'],
                'n_bins_used': n_bins_used,
                'params': fit_result['params'],
            }
        else:
            estimated_vanilla[m] = np.nan
            curve_info[m] = {
                'kind': None, 'r2': np.nan,
                'r2_exp': np.nan, 'r2_gauss': np.nan,
                'n_bins_used': n_bins_used, 'params': None,
            }

    return {
        'estimated': estimated,
        'estimated_vanilla': estimated_vanilla,
        'curve_info': curve_info,
        'n_bins_used': n_bins_used,
    }


# ══════════════════════════════════════════════════════════════════════
# DISTANCE COMPUTATION — strategy-aware convenience function
# ══════════════════════════════════════════════════════════════════════

def compute_s2dd_distances(test_df, train_df, chain_cols, k, b, K,
                           task='subset', subsample=500):
    """Compute S2DD distances with task-appropriate combining strategy.

    Selects the optimal combining strategy based on the downstream task:
      - task='subset': uniform + znorm_sum (per_epitope strategy)
            Best for per-epitope/per-antigen subset-level prediction.
            Uniform weighting gives equal influence to all chains, preventing
            dominant chains (e.g., peptide=91% in sigma_C) from drowning
            sequence-level variation important for subset differentiation.
      - task='degradation': sigma_C + weighted_max_znorm (degradation strategy)
            Best for bin-level degradation curves (bin_R²=0.791).

    Verified 2026-04-28:
      TCR per-epitope: uniform wins all 6 cells (+0.007 to +0.049 r vs sigma_C)
      BCR per-antigen: sigma_C wins all 6 cells (-0.002 to -0.020 r vs uniform)
      For BCR, sigma_C is better because variant_seq carries ~99.5% weight and
      IS the antigen signal; for TCR, uniform is better because peptide dominance
      (~91%) drowns CDR3 variation needed to differentiate epitopes.

    Args:
        test_df: DataFrame with chain columns for test samples
        train_df: DataFrame with chain columns for training/reference samples
        chain_cols: list of column names for sequence chains
        k, b, K: LogDist parameters
        task: 'subset' (default) or 'degradation'
        subsample: number of training samples for chain stat computation

    Returns:
        1-D numpy array of per-test-sample S2DD distances
    """
    import warnings
    # Check if short sequences would benefit from BLOSUM-sqrt instead of Levenshtein
    for col in chain_cols:
        sample = test_df[col].astype(str).head(100)
        mean_len = sample.str.len().mean()
        if mean_len <= 20:
            warnings.warn(
                f"Levenshtein-log on short sequences ({col}: mean {mean_len:.0f} AA). "
                f"BLOSUM-sqrt captures biochemical similarity that Levenshtein misses "
                f"for short sequences (≤20 AA), improving recalibration by +0.02 to "
                f"+0.08 ΔAUROC. Consider compute_s2dd_pluggable() with BLOSUM-SW "
                f"similarity and transform='sqrt'. "
                f"See feedback_blosum_vs_lev_limitation.md for evidence.",
                UserWarning, stacklevel=2)
            break  # One warning is enough

    strategy = 'per_epitope' if task == 'subset' else 'degradation'
    weights, combine_method, _ = get_s2dd_config(
        train_df, chain_cols, k, b, K, strategy=strategy, subsample=subsample)
    return compute_combine_first_distances(
        test_df, train_df, chain_cols, weights, k, b, K,
        combine_method=combine_method)


# ══════════════════════════════════════════════════════════════════════
# SUBSET-LEVEL PREDICTION — shared function for fig4 scripts
# ══════════════════════════════════════════════════════════════════════

def predict_subset_metric(cal_data: Dict[str, Tuple],
                           test_subsets: Dict[str, Tuple],
                           metrics: Sequence[str] = ('aucroc', 'ap', 'f1'),
                           n_bins: int | None = None,
                           threshold: float = 0.5,
                           adaptive_bins: bool = True,
                           bin_strategy: str = 'distance',
                           cal_subsets: Dict[str, Tuple] | None = None) -> list:
    """[S2DD v2.7] Predict per-subset performance on held-out test data.

    Encapsulates the full PAPE + joint curve+β vbias pipeline for subset-level
    prediction. Scripts should call this instead of reimplementing the pipeline.

    IMPORTANT — distance strategy:
        When bin_strategy='subset' (epitope/variant prediction), input distances
        should be computed with ``compute_s2dd_distances(task='subset')`` which
        uses uniform+znorm_sum combining. This improves TCR per-epitope prediction
        by +0.049 r vs sigma_C (verified 2026-04-28). For BCR per-antigen,
        sigma_C is better (-0.014 r), so use ``task='degradation'`` or the default
        sigma_C distances.

    Args:
        cal_data: dict{name: (y, pred, distance)} — calibration test sets
        test_subsets: dict{subset_name: (y, pred, distance)} — test subsets to predict
        metrics: metrics to estimate per subset
        n_bins: explicit bin count; if None and adaptive_bins=True, computed
        threshold: binarisation threshold (default 0.5)
        adaptive_bins: if True, compute n_bins from n_minority in cal
        bin_strategy: 'distance' (default) — bin cal data by distance for curve fitting.
            'subset' — bin cal data by subset identity (one bin per cal_subset).
            Use 'subset' for epitope/variant subset prediction; 'distance' for
            distance-stratified prediction.
        cal_subsets: dict{subset_name: (y, pred, distance)} — cal data split by
            the same subset definition as test_subsets. Required when
            bin_strategy='subset'. Each entry is one regression point for curve
            fitting. Must have >= 4 entries with valid metrics.

    Returns:
        list of dicts with keys: subset, metric, predicted, actual, n, prevalence
    """
    cal_y = np.concatenate([v[0] for v in cal_data.values()])
    cal_p = np.concatenate([v[1] for v in cal_data.values()])
    cal_d = np.concatenate([v[2] for v in cal_data.values()])

    # Adaptive bin count (only used for distance binning)
    if adaptive_bins and n_bins is None:
        n_pos = int((cal_y == 1).sum())
        n_neg = int((cal_y == 0).sum())
        n_bins_used = adaptive_n_bins(n_pos, n_neg)
    elif n_bins is None:
        n_bins_used = 8
    else:
        n_bins_used = int(n_bins)

    # DRE calibrator — pool all test subsets for DRE target
    all_test_p = np.concatenate([v[1] for v in test_subsets.values()])
    all_test_d = np.concatenate([v[2] for v in test_subsets.values()])
    cal_feats = np.stack([cal_d, cal_p], axis=1)
    test_feats = np.stack([all_test_d, all_test_p], axis=1)
    w_dre, _, _ = estimate_importance_weights(cal_feats, test_feats)
    cal_model = fit_weighted_calibration(cal_p, cal_y, w_dre)
    c_cal = apply_calibration(cal_model, cal_p)

    results = []
    for m in metrics:
        if bin_strategy == 'subset':
            # ── Subset-based binning: each cal_subset = one regression point ──
            if cal_subsets is None:
                raise ValueError("bin_strategy='subset' requires cal_subsets")
            bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
            for sub_name, (sy, sp, sd) in cal_subsets.items():
                if len(sy) < MIN_BIN_SAMPLES:
                    continue
                actual_m = safe_metric(m, sy, sp)
                # PAPE for this cal subset using DRE calibrator
                c_sub = apply_calibration(cal_model, sp)
                pape_m = pape_eq4(c_sub, sp, m, threshold=threshold)
                if not np.isnan(actual_m) and not np.isnan(pape_m):
                    bin_d.append(sd.mean())
                    bin_mp.append(sp.mean())
                    bin_actual.append(actual_m)
                    bin_pape.append(pape_m)
        else:
            # ── Distance-based binning (default): per-cal-set distance bins ──
            bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
            offset = 0
            for name, (y_s, p_s, d_s) in cal_data.items():
                si = np.argsort(d_s)
                bs = max(len(si) // n_bins_used, 1)
                if bs < MIN_BIN_SAMPLES:
                    offset += len(y_s)
                    continue
                for i in range(n_bins_used):
                    s = i * bs
                    e = len(si) if i == n_bins_used - 1 else (i + 1) * bs
                    idx = si[s:e]
                    idx_global = idx + offset
                    actual_m = safe_metric(m, cal_y[idx_global], cal_p[idx_global])
                    pape_m = pape_eq4(c_cal[idx_global], cal_p[idx_global], m, threshold=threshold)
                    if not np.isnan(actual_m) and not np.isnan(pape_m):
                        bin_d.append(cal_d[idx_global].mean())
                        bin_mp.append(cal_p[idx_global].mean())
                        bin_actual.append(actual_m)
                        bin_pape.append(pape_m)
                offset += len(y_s)

        bin_d = np.array(bin_d)
        bin_mp = np.array(bin_mp)
        bin_actual = np.array(bin_actual)
        bin_pape = np.array(bin_pape)

        # Joint curve+β vbias on PAPE residuals
        residual = bin_actual - bin_pape
        if len(residual) >= 4:
            fit_res = fit_best_curve(bin_d, bin_mp, residual, lam=VBIAS_BETA_LAM)
        else:
            fit_res = {'params': None}

        # Predict each test subset
        for sub_name, (sub_y, sub_p, sub_d) in test_subsets.items():
            actual_m = safe_metric(m, sub_y, sub_p)
            if np.isnan(actual_m):
                continue
            c_sub = apply_calibration(cal_model, sub_p)
            pape_sub = pape_eq4(c_sub, sub_p, m, threshold=threshold)
            if fit_res['params'] is not None:
                correction = float(predict_best_curve(
                    fit_res, np.array([sub_d.mean()]),
                    np.array([sub_p.mean()]))[0])
            else:
                correction = 0.0
            results.append({
                'subset': sub_name,
                'metric': m,
                'predicted': float(np.clip(pape_sub + correction, 0, 1)),
                'actual': actual_m,
                'n': len(sub_y),
                'prevalence': float(sub_y.mean()),
            })

    return results




# ══════════════════════════════════════════════════════════════════════
# RECALIBRATION — unchanged from v2.6
# ══════════════════════════════════════════════════════════════════════

def fit_recalibration(cal_data, n_bins=None, threshold=None, min_samples=30,
                      bin_strategy='distance', cal_subsets=None,
                      train_anchor=None):
    """[S2DD v2.7] Fit PPV/NPV curves for Bayesian recalibration.

    Vanilla PPV/NPV (no DRE, no PAPE). Uses adaptive defaults:
      - threshold=None → auto-compute adaptive theta
      - n_bins=None → adaptive max(4, min(8, n_minority // 4))

    Args:
        cal_data: dict{name: (y, pred, distance)} — calibration test sets
        n_bins: explicit bin count per cal set; None = adaptive from class balance
        threshold: binarization threshold for PPV/NPV; None = adaptive
        min_samples: minimum samples per bin
        bin_strategy: 'distance' (default) — bin by distance within each cal set.
            'subset' — bin by subset identity (one bin per cal_subset entry).
            Mirrors predict_subset_metric's bin_strategy parameter.
        cal_subsets: dict{name: (y, pred, distance)} — cal data split by
            epitope/variant. Required when bin_strategy='subset'. Each entry
            becomes one regression point for PPV/NPV curve fitting.
        train_anchor: optional dict with keys {ppv, npv, mp, distance}.
            Injects the training set's aggregate performance as one fixed
            bin-level data point in the PPV/NPV curve fitting. Use when
            the model performs well on training (e.g. AUROC>0.9) but all
            test samples are equally far from training (no within-test
            distance gradient). The anchor captures the steep train→test
            degradation that is invisible from within-test data alone.
            Compute from full training set:
                pp = pred >= threshold
                # anchor distance = z-normed identical-sequence distance
                # raw distance of identical seq = 0 (sqrt(1-1)=0 or log(k*b))
                # z-norm with same train-vs-train mu/sigma as cal/test:
                #   anchor_d = (0 - mu_chain) / sigma_chain
                anchor = dict(
                    ppv = TP / (TP + FP),
                    npv = TN / (TN + FN),
                    mp  = pred.mean(),
                    distance = (0 - mu_chain) / sigma_chain
                )

    Returns:
        ppv_params, npv_params, p_pos, p_neg, cal_prev
    """
    # Pool cal labels to compute adaptive defaults
    cal_y_all = np.concatenate([v[0] for v in cal_data.values()])
    cal_p_all = np.concatenate([v[1] for v in cal_data.values()])
    cal_prev = float(cal_y_all.mean())

    # Adaptive theta: symmetric clamp around 0.5
    if threshold is None:
        prev = float(cal_y_all.mean())
        threshold = max(2 * prev - 1, min(2 * prev, 0.5))

    bin_d, bin_mp, bin_ppv, bin_npv = [], [], [], []
    all_preds = []

    # Inject training anchor as one fixed bin (before cal bins)
    if train_anchor is not None:
        bin_d.append(float(train_anchor['distance']))
        bin_mp.append(float(train_anchor['mp']))
        bin_ppv.append(float(train_anchor['ppv']))
        bin_npv.append(float(train_anchor['npv']))

    if bin_strategy == 'subset':
        # ── Subset-based binning: each cal_subset = one regression point ──
        if cal_subsets is None:
            raise ValueError("bin_strategy='subset' requires cal_subsets")
        for sub_name, (sy, sp, sd) in cal_subsets.items():
            if len(sy) < min_samples:
                continue
            all_preds.extend(sp.tolist())
            pp = sp >= threshold
            tp = int((pp & (sy == 1)).sum())
            fp = int((pp & (sy == 0)).sum())
            tn = int(((~pp) & (sy == 0)).sum())
            fn = int(((~pp) & (sy == 1)).sum())
            bin_d.append(sd.mean())
            bin_mp.append(sp.mean())
            bin_ppv.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
            bin_npv.append(tn / (tn + fn) if tn + fn > 0 else np.nan)
    else:
        # ── Distance-based binning (default) ──
        # Adaptive bin count: from class balance, capped by smallest cal set
        if n_bins is None:
            n_pos = int((cal_y_all == 1).sum())
            n_neg = int((cal_y_all == 0).sum())
            n_bins = adaptive_n_bins(n_pos, n_neg)
            min_set_size = min(len(v[0]) for v in cal_data.values())
            max_safe_bins = max(4, min_set_size // min_samples)
            n_bins = min(n_bins, max_safe_bins)

        for ts_name, (y_ts, p_ts, d_ts) in cal_data.items():
            si = np.argsort(d_ts)
            bs = len(si) // n_bins
            if bs < min_samples:
                continue
            all_preds.extend(p_ts.tolist())
            for i in range(n_bins):
                s = i * bs
                e = len(si) if i == n_bins - 1 else (i + 1) * bs
                idx = si[s:e]
                yi, fi = y_ts[idx], p_ts[idx]
                pp = fi >= threshold
                tp = int((pp & (yi == 1)).sum())
                fp = int((pp & (yi == 0)).sum())
                tn = int(((~pp) & (yi == 0)).sum())
                fn = int(((~pp) & (yi == 1)).sum())
                bin_d.append(d_ts[idx].mean())
                bin_mp.append(fi.mean())
                bin_ppv.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
                bin_npv.append(tn / (tn + fn) if tn + fn > 0 else np.nan)

    bin_d = np.array(bin_d); bin_mp = np.array(bin_mp)
    bin_ppv = np.array(bin_ppv); bin_npv = np.array(bin_npv)
    if not all_preds:
        all_preds = cal_p_all.tolist()
    all_preds = np.array(all_preds)

    ppv_params = npv_params = None
    for vals, name in [(bin_ppv, 'ppv'), (bin_npv, 'npv')]:
        v = ~np.isnan(vals)
        if v.sum() >= 4:
            params = fit_ridge_vbias(bin_d[v], bin_mp[v], vals[v],
                                      lam=CALIBRATION_LAM)
            if name == 'ppv': ppv_params = params
            else: npv_params = params

    pos_p = all_preds[all_preds >= threshold]
    neg_p = all_preds[all_preds < threshold]
    p_pos = float(np.quantile(pos_p, 0.25)) if len(pos_p) > 0 else 0.75
    p_neg = float(np.quantile(neg_p, 0.75)) if len(neg_p) > 0 else 0.25
    return ppv_params, npv_params, p_pos, p_neg, cal_prev


def apply_recalibration(test_y, test_p, test_d,
                         ppv_params, npv_params, p_pos, p_neg,
                         n_bins=None, prev=None):
    """[S2DD v2.7] Apply Bayesian sigmoid recalibration.

    n_bins: number of bins for computing bin-level mean prediction (mp).
    If None, uses adaptive_n_bins from test_y class balance, capped to
    ensure ≥4 samples per bin on the test set. This balances matching
    the fit granularity with having enough samples per bin for stable mp.

    prev: class prevalence for the Platt scaling base rate. Should be set
    to the CALIBRATION prevalence (returned by fit_recalibration) to avoid
    test label leakage. If None, falls back to test_y.mean() for backward
    compatibility, but this uses test labels and is NOT recommended.
    """
    if n_bins is None:
        n_pos = int((test_y == 1).sum())
        n_neg = int((test_y == 0).sum())
        # Use same adaptive formula as fit, then cap by test set size
        n_bins = adaptive_n_bins(n_pos, n_neg)
        max_bins_for_test = max(4, len(test_y) // 4)
        n_bins = min(n_bins, max_bins_for_test)
    def _eval_curve(params, d, mp):
        if params is None:
            return np.full_like(d, 0.5, dtype=float)
        a, bx, c, beta = params
        return np.clip(a * np.exp(-bx * d) + c + beta * mp, 0.01, 0.99)

    if prev is None:
        prev = test_y.mean()  # backward compat; prefer cal_prev
    si = np.argsort(test_d)
    bs = max(len(si) // n_bins, 1)
    mp_per = np.zeros(len(test_d))
    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        mp_per[idx] = test_p[idx].mean()

    ppv_s = _eval_curve(ppv_params, test_d, mp_per)
    npv_s = _eval_curve(npv_params, test_d, mp_per)

    lp_pp, lp_pn = _logit(p_pos), _logit(p_neg)
    denom = lp_pp - lp_pn
    b_raw = (_logit(ppv_s) - _logit(1.0 - npv_s)) / np.where(
        np.abs(denom) > 1e-12, denom, 1e-12)
    a_raw = _logit(ppv_s) - _logit(prev) - b_raw * lp_pp
    w = np.clip(np.clip(ppv_s + npv_s - 1.0, 0.0, 1.0), 0.1, 1.0)
    cal = _sig(_logit(prev) + w * a_raw + w * b_raw * _logit(test_p))
    return np.where(np.isfinite(cal), cal, 0.5)
