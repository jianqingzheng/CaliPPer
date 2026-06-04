"""Shared helpers for combine-first LogDist distance pipeline.

All scripts that compute LogDist distances should use these functions
to ensure a consistent configuration.

Unified method for subset-level performance prediction (2026-04-12):

  Fit:         fit_ridge_vbias(d, bin_mp, y, lam=PERFORMANCE_PREDICTION_LAM)
               lam=0.0 for TCR CV/CT and BCR CV; lam=0.05 for BCR CT only
  Predict:     predict_subset_metric(params, subset_df) — avg_then_eval
               Evaluates curve at subset mean distance with subset mean pred.
               NEVER use per-sample evaluation then average (eval_then_avg).
  Curve:       a * exp(-b * d) + c + beta * mean_p_bin
  Evidence:    avg_then_eval wins 50/60 cells (mean +0.193 R vs eval_then_avg)
               Matched-granularity principle: predict at the level you fit.

Two S2DD distance strategies (weighting + combine method):

  1. 'degradation' (default) — sigma_C + weighted_max_znorm + topk
     Best for: bin-level degradation curves (bin_R²=0.791), distance-bin NPV
     prediction, BCR CT variant strategy, BCR CT AP.

  2. 'per_epitope' — uniform + znorm_sum + topk
     Best for: per-epitope/per-antigen prediction (ep_R=0.430), TCR CT
     distance-bin AUROC/AP/F1/PPV (+0.023 to +0.099 R vs sigma_C),
     TCR CT epitope strategy (all 5 metrics).

  No single weighting wins all cells (uniform 19/30, sigma_C 9/30 on CT).
  Use get_s2dd_config() to obtain (weights, combine_method) for a given strategy.

Unified method for Bayesian calibration via PPV/NPV (2026-04-12):

  The calibration pipeline uses bin-level mp at BOTH fit and predict time,
  fully unified with the performance prediction pipeline. Key settings:
    Fit:         fit_ridge_vbias(d, bin_mp, ppv/npv, lam=CALIBRATION_LAM=0.0)
                 bin_mp = per-bin mean prediction from calibration data (v3+v4)
    Threshold:   CALIBRATION_THRESHOLD = 0.5 (fixed, model-independent)
    Predict:     per-sample f(d_i, bin_mp_i) for per-sample PPV/NPV
                 bin_mp_i = mean prediction of the distance bin sample i belongs to
                 (bin held-out test set into N_SUB=8 distance bins, assign each
                 sample its bin's mean prediction)
    Blend:       linear_floor: w = max(0.1, clip(PPV+NPV-1, 0, 1))
    Evidence:    5-model TCR CT ΔAUROC=+0.016, ΔAP=+0.010 (bin predict)
                 vs test-set predict: ΔAUROC=+0.020, ΔAP=+0.014
                 Cost of unification: −0.004 AUROC (driven by TCR-BERT)
                 BCR CV at λ=0.05: identical (+0.003 both methods)
"""

import numpy as np
from collections import Counter

from .general_evaluator import (
    compute_pairwise_ratios,
    logdist_from_ratios,
    distances_combine_first_multi,
    compute_pairwise_chain_stats,
    build_sample_to_unique_index,
    compute_sample_probs,
    safe_metric,
    binned_correlations,
)


# ── S2DD Strategy Presets ──
S2DD_STRATEGIES = {
    'degradation': {
        'weight_formula': 'sigma_C',
        'combine_method': 'weighted_max_znorm',
        'description': 'Best for bin-level degradation curves (bin_R²=0.791)',
    },
    'per_epitope': {
        'weight_formula': 'uniform',
        'combine_method': 'znorm_sum',
        'description': 'Best for per-epitope/per-antigen performance prediction (ep_R=0.430)',
    },
}


def get_s2dd_config(train_df, chain_cols, k, b, K, strategy='degradation',
                    subsample=500, seed=42, use_log=True):
    """Get recommended (weights, combine_method) for a given S2DD strategy.

    Args:
        strategy: 'degradation' (default) or 'per_epitope'
            - 'degradation': sigma_C + weighted_max_znorm (best bin-level R²)
            - 'per_epitope': uniform + znorm_sum (best per-epitope prediction R)

    Returns:
        weights: 1-D array of chain weights (normalized to sum to 1)
        combine_method: str, the recommended combine function name
        chain_stds: 1-D array of per-chain LogDist std
    """
    if strategy not in S2DD_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(S2DD_STRATEGIES.keys())}")

    cfg = S2DD_STRATEGIES[strategy]
    formula = cfg['weight_formula']
    combine_method = cfg['combine_method']

    print(f"  S2DD strategy: {strategy} ({cfg['description']})")
    weights, chain_stds = compute_chain_weights(
        train_df, chain_cols, k, b, K,
        formula=formula, subsample=subsample, seed=seed, use_log=use_log)

    return weights, combine_method, chain_stds


def compute_chain_weights(train_df, chain_cols, k, b, K, formula='sigma_C',
                          subsample=500, seed=42, use_log=True, epsilon=0.0):
    """Compute chain weights using the specified formula.

    Args:
        formula: 'sigma_C' (default), 'sigma_H', or 'uniform'
        use_log: if True, apply log transform; if False, use raw k*(1-ratio+b)
        epsilon: minimum weight floor per chain (default 0.0 = no floor).
            Added to normalized weights before re-normalization, ensuring
            no chain is fully suppressed. Useful for BCR where sigma_C
            puts 98.9% on variant_seq, making Heavy/Light invisible.
            Recommended: 0.01-0.05 when using BLOSUM-sqrt for BCR.

    Returns:
        weights: 1-D array, normalized to sum to 1
        chain_stds: 1-D array of per-chain LogDist std (zeros for uniform)
    """
    n_chains = len(chain_cols)

    if formula == 'uniform':
        weights = np.ones(n_chains) / n_chains
        chain_stds = np.zeros(n_chains)
        w_str = ', '.join(f'{c}={weights[i]:.4f}' for i, c in enumerate(chain_cols))
        print(f"    uniform weights: {w_str}")
        return weights, chain_stds

    # For sigma_H and sigma_C, we need per-chain LogDist std
    n_train = len(train_df)
    n_sub = min(subsample, n_train)
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(n_train, size=n_sub, replace=False)

    chain_stds = np.zeros(n_chains)
    for ch_idx, col in enumerate(chain_cols):
        ref_seqs = train_df[col].tolist()
        sub_seqs = [ref_seqs[i] for i in sub_idx]
        _, _, train_rmap = compute_pairwise_ratios(sub_seqs, ref_seqs)
        train_dists = np.array([
            logdist_from_ratios(train_rmap[s], k, b, K, use_log=use_log)
            for s in sub_seqs])
        chain_stds[ch_idx] = float(np.std(train_dists))

    if formula == 'sigma_H':
        # w_i = sigma_i * H_i (Shannon entropy)
        chain_H = np.zeros(n_chains)
        for ch_idx, col in enumerate(chain_cols):
            ref_seqs = train_df[col].tolist()
            freq = Counter(ref_seqs)
            counts = np.array(list(freq.values()), dtype=float)
            p = counts / counts.sum()
            chain_H[ch_idx] = float(-np.sum(p * np.log(p)))
            print(f"    {col}: std={chain_stds[ch_idx]:.6f}, H={chain_H[ch_idx]:.6f}")
        raw = chain_stds * chain_H
        weights = raw / raw.sum() if raw.sum() > 0 else np.ones(n_chains) / n_chains

    elif formula == 'sigma_C':
        # w_i = sigma_i * C_i (Simpson concentration)
        C_arr = np.zeros(n_chains)
        for ch_idx, col in enumerate(chain_cols):
            ref_seqs = train_df[col].tolist()
            freq = Counter(ref_seqs)
            n = len(ref_seqs)
            probs = np.array([c / n for c in freq.values()])
            C_arr[ch_idx] = float(np.sum(probs ** 2))
            print(f"    {col}: std={chain_stds[ch_idx]:.6f}, C={C_arr[ch_idx]:.6f}")
        raw = chain_stds * C_arr
        weights = raw / raw.sum() if raw.sum() > 0 else np.ones(n_chains) / n_chains

    else:
        raise ValueError(f"Unknown weight formula: {formula}")

    # Apply epsilon floor: ensure no chain is fully suppressed
    if epsilon > 0:
        weights = np.maximum(weights, epsilon)
        weights = weights / weights.sum()

    w_str = ', '.join(f'{c}={weights[i]:.4f}' for i, c in enumerate(chain_cols))
    print(f"    {formula} weights: {w_str}")
    return weights, chain_stds


def compute_sigma_c_weights(train_df, chain_cols, k, b, K, subsample=500, seed=42,
                             epsilon=0.0):
    """Compute sigma_C (sigma * Simpson concentration) chain weights.

    Backward-compatible alias for compute_chain_weights(..., formula='sigma_C').
    """
    return compute_chain_weights(train_df, chain_cols, k, b, K,
                                 formula='sigma_C', subsample=subsample, seed=seed,
                                 epsilon=epsilon)


def compute_combine_first_distances(test_df, train_df, chain_cols, weights,
                                      k, b, K, subsample=500,
                                      combine_method='weighted_max_znorm',
                                      use_log=True):
    """Compute combine-first distances with configurable method.

    Args:
        weights: 1-D array of chain weights (pre-computed).
        combine_method: one of COMBINE_FUNCTIONS keys (default: weighted_max_znorm)
        use_log: if True, apply log transform; if False, use raw k*(1-ratio+b)

    Returns 1-D numpy array of per-query distances.
    """
    from .general_evaluator import ZNORM_METHODS

    n_chains = len(chain_cols)
    ref_seqs_list = [train_df[col].tolist() for col in chain_cols]
    qry_seqs_list = [test_df[col].astype(str).tolist() for col in chain_cols]

    # Compute pairwise ratios
    ratio_maps_list = []
    unique_refs_list = []
    for ch_idx, col in enumerate(chain_cols):
        all_qry = list(dict.fromkeys(qry_seqs_list[ch_idx]))
        _, unique_refs, rmap = compute_pairwise_ratios(
            all_qry, ref_seqs_list[ch_idx])
        ratio_maps_list.append(rmap)
        unique_refs_list.append(unique_refs)

    # Compute chain stats only for znorm-based methods
    chain_stats = None
    if combine_method in ZNORM_METHODS:
        rng = np.random.RandomState(42)
        n_sub = min(subsample, len(ref_seqs_list[0]))
        sub_idx = rng.choice(len(ref_seqs_list[0]), size=n_sub, replace=False)

        train_ratio_maps_list = []
        train_unique_refs_list = []
        for ch_idx in range(n_chains):
            sub_seqs = [ref_seqs_list[ch_idx][i] for i in sub_idx]
            _, sub_u_ref, sub_rmap = compute_pairwise_ratios(
                sub_seqs, ref_seqs_list[ch_idx])
            train_ratio_maps_list.append(sub_rmap)
            train_unique_refs_list.append(sub_u_ref)

        chain_stats = compute_pairwise_chain_stats(
            train_ratio_maps_list, train_unique_refs_list, ref_seqs_list, k, b,
            subsample=subsample, use_log=use_log)

    # Compute distances
    distances = distances_combine_first_multi(
        qry_seqs_list, ratio_maps_list, unique_refs_list, ref_seqs_list,
        weights, k, b, K,
        reduction='topk',
        combine_method=combine_method,
        chain_stats=chain_stats,
        use_log=use_log)

    return distances


# ── Ridge-regularized vbias regression (unified for TCR + BCR) ──
#
# Canonical curve:  metric = a * exp(-b * d) + c + β * mean_p_bin
# Ridge penalty:    loss = MSE + lam * β²
#
# Pipeline-specific λ (2026-04-12 unified decision):
#   λ = 0.0  for TCR CV, TCR CT, BCR CV — stable fit at ≥8 training points
#   λ = 0.05 for BCR binding CT only — 8 cal bins per within-test half-split,
#            1 residual DoF on 4 params; β overfits at λ=0
#
# Prediction aggregation (2026-04-12 matched-granularity principle):
#   ALWAYS use avg_then_eval: predict_ridge_vbias(params, [mean_d], [mean_mp])
#   NEVER use eval_then_avg:  mean(predict_ridge_vbias(params, d_i, mp)) — this
#   adds Jensen-shifted noise uncorrelated with truth (loses 50/60 cells,
#   mean −0.193 R vs avg_then_eval). The curve is calibrated at bin centroids;
#   evaluating off-centroid and averaging gives systematically biased predictions.
#   Use predict_subset_metric() below for the correct one-call API.
#
# Reference: results/debug_sec17_1/full_ppvnpv_sweep.csv (120-row sweep),
#   results/debug_sec17_1/weighting_audit.csv (§15.6 reproduction),
#   UNIFIED_VBIAS_REPRODUCIBILITY.md §3.10-3.11

RIDGE_VBIAS_DEFAULT_LAM = 0.05
PERFORMANCE_PREDICTION_LAM = 0.0
CALIBRATION_LAM = 0.0
CALIBRATION_THRESHOLD = 0.5
RIDGE_VBIAS_BOUNDS = ([-2.0, 0.01, -1.0, -5.0], [2.0, 50.0, 1.0, 5.0])
RIDGE_VBIAS_INIT_GRID = [
    (a0, b0, c0, 0.0)
    for a0 in (0.0, 0.5, 1.0)
    for b0 in (0.1, 1.0, 5.0)
    for c0 in (0.0, 0.3, 0.6)
]


def fit_ridge_vbias(d, mean_p_bin, y, lam=RIDGE_VBIAS_DEFAULT_LAM):
    """Fit ridge-regularized vbias curve: a*exp(-b*d) + c + β*mean_p_bin.

    Minimizes:   MSE(y, pred) + lam * β²

    Parameters
    ----------
    d : np.ndarray
        Per-bin distance values.
    mean_p_bin : np.ndarray
        Per-bin mean prediction (observable at test time without labels).
    y : np.ndarray
        Per-bin target metric (e.g., PPV, NPV, AP, F1, AUROC).
    lam : float
        L2 regularization strength on β. Default 0.05 from v3 sweep.

    Returns
    -------
    params : tuple(a, b, c, β) or None
    """
    from scipy.optimize import minimize

    d = np.asarray(d, dtype=float)
    mean_p_bin = np.asarray(mean_p_bin, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~np.isnan(y) & ~np.isnan(d) & ~np.isnan(mean_p_bin)
    if valid.sum() < 4:
        return None
    dv, mpv, yv = d[valid], mean_p_bin[valid], y[valid]

    def loss(params):
        a, bx, c, beta = params
        pred = a * np.exp(-bx * dv) + c + beta * mpv
        return float(np.mean((yv - pred) ** 2) + lam * beta ** 2)

    best_params = None
    best_loss = np.inf
    lo, hi = RIDGE_VBIAS_BOUNDS
    bounds = list(zip(lo, hi))
    for x0 in RIDGE_VBIAS_INIT_GRID:
        try:
            res = minimize(loss, x0=list(x0), method='L-BFGS-B', bounds=bounds)
        except Exception:
            continue
        if res.success and res.fun < best_loss:
            best_params = tuple(res.x)
            best_loss = res.fun
    return best_params


def predict_ridge_vbias(params, d, mean_p_bin, lo=0.0, hi=1.0):
    """Apply ridge-vbias prediction. Returns NaN array if params is None.

    For subset-level performance prediction, prefer predict_subset_metric()
    which enforces the matched-granularity avg_then_eval pattern.
    """
    if params is None:
        return np.full_like(np.asarray(d, dtype=float), np.nan, dtype=float)
    a, bx, c, beta = params
    d = np.asarray(d, dtype=float)
    mp = np.asarray(mean_p_bin, dtype=float)
    return np.clip(a * np.exp(-bx * d) + c + beta * mp, lo, hi)


def predict_subset_metric(params, subset_df, lo=0.0, hi=1.0):
    """Predict a subset's metric using avg_then_eval (matched granularity).

    Evaluates the vbias curve at the subset's mean distance using the
    subset's own mean prediction as the vbias feature.  This is the
    correct prediction method when the curve was fit on bin-level tuples
    (mean_d, mean_pred, metric).

    DO NOT compute per-sample f(d_i) and average — that adds Jensen-
    shifted noise uncorrelated with truth.  See matched-granularity
    principle in UNIFIED_VBIAS_REPRODUCIBILITY.md §3.10.

    Parameters
    ----------
    params : tuple(a, b, c, beta) or None
        Fitted vbias curve parameters from fit_ridge_vbias().
    subset_df : DataFrame
        Must contain 'distance' and 'pred' columns.
    lo, hi : float
        Clip bounds (default 0-1 for metrics).

    Returns
    -------
    float : predicted metric value for the subset, or NaN if params is None.
    """
    if params is None:
        return float('nan')
    a, bx, c, beta = params
    mean_d = float(subset_df['distance'].mean())
    mean_p = float(subset_df['pred'].mean())
    return float(np.clip(a * np.exp(-bx * mean_d) + c + beta * mean_p, lo, hi))
