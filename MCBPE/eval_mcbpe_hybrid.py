#!/usr/bin/env python3
"""M-CBPE + S2DD hybrid evaluation experiment.

Compares three approaches for correlation consistency and performance prediction:
  1. S2DD-only: standard distance-performance curves (baseline from eval_performance_prediction.py)
  2. M-CBPE-only: density-ratio-based calibration for aggregate metric estimation
  3. Hybrid: S2DD binning + M-CBPE calibrated predictions per bin

Experiments:
  A. Correlation consistency: does M-CBPE calibration improve the
     negative distance-performance correlation (Pearson r)?
  B. Performance prediction: does hybrid improve leave-one-out MAE?

Models: 5 TCR (NetTCR, ATM-TCR, BLOSUM-RF, ERGO-II, TCR-BERT)
Protocols: 5-fold CV + 5-set cross-test
Metrics: aucroc, ap, f1
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from calipper.general_evaluator import safe_metric, binned_correlations
from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
)
from calipper.pattern_analysis import select_best_fit
from MCBPE.mcbpe_core import (
    mcbpe_estimate, build_features, hybrid_binned_evaluation,
    estimate_density_ratios, fit_weighted_calibrator, calibrate_predictions,
)

# ── Constants ───────────────────────────────────────────────────────────
CHAIN_COLS = ['peptide', 'CDR3a', 'CDR3b']
CV_FOLDS = 5
CROSS_TEST_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas']
CROSS_TEST_LABELS = {
    'seen_test': 'Seen', 'unseen_fold34': 'Unseen34',
    'v3_combined': 'V3', 'v4_combined': 'V4', 'mcpas': 'McPAS',
}
ALL_TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
MODEL_DISPLAY = {
    'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
    'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT',
}
FIT_MODELS = ['exponential']
K, k_param, b_param = 50, 0.1, 0.1
BIN_NUM = 8
METRICS = ['aucroc', 'ap', 'f1']


# ── Data Loading ────────────────────────────────────────────────────────

def load_cv_data(model, fold, results_base):
    """Load train and test data for a CV fold."""
    fold_dir = os.path.join(results_base, model, 'cv_logdist', f'fold{fold}')
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(fold_dir, 'test_predictions_with_label.csv'))
    return train_df, test_df


def load_crosstest_data(model, results_base):
    """Load training data and all cross-test prediction sets."""
    base = os.path.join(results_base, model, 'cross_test_logdist')
    splits_dir = os.path.join(base, 'splits')
    pred_dir = os.path.join(base, 'predictions')

    train_path = os.path.join(splits_dir, 'train_full.csv')
    if not os.path.exists(train_path):
        train_path = os.path.join(splits_dir, 'train.csv')
    train_df = pd.read_csv(train_path)

    test_dfs = {}
    for ts in CROSS_TEST_SETS:
        path = os.path.join(pred_dir, f'{ts}_predictions_with_label.csv')
        if os.path.exists(path):
            test_dfs[ts] = pd.read_csv(path)
    return train_df, test_dfs


# ── Distance Computation ────────────────────────────────────────────────

def compute_eval_df(test_df, train_df, weights=None):
    """Compute S2DD distances and build eval DataFrame."""
    if weights is None:
        weights, _ = compute_chain_weights(
            train_df, CHAIN_COLS, k_param, b_param, K, formula='sigma_C')

    distances = compute_combine_first_distances(
        test_df, train_df, CHAIN_COLS, weights, k_param, b_param, K)

    return pd.DataFrame({
        'label': test_df['binder'].values.astype(int),
        'pred': test_df['prediction'].values.astype(float),
        'distance': distances,
    }), weights


# ── M-CBPE Calibration ─────────────────────────────────────────────────

def apply_mcbpe_calibration(ref_eval_df, prod_eval_df, feature_mode='dist_pred'):
    """Apply M-CBPE calibration using S2DD distance as the DRE feature.

    Args:
        ref_eval_df: reference (training fold) eval DataFrame
        prod_eval_df: production (test) eval DataFrame
        feature_mode: which features for DRE
            'dist_only': S2DD distance only
            'dist_pred': S2DD distance + model prediction (default)

    Returns:
        calibrated_probs: (n_prod,) calibrated probabilities
        mcbpe_result: full M-CBPE result dict
    """
    # Build features for DRE
    if feature_mode == 'dist_only':
        ref_feats = ref_eval_df['distance'].values.reshape(-1, 1)
        prod_feats = prod_eval_df['distance'].values.reshape(-1, 1)
    elif feature_mode == 'dist_pred':
        ref_feats = ref_eval_df[['distance', 'pred']].values
        prod_feats = prod_eval_df[['distance', 'pred']].values
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    result = mcbpe_estimate(
        ref_predictions=ref_eval_df['pred'].values,
        ref_labels=ref_eval_df['label'].values,
        prod_predictions=prod_eval_df['pred'].values,
        ref_features=ref_feats,
        prod_features=prod_feats,
        metrics=METRICS,
        dre_method='logistic',
    )
    return result['calibrated_probs'], result


# ── Curve-based prediction (S2DD baseline) ──────────────────────────────

def apply_fit(fit_result, x_new):
    """Evaluate a fitted curve at new x values."""
    model = fit_result['model']
    p = fit_result['params']
    x = np.asarray(x_new, dtype=float)
    if model == 'linear':
        y = p['slope'] * x + p['intercept']
    elif model.startswith('polynomial'):
        y = np.polyval(p['coefficients'], x)
    elif model == 'exponential_decay':
        y = p['a'] * np.exp(-p['b'] * x) + p['c']
    elif model == 'sigmoid':
        y = p['L'] / (1 + np.exp(-p['k'] * (x - p['x0'])))
    elif model == 'isotonic':
        y = np.interp(x, p['x_knots'], p['y_knots'])
    else:
        raise ValueError(f"Unknown model type: {model}")
    return np.clip(y, 0.0, 1.0)


def curve_predict_overall(fit_result, eval_df, metric, bin_num):
    """Predict overall performance from fitted curve (S2DD baseline)."""
    sorted_df = eval_df.sort_values('distance').reset_index(drop=True)
    bs = len(sorted_df) // bin_num
    if bs == 0:
        return np.nan

    weighted_sum = 0.0
    total_n = 0
    for i in range(bin_num):
        s = i * bs
        e = len(sorted_df) if i == bin_num - 1 else (i + 1) * bs
        bd = sorted_df.iloc[s:e]
        n_i = len(bd)
        mean_dist = bd['distance'].mean()
        pred_perf = float(apply_fit(fit_result, mean_dist))
        weighted_sum += n_i * pred_perf
        total_n += n_i

    return weighted_sum / total_n if total_n > 0 else np.nan


# ── Experiment A: Correlation Consistency ───────────────────────────────

def run_correlation_experiment(models, results_base, output_dir):
    """Compare raw vs M-CBPE-calibrated correlations across all folds/tests."""
    rows = []

    for model in models:
        model_cv_dir = os.path.join(results_base, model, 'cv_logdist')
        if not os.path.isdir(model_cv_dir):
            print(f"  Skipping {model}: no cv_logdist directory")
            continue

        print(f"\n{'='*60}")
        print(f"Correlation: {MODEL_DISPLAY.get(model, model)} — CV")
        print(f"{'='*60}")

        for fold in range(CV_FOLDS):
            try:
                train_df, test_df = load_cv_data(model, fold, results_base)
            except FileNotFoundError:
                print(f"  Fold {fold}: data not found, skipping")
                continue

            print(f"\n  Fold {fold}: computing distances...")
            t0 = time.time()
            eval_df, weights = compute_eval_df(test_df, train_df)
            print(f"    Distances: {time.time()-t0:.1f}s ({len(eval_df)} samples)")

            # Build reference eval_df: use training data with dummy pred=0.5
            # For DRE, we need reference features — use training data distances
            print(f"    Computing reference (train-on-train) distances...")
            t0 = time.time()
            # Use a subsample of train as "reference production" for DRE feature space
            # DRE distinguishes train vs test in the distance+pred feature space
            ref_eval_df = eval_df.copy()  # Will use test fold's ref for DRE context

            # Apply M-CBPE: use OTHER folds as reference, this fold as production
            # For a cleaner setup: pool other folds' test predictions as reference
            other_folds_eval = []
            for of in range(CV_FOLDS):
                if of == fold:
                    continue
                try:
                    of_train, of_test = load_cv_data(model, of, results_base)
                    of_eval, _ = compute_eval_df(of_test, of_train, weights=weights)
                    other_folds_eval.append(of_eval)
                except Exception:
                    pass

            if not other_folds_eval:
                print(f"    No other folds available for reference, skipping M-CBPE")
                # Still record raw correlations
                for metric in METRICS:
                    bc = binned_correlations(eval_df, 'distance', [metric], BIN_NUM)
                    raw_r = bc[metric].get('pearson_r', np.nan)
                    raw_p = bc[metric].get('pearson_p', np.nan)
                    rows.append({
                        'model': model, 'experiment': 'cv', 'setting': f'fold{fold}',
                        'metric': metric,
                        'raw_r': raw_r, 'raw_p': raw_p,
                        'calib_r': np.nan, 'calib_p': np.nan,
                        'improvement': np.nan,
                    })
                continue

            ref_pool = pd.concat(other_folds_eval, ignore_index=True)
            print(f"    Reference pool: {len(ref_pool)} samples from {len(other_folds_eval)} folds ({time.time()-t0:.1f}s)")

            # M-CBPE calibration
            try:
                calib_probs, mcbpe_res = apply_mcbpe_calibration(ref_pool, eval_df)
            except Exception as e:
                print(f"    M-CBPE failed: {e}")
                calib_probs = None

            for metric in METRICS:
                if calib_probs is not None:
                    hbe = hybrid_binned_evaluation(eval_df, calib_probs, metric, BIN_NUM)
                    raw_r = hbe['raw_r']
                    raw_p_val = hbe['raw_p']
                    cal_r = hbe['calib_r']
                    cal_p_val = hbe['calib_p']
                else:
                    bc = binned_correlations(eval_df, 'distance', [metric], BIN_NUM)
                    raw_r = bc[metric].get('pearson_r', np.nan)
                    raw_p_val = bc[metric].get('pearson_p', np.nan)
                    cal_r = np.nan
                    cal_p_val = np.nan

                improvement = np.nan
                if not np.isnan(raw_r) and not np.isnan(cal_r):
                    # Improvement = more negative (or larger |r|)
                    improvement = abs(cal_r) - abs(raw_r)

                rows.append({
                    'model': model, 'experiment': 'cv', 'setting': f'fold{fold}',
                    'metric': metric,
                    'raw_r': raw_r, 'raw_p': raw_p_val,
                    'calib_r': cal_r, 'calib_p': cal_p_val,
                    'improvement': improvement,
                })
                sign_raw = '-' if raw_r < 0 else '+'
                sign_cal = '-' if cal_r < 0 else '+' if not np.isnan(cal_r) else '?'
                print(f"    {metric:>7s}: raw_r={raw_r:+.3f}({sign_raw}) "
                      f"calib_r={cal_r:+.3f}({sign_cal}) "
                      f"Δ|r|={improvement:+.3f}" if not np.isnan(improvement) else
                      f"    {metric:>7s}: raw_r={raw_r:+.3f}({sign_raw}) calib_r=N/A")

    # Cross-test
    for model in models:
        ct_dir = os.path.join(results_base, model, 'cross_test_logdist')
        if not os.path.isdir(ct_dir):
            continue

        print(f"\n{'='*60}")
        print(f"Correlation: {MODEL_DISPLAY.get(model, model)} — Cross-test")
        print(f"{'='*60}")

        try:
            train_df, test_dfs = load_crosstest_data(model, results_base)
        except FileNotFoundError:
            continue

        if len(test_dfs) < 2:
            continue

        print(f"  Computing sigma_C weights...")
        ct_weights, _ = compute_chain_weights(
            train_df, CHAIN_COLS, k_param, b_param, K, formula='sigma_C')

        # Compute distances for all test sets
        ts_eval_dfs = {}
        for ts_name, ts_df in test_dfs.items():
            print(f"  Computing distances for {ts_name}...")
            t0 = time.time()
            ts_eval_dfs[ts_name], _ = compute_eval_df(
                ts_df, train_df, weights=ct_weights)
            print(f"    {ts_name}: {time.time()-t0:.1f}s ({len(ts_eval_dfs[ts_name])} samples)")

        # For each test set: use other test sets as reference
        for held_ts in ts_eval_dfs:
            other_ts = [t for t in ts_eval_dfs if t != held_ts]
            ref_pool = pd.concat([ts_eval_dfs[t] for t in other_ts], ignore_index=True)

            try:
                calib_probs, _ = apply_mcbpe_calibration(ref_pool, ts_eval_dfs[held_ts])
            except Exception as e:
                print(f"    M-CBPE for {held_ts} failed: {e}")
                calib_probs = None

            for metric in METRICS:
                if calib_probs is not None:
                    hbe = hybrid_binned_evaluation(
                        ts_eval_dfs[held_ts], calib_probs, metric, BIN_NUM)
                    raw_r = hbe['raw_r']
                    raw_p_val = hbe['raw_p']
                    cal_r = hbe['calib_r']
                    cal_p_val = hbe['calib_p']
                else:
                    bc = binned_correlations(
                        ts_eval_dfs[held_ts], 'distance', [metric], BIN_NUM)
                    raw_r = bc[metric].get('pearson_r', np.nan)
                    raw_p_val = bc[metric].get('pearson_p', np.nan)
                    cal_r = np.nan
                    cal_p_val = np.nan

                improvement = np.nan
                if not np.isnan(raw_r) and not np.isnan(cal_r):
                    improvement = abs(cal_r) - abs(raw_r)

                rows.append({
                    'model': model, 'experiment': 'crosstest',
                    'setting': held_ts, 'metric': metric,
                    'raw_r': raw_r, 'raw_p': raw_p_val,
                    'calib_r': cal_r, 'calib_p': cal_p_val,
                    'improvement': improvement,
                })

            ts_label = CROSS_TEST_LABELS.get(held_ts, held_ts)
            print(f"  {ts_label}: " + ', '.join(
                f"{m}={rows[-3 + i]['raw_r']:+.3f}→{rows[-3 + i]['calib_r']:+.3f}"
                for i, m in enumerate(METRICS)
            ))

    return pd.DataFrame(rows)


# ── Experiment B: Performance Prediction ────────────────────────────────

def run_prediction_experiment(models, results_base, output_dir):
    """Compare S2DD-only vs M-CBPE-only vs hybrid for leave-one-out prediction."""
    rows = []

    # Only use the 3 models from the original prediction experiment for comparison
    pred_models = [m for m in models if m in ['nettcr', 'atm_tcr', 'blosum_rf']]

    for model in pred_models:
        print(f"\n{'='*60}")
        print(f"Prediction: {MODEL_DISPLAY.get(model, model)} — CV")
        print(f"{'='*60}")

        # Load all folds
        fold_data = {}
        fold_eval_dfs = {}
        fold_weights = None

        for fold in range(CV_FOLDS):
            try:
                train_df, test_df = load_cv_data(model, fold, results_base)
                fold_data[fold] = {'train': train_df, 'test': test_df}
            except FileNotFoundError:
                print(f"  Fold {fold}: not found")
                continue

        if len(fold_data) < CV_FOLDS:
            print(f"  Only {len(fold_data)} folds, skipping")
            continue

        # Compute distances
        for fold in range(CV_FOLDS):
            print(f"  Fold {fold}: computing distances...")
            t0 = time.time()
            fold_eval_dfs[fold], w = compute_eval_df(
                fold_data[fold]['test'], fold_data[fold]['train'])
            if fold_weights is None:
                fold_weights = w
            print(f"    {time.time()-t0:.1f}s ({len(fold_eval_dfs[fold])} samples)")

        # Leave-one-fold-out
        for held_out in range(CV_FOLDS):
            other_folds = [f for f in range(CV_FOLDS) if f != held_out]
            ref_pool = pd.concat(
                [fold_eval_dfs[f] for f in other_folds], ignore_index=True)
            held_df = fold_eval_dfs[held_out]

            # M-CBPE calibration
            try:
                calib_probs, mcbpe_res = apply_mcbpe_calibration(ref_pool, held_df)
            except Exception as e:
                print(f"  Fold {held_out}: M-CBPE failed: {e}")
                calib_probs = None

            for metric in METRICS:
                actual = safe_metric(
                    metric, held_df['label'].values, held_df['pred'].values)

                # Method 1: S2DD curve-based prediction
                pool_dists, pool_perfs = [], []
                for f in other_folds:
                    bc = binned_correlations(
                        fold_eval_dfs[f], 'distance', [metric], BIN_NUM)
                    pool_dists.extend(bc[metric]['bin_dists'])
                    pool_perfs.extend(bc[metric]['bin_perfs'])

                pool_dists = np.array(pool_dists)
                pool_perfs = np.array(pool_perfs)
                valid = ~np.isnan(pool_perfs)
                s2dd_pred = np.nan
                if valid.sum() >= 3:
                    _, best_fit = select_best_fit(
                        pool_dists[valid].tolist(), pool_perfs[valid].tolist(),
                        FIT_MODELS)
                    s2dd_pred = curve_predict_overall(best_fit, held_df, metric, BIN_NUM)

                # Method 2: M-CBPE direct estimate
                mcbpe_pred = np.nan
                if calib_probs is not None and mcbpe_res is not None:
                    mcbpe_pred = mcbpe_res['estimated'].get(metric, np.nan)

                # Method 3: Hybrid — average of S2DD and M-CBPE
                hybrid_pred = np.nan
                if not np.isnan(s2dd_pred) and not np.isnan(mcbpe_pred):
                    hybrid_pred = 0.5 * s2dd_pred + 0.5 * mcbpe_pred

                # Method 4: Hybrid-calibrated — S2DD curve on calibrated predictions
                calib_curve_pred = np.nan
                if calib_probs is not None and valid.sum() >= 3:
                    calib_held = held_df.copy()
                    calib_held['pred'] = calib_probs
                    # Collect calibrated per-bin data from other folds
                    calib_pool_dists, calib_pool_perfs = [], []
                    for f in other_folds:
                        other_ref = pd.concat(
                            [fold_eval_dfs[ff] for ff in range(CV_FOLDS)
                             if ff != f and ff != held_out],
                            ignore_index=True)
                        try:
                            of_calib, _ = apply_mcbpe_calibration(
                                other_ref, fold_eval_dfs[f])
                            of_calib_df = fold_eval_dfs[f].copy()
                            of_calib_df['pred'] = of_calib
                            bc_c = binned_correlations(
                                of_calib_df, 'distance', [metric], BIN_NUM)
                            calib_pool_dists.extend(bc_c[metric]['bin_dists'])
                            calib_pool_perfs.extend(bc_c[metric]['bin_perfs'])
                        except Exception:
                            # Fall back to raw
                            bc_c = binned_correlations(
                                fold_eval_dfs[f], 'distance', [metric], BIN_NUM)
                            calib_pool_dists.extend(bc_c[metric]['bin_dists'])
                            calib_pool_perfs.extend(bc_c[metric]['bin_perfs'])

                    cpd = np.array(calib_pool_dists)
                    cpp = np.array(calib_pool_perfs)
                    cvalid = ~np.isnan(cpp)
                    if cvalid.sum() >= 3:
                        _, cfit = select_best_fit(
                            cpd[cvalid].tolist(), cpp[cvalid].tolist(), FIT_MODELS)
                        calib_curve_pred = curve_predict_overall(
                            cfit, calib_held, metric, BIN_NUM)

                rows.append({
                    'model': model, 'experiment': 'cv', 'setting': f'fold{held_out}',
                    'metric': metric, 'actual': actual,
                    's2dd_pred': s2dd_pred, 's2dd_err': abs(s2dd_pred - actual) if not np.isnan(s2dd_pred) else np.nan,
                    'mcbpe_pred': mcbpe_pred, 'mcbpe_err': abs(mcbpe_pred - actual) if not np.isnan(mcbpe_pred) else np.nan,
                    'hybrid_pred': hybrid_pred, 'hybrid_err': abs(hybrid_pred - actual) if not np.isnan(hybrid_pred) else np.nan,
                    'calib_curve_pred': calib_curve_pred, 'calib_curve_err': abs(calib_curve_pred - actual) if not np.isnan(calib_curve_pred) else np.nan,
                })
                print(f"  Fold {held_out}, {metric}: actual={actual:.3f} | "
                      f"S2DD={s2dd_pred:.3f} M-CBPE={mcbpe_pred:.3f} "
                      f"Hybrid={hybrid_pred:.3f} CalibCurve={calib_curve_pred:.3f}")

    # Cross-test prediction
    for model in pred_models:
        ct_dir = os.path.join(results_base, model, 'cross_test_logdist')
        if not os.path.isdir(ct_dir):
            continue

        print(f"\n{'='*60}")
        print(f"Prediction: {MODEL_DISPLAY.get(model, model)} — Cross-test")
        print(f"{'='*60}")

        try:
            train_df, test_dfs = load_crosstest_data(model, results_base)
        except FileNotFoundError:
            continue

        ct_weights, _ = compute_chain_weights(
            train_df, CHAIN_COLS, k_param, b_param, K, formula='sigma_C')

        ts_eval_dfs = {}
        for ts_name, ts_df in test_dfs.items():
            print(f"  {ts_name}: computing distances...")
            ts_eval_dfs[ts_name], _ = compute_eval_df(
                ts_df, train_df, weights=ct_weights)

        available_ts = list(ts_eval_dfs.keys())
        for held_ts in available_ts:
            other_ts = [t for t in available_ts if t != held_ts]
            ref_pool = pd.concat(
                [ts_eval_dfs[t] for t in other_ts], ignore_index=True)
            held_df = ts_eval_dfs[held_ts]

            try:
                calib_probs, mcbpe_res = apply_mcbpe_calibration(ref_pool, held_df)
            except Exception:
                calib_probs = None
                mcbpe_res = None

            for metric in METRICS:
                actual = safe_metric(
                    metric, held_df['label'].values, held_df['pred'].values)

                # S2DD curve
                pool_dists, pool_perfs = [], []
                for ts in other_ts:
                    bc = binned_correlations(
                        ts_eval_dfs[ts], 'distance', [metric], BIN_NUM)
                    pool_dists.extend(bc[metric]['bin_dists'])
                    pool_perfs.extend(bc[metric]['bin_perfs'])

                pool_dists = np.array(pool_dists)
                pool_perfs = np.array(pool_perfs)
                valid_mask = ~np.isnan(pool_perfs)
                s2dd_pred = np.nan
                if valid_mask.sum() >= 3:
                    _, best_fit = select_best_fit(
                        pool_dists[valid_mask].tolist(),
                        pool_perfs[valid_mask].tolist(), FIT_MODELS)
                    s2dd_pred = curve_predict_overall(best_fit, held_df, metric, BIN_NUM)

                mcbpe_pred = np.nan
                if mcbpe_res is not None:
                    mcbpe_pred = mcbpe_res['estimated'].get(metric, np.nan)

                hybrid_pred = np.nan
                if not np.isnan(s2dd_pred) and not np.isnan(mcbpe_pred):
                    hybrid_pred = 0.5 * s2dd_pred + 0.5 * mcbpe_pred

                rows.append({
                    'model': model, 'experiment': 'crosstest',
                    'setting': held_ts, 'metric': metric, 'actual': actual,
                    's2dd_pred': s2dd_pred, 's2dd_err': abs(s2dd_pred - actual) if not np.isnan(s2dd_pred) else np.nan,
                    'mcbpe_pred': mcbpe_pred, 'mcbpe_err': abs(mcbpe_pred - actual) if not np.isnan(mcbpe_pred) else np.nan,
                    'hybrid_pred': hybrid_pred, 'hybrid_err': abs(hybrid_pred - actual) if not np.isnan(hybrid_pred) else np.nan,
                    'calib_curve_pred': np.nan, 'calib_curve_err': np.nan,
                })

    return pd.DataFrame(rows)


# ── Reporting ───────────────────────────────────────────────────────────

def summarize_correlations(corr_df, output_dir):
    """Summarize correlation improvements."""
    if corr_df.empty:
        return

    summary_rows = []
    for exp in corr_df['experiment'].unique():
        for model in corr_df['model'].unique():
            for metric in METRICS:
                sub = corr_df[(corr_df['experiment'] == exp) &
                              (corr_df['model'] == model) &
                              (corr_df['metric'] == metric)]
                if sub.empty:
                    continue
                n = len(sub)
                raw_neg = (sub['raw_r'] < 0).sum()
                cal_neg = (sub['calib_r'] < 0).sum()
                mean_raw = sub['raw_r'].mean()
                mean_cal = sub['calib_r'].mean()
                mean_imp = sub['improvement'].mean()
                summary_rows.append({
                    'experiment': exp, 'model': model, 'metric': metric,
                    'n_settings': n,
                    'raw_mean_r': mean_raw, 'raw_neg_frac': raw_neg / n,
                    'calib_mean_r': mean_cal, 'calib_neg_frac': cal_neg / n,
                    'mean_delta_abs_r': mean_imp,
                })

    summary_df = pd.DataFrame(summary_rows)
    path = os.path.join(output_dir, 'correlation_summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Print summary
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"  {row['experiment']:>10s} | {MODEL_DISPLAY.get(row['model'], row['model']):>10s} | "
              f"{row['metric']:>7s} | raw_r={row['raw_mean_r']:+.3f} "
              f"({row['raw_neg_frac']*100:.0f}% neg) | "
              f"calib_r={row['calib_mean_r']:+.3f} "
              f"({row['calib_neg_frac']*100:.0f}% neg) | "
              f"Δ|r|={row['mean_delta_abs_r']:+.3f}")

    return summary_df


def summarize_predictions(pred_df, output_dir):
    """Summarize prediction accuracy comparison."""
    if pred_df.empty:
        return

    summary_rows = []
    for exp in pred_df['experiment'].unique():
        for method in ['s2dd', 'mcbpe', 'hybrid', 'calib_curve']:
            sub = pred_df[pred_df['experiment'] == exp]
            err_col = f'{method}_err'
            pred_col = f'{method}_pred'
            valid = sub[~sub[err_col].isna()]
            if valid.empty:
                continue

            mae = valid[err_col].mean()
            try:
                r, p = pearsonr(valid[pred_col], valid['actual'])
            except Exception:
                r, p = np.nan, np.nan

            summary_rows.append({
                'experiment': exp, 'method': method,
                'n': len(valid), 'mae': mae,
                'pearson_r': r, 'pearson_p': p,
            })

    summary_df = pd.DataFrame(summary_rows)
    path = os.path.join(output_dir, 'prediction_summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"  {row['experiment']:>10s} | {row['method']:>12s} | "
              f"MAE={row['mae']:.4f} | R={row['pearson_r']:.3f} "
              f"(p={row['pearson_p']:.2e}) | n={row['n']}")

    return summary_df


def plot_comparison(pred_df, output_dir):
    """Plot predicted vs actual for each method."""
    if pred_df.empty:
        return

    methods = ['s2dd', 'mcbpe', 'hybrid']
    method_labels = {'s2dd': 'S2DD (curve)', 'mcbpe': 'M-CBPE', 'hybrid': 'Hybrid'}
    colors = {'s2dd': '#1f77b4', 'mcbpe': '#ff7f0e', 'hybrid': '#2ca02c'}

    for exp in pred_df['experiment'].unique():
        sub = pred_df[pred_df['experiment'] == exp]
        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            pred_col = f'{method}_pred'
            valid = sub[~sub[pred_col].isna()]
            if valid.empty:
                ax.set_title(f'{method_labels[method]}\n(no data)')
                continue

            ax.scatter(valid['actual'], valid[pred_col],
                       c=colors[method], alpha=0.7, s=50, edgecolors='white')
            vmin = min(valid['actual'].min(), valid[pred_col].min()) - 0.02
            vmax = max(valid['actual'].max(), valid[pred_col].max()) + 0.02
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5)

            try:
                r, _ = pearsonr(valid[pred_col], valid['actual'])
                mae = valid[f'{method}_err'].mean()
                ax.text(0.05, 0.95, f'R={r:.3f}\nMAE={mae:.4f}',
                        transform=ax.transAxes, va='top', fontsize=10,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            except Exception:
                pass

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(method_labels[method])
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect('equal')

        fig.suptitle(f'Performance Prediction: {"CV" if exp == "cv" else "Cross-Test"}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'prediction_scatter_{exp}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='M-CBPE + S2DD hybrid evaluation')
    parser.add_argument('--output-dir', default='MCBPE/results')
    parser.add_argument('--models', nargs='+', default=ALL_TCR_MODELS)
    parser.add_argument('--skip-correlation', action='store_true',
                        help='Skip correlation experiment')
    parser.add_argument('--skip-prediction', action='store_true',
                        help='Skip prediction experiment')
    args = parser.parse_args()

    results_base = os.path.join(PROJECT_ROOT, 'results')
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    corr_df = None
    pred_df = None

    # ── Experiment A: Correlations ──
    if not args.skip_correlation:
        print("\n" + "=" * 70)
        print("EXPERIMENT A: Correlation Consistency (raw vs M-CBPE calibrated)")
        print("=" * 70)
        corr_df = run_correlation_experiment(args.models, results_base, output_dir)
        if not corr_df.empty:
            corr_path = os.path.join(output_dir, 'correlation_results.csv')
            corr_df.to_csv(corr_path, index=False)
            print(f"\nSaved: {corr_path} ({len(corr_df)} rows)")
            summarize_correlations(corr_df, output_dir)

    # ── Experiment B: Prediction ──
    if not args.skip_prediction:
        print("\n" + "=" * 70)
        print("EXPERIMENT B: Performance Prediction (S2DD vs M-CBPE vs Hybrid)")
        print("=" * 70)
        pred_df = run_prediction_experiment(args.models, results_base, output_dir)
        if not pred_df.empty:
            pred_path = os.path.join(output_dir, 'prediction_results.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"\nSaved: {pred_path} ({len(pred_df)} rows)")
            summarize_predictions(pred_df, output_dir)
            plot_comparison(pred_df, output_dir)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
