#!/usr/bin/env python3
"""Pre-compute and cache ALL BCR data for Fig 3 (prediction) and Fig 4 (subset).

Fig 3 caches:
  - CV within-fold halfsplit: vbias curve points + (predicted, actual) per fold
  - CT LOO: vbias curve points + (predicted, actual) per held-out test set
  - Baseline comparison (PAPE, M-CBPE) from existing CSVs

Fig 4 caches:
  - Per-antigen subset: (predicted, actual) for AUROC/AP
  - Per-distance subset: (predicted, actual) for AUROC/AP
  - Source identity scatter: per test-set (actual, predicted, n, prevalence)

All distances: sigma_C 3-chain (Heavy + Light + variant_seq), k=0.1, b=0.03, K=30.
BCR uses XBCR-net only (single model), 5-fold CV + 4 CT test sets.
"""
import os, sys, time, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
)

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
os.makedirs(CACHE, exist_ok=True)

CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
k, b, K = 0.1, 0.03, 30
N_BINS = 8
MIN_SAMPLES = 30
METRICS = ['aucroc', 'ap', 'f1']
from calipper.core import VBIAS_BETA_LAM
UNIFIED_LAM = VBIAS_BETA_LAM   # 0.05, from General_Eval/s2dd.py
RESIDUAL_LAM = VBIAS_BETA_LAM  # same

CT_SETS = ['A1-A11', 'unseen', 'flu']
CT_DIST_DIR = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')  # fold4-as-cal pipeline (fold95 model)

# ── PAPE functions ──
from PAPE.pape_core import (estimate_importance_weights,
                             fit_weighted_calibration,
                             apply_calibration,
                             estimate_metric as pape_eq4)


def _split_dist_indices(distances, n_sub=N_BINS):
    si = np.argsort(distances)
    bs = len(si) // n_sub
    if bs < MIN_SAMPLES:
        return []
    return [si[i * bs:(len(si) if i == n_sub - 1 else (i + 1) * bs)]
            for i in range(n_sub)]


def split_halves_distance(distances):
    si = np.argsort(distances)
    even = si[::2]
    odd = si[1::2]
    return even, odd


# ═══════════════════════════════════════════
# Load BCR data with pre-cached distances
# ═══════════════════════════════════════════
print("Loading BCR data...")
t0 = time.time()

# CV data (XBCR-net, 5 folds)
cv_data = {}  # fold → dict(label, pred, distance)
bcr_cv_cache = os.path.join(RESULTS, 'fig2_bcr_cache')
for fold in range(5):
    tp = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
    if not os.path.exists(tp):
        continue
    te = pd.read_csv(tp)
    # Load pre-cached distances from fig2 cache
    dist_cache = os.path.join(bcr_cv_cache, f'bcr_cv_xbcr_net_fold{fold}_bins.npz')
    if os.path.exists(dist_cache):
        d_data = np.load(dist_cache)
        d = d_data['distances']
    else:
        # Recompute if no cache
        tr = pd.read_csv(os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'train.csv'))
        w, _ = compute_chain_weights(tr, CHAIN_COLS, k, b, K, formula='sigma_C')
        d = compute_combine_first_distances(te, tr, CHAIN_COLS, w, k, b, K)

    cv_data[fold] = {
        'label': te['rbd'].values.astype(int),
        'pred': te['pred_prob'].values.astype(float),
        'distance': d,
        'variant_seq': te['variant_seq'].values,
    }
    print(f"  CV fold{fold}: {len(te)} samples")

# CT data (fold4-as-cal pipeline, XBCR-net fold95 model, updated 2026-04-25)
ct_data = {}  # test_set → dict(label, pred, distance)
for ts in CT_SETS:
    pred_path = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal', 'xbcr', f'{ts}_predictions.csv')
    if not os.path.exists(pred_path):
        continue
    te = pd.read_csv(pred_path)
    if 'distance' not in te.columns:
        continue
    ct_data[ts] = {
        'label': te['rbd'].values.astype(int),
        'pred': te['pred_prob'].values.astype(float),
        'distance': te['distance'].values.astype(float),
        'variant_seq': te['variant_seq'].values,
    }
    print(f"  CT {ts}: {len(te)} samples")

print(f"Data loaded ({time.time()-t0:.0f}s)\n")


# ═══════════════════════════════════════════
# Fig 3: CV within-fold halfsplit prediction
# ═══════════════════════════════════════════
print("=== Fig 3: CV within-fold halfsplit ===")

cv_cache = os.path.join(CACHE, 'bcr_fig3_cv_prediction.npz')
if not os.path.exists(cv_cache):
    all_cv_results = {m: [] for m in METRICS}
    all_cv_curves = []  # (fold, d_bins, mp_bins, actual_bins, predicted_bins, metric)

    for fold in sorted(cv_data.keys()):
        data = cv_data[fold]
        y, p, d = data['label'], data['pred'], data['distance']

        cal_idx, val_idx = split_halves_distance(d)
        cal_y, cal_p, cal_d = y[cal_idx], p[cal_idx], d[cal_idx]
        val_y, val_p, val_d = y[val_idx], p[val_idx], d[val_idx]

        # v2.6 step 1-2: DRE + calibrator
        w_dre, _, _ = estimate_importance_weights(
            np.stack([cal_d, cal_p], 1), np.stack([val_d, val_p], 1))
        cal_model = fit_weighted_calibration(cal_p, cal_y, w_dre)
        c_cal = apply_calibration(cal_model, cal_p)
        c_val = apply_calibration(cal_model, val_p)

        for metric in METRICS:
            # v2.6 step 3: cal bins
            bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
            for idx in _split_dist_indices(cal_d):
                a_m = safe_metric(metric, cal_y[idx], cal_p[idx])
                p_m = pape_eq4(c_cal[idx], cal_p[idx], metric, threshold=0.5)
                if not np.isnan(a_m) and not np.isnan(p_m):
                    bin_d.append(cal_d[idx].mean())
                    bin_mp.append(cal_p[idx].mean())
                    bin_actual.append(a_m)
                    bin_pape.append(p_m)

            if len(bin_actual) < 4:
                continue
            bin_d, bin_mp = np.array(bin_d), np.array(bin_mp)
            bin_actual, bin_pape = np.array(bin_actual), np.array(bin_pape)

            # v2.7: joint curve + β vbias correction on PAPE residuals
            from calipper.core import fit_best_curve, predict_best_curve
            residual = bin_actual - bin_pape
            if len(residual) >= 4:
                fit_res = fit_best_curve(bin_d, bin_mp, residual, lam=RESIDUAL_LAM)
            else:
                fit_res = {'params': None}

            # Predict each val bin
            for idx in _split_dist_indices(val_d):
                actual = safe_metric(metric, val_y[idx], val_p[idx])
                if np.isnan(actual):
                    continue
                pape_bin = pape_eq4(c_val[idx], val_p[idx], metric, threshold=0.5)
                if fit_res['params'] is not None:
                    correction = float(predict_best_curve(
                        fit_res, np.array([val_d[idx].mean()]),
                        np.array([val_p[idx].mean()]))[0])
                else:
                    correction = 0.0
                predicted = float(np.clip(pape_bin + correction, 0, 1))
                all_cv_results[metric].append((predicted, actual))

            # Save curve points for plotting
            all_cv_curves.append({
                'fold': fold, 'metric': metric,
                'bin_d': bin_d.tolist(), 'bin_mp': bin_mp.tolist(),
                'bin_actual': bin_actual.tolist(), 'bin_pape': bin_pape.tolist(),
            })

    # Save
    save_dict = {}
    for m in METRICS:
        pairs = all_cv_results[m]
        if pairs:
            arr = np.array(pairs)
            save_dict[f'cv_{m}_predicted'] = arr[:, 0]
            save_dict[f'cv_{m}_actual'] = arr[:, 1]
    # Save curve JSON separately
    with open(os.path.join(CACHE, 'bcr_fig3_cv_curves.json'), 'w') as f:
        json.dump(all_cv_curves, f)
    np.savez(os.path.join(CACHE, 'bcr_fig3_cv_prediction.npz'), **save_dict)
    print(f"  Saved CV prediction ({sum(len(v) for v in all_cv_results.values())} total points)")
else:
    print("  [CACHED] CV prediction")


# ═══════════════════════════════════════════
# Fig 3: CT LOO prediction
# ═══════════════════════════════════════════
print("=== Fig 3: CT LOO prediction ===")

ct_pred_cache = os.path.join(CACHE, 'bcr_fig3_ct_prediction.npz')
if not os.path.exists(ct_pred_cache):
    all_ct_results = {m: [] for m in METRICS}
    all_ct_curves = []

    partitions = sorted(ct_data.keys())
    for held in partitions:
        others = [p for p in partitions if p != held]
        cal_y = np.concatenate([ct_data[p]['label'] for p in others])
        cal_p = np.concatenate([ct_data[p]['pred'] for p in others])
        cal_d = np.concatenate([ct_data[p]['distance'] for p in others])
        test = ct_data[held]
        test_y, test_p, test_d = test['label'], test['pred'], test['distance']

        # v2.6 step 1-2
        w_dre, _, _ = estimate_importance_weights(
            np.stack([cal_d, cal_p], 1), np.stack([test_d, test_p], 1))
        cal_model = fit_weighted_calibration(cal_p, cal_y, w_dre)
        c_cal = apply_calibration(cal_model, cal_p)
        c_test = apply_calibration(cal_model, test_p)

        for metric in METRICS:
            # v2.6 step 3: per-partition bins with global offset
            bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
            offset = 0
            for part_name in others:
                part_d = ct_data[part_name]['distance']
                n_p = len(part_d)
                for idx_local in _split_dist_indices(part_d):
                    idx_global = idx_local + offset
                    yi = cal_y[idx_global]
                    fi = cal_p[idx_global]
                    di = cal_d[idx_global]
                    ci = c_cal[idx_global]
                    a_m = safe_metric(metric, yi, fi)
                    p_m = pape_eq4(ci, fi, metric, threshold=0.5)
                    if not np.isnan(a_m) and not np.isnan(p_m):
                        bin_d.append(di.mean())
                        bin_mp.append(fi.mean())
                        bin_actual.append(a_m)
                        bin_pape.append(p_m)
                offset += n_p

            if len(bin_actual) < 4:
                continue
            bin_d, bin_mp = np.array(bin_d), np.array(bin_mp)
            bin_actual, bin_pape = np.array(bin_actual), np.array(bin_pape)

            # v2.7: joint curve + β vbias correction on PAPE residuals
            residual = bin_actual - bin_pape
            if len(residual) >= 4:
                fit_res = fit_best_curve(bin_d, bin_mp, residual, lam=RESIDUAL_LAM)
            else:
                fit_res = {'params': None}

            # Predict held-out bins
            for idx in _split_dist_indices(test_d):
                actual = safe_metric(metric, test_y[idx], test_p[idx])
                if np.isnan(actual):
                    continue
                pape_bin = pape_eq4(c_test[idx], test_p[idx], metric, threshold=0.5)
                if fit_res['params'] is not None:
                    correction = float(predict_best_curve(
                        fit_res, np.array([test_d[idx].mean()]),
                        np.array([test_p[idx].mean()]))[0])
                else:
                    correction = 0.0
                predicted = float(np.clip(pape_bin + correction, 0, 1))
                all_ct_results[metric].append((predicted, actual, held))

            all_ct_curves.append({
                'held': held, 'metric': metric,
                'bin_d': bin_d.tolist(), 'bin_mp': bin_mp.tolist(),
                'bin_actual': bin_actual.tolist(),
            })

    save_dict = {}
    for m in METRICS:
        pairs = all_ct_results[m]
        if pairs:
            arr = np.array([(p[0], p[1]) for p in pairs])
            save_dict[f'ct_{m}_predicted'] = arr[:, 0]
            save_dict[f'ct_{m}_actual'] = arr[:, 1]
            save_dict[f'ct_{m}_held'] = np.array([p[2] for p in pairs], dtype=object)
    with open(os.path.join(CACHE, 'bcr_fig3_ct_curves.json'), 'w') as f:
        json.dump(all_ct_curves, f)
    np.savez(os.path.join(CACHE, 'bcr_fig3_ct_prediction.npz'), **save_dict,
             allow_pickle=True)
    print(f"  Saved CT prediction ({sum(len(v) for v in all_ct_results.values())} total points)")
else:
    print("  [CACHED] CT prediction")


# ═══════════════════════════════════════════
# Fig 4: Per-antigen subset prediction
# ═══════════════════════════════════════════
print("\n=== Fig 4: Per-antigen subset prediction ===")

antigen_cache = os.path.join(CACHE, 'bcr_fig4_antigen_subset.npz')
if not os.path.exists(antigen_cache):
    # For each CV fold, compute per-antigen actual and predicted metrics
    antigen_results = []
    for fold in sorted(cv_data.keys()):
        data = cv_data[fold]
        y, p, d, vs = data['label'], data['pred'], data['distance'], data['variant_seq']

        for variant in np.unique(vs):
            mask = vs == variant
            n = mask.sum()
            if n < MIN_SAMPLES:
                continue
            yi, pi, di = y[mask], p[mask], d[mask]
            for metric in ['aucroc', 'ap']:
                actual = safe_metric(metric, yi, pi)
                if np.isnan(actual):
                    continue
                antigen_results.append({
                    'fold': fold, 'variant': variant, 'metric': metric,
                    'actual': actual, 'mean_dist': di.mean(),
                    'n': n, 'prevalence': yi.mean(),
                })

    antigen_df = pd.DataFrame(antigen_results)
    antigen_df.to_csv(os.path.join(CACHE, 'bcr_fig4_antigen_subset.csv'), index=False)
    np.savez(antigen_cache, n_rows=len(antigen_df))
    print(f"  Saved antigen subset ({len(antigen_df)} rows)")
else:
    print("  [CACHED] Antigen subset")


# ═══════════════════════════════════════════
# Fig 4: Per-distance subset prediction
# ═══════════════════════════════════════════
print("=== Fig 4: Per-distance subset prediction ===")

distance_cache = os.path.join(CACHE, 'bcr_fig4_distance_subset.npz')
if not os.path.exists(distance_cache):
    distance_results = []

    # CV: within-fold halfsplit distance bins
    for fold in sorted(cv_data.keys()):
        data = cv_data[fold]
        y, p, d = data['label'], data['pred'], data['distance']
        for idx in _split_dist_indices(d):
            for metric in ['aucroc', 'ap']:
                actual = safe_metric(metric, y[idx], p[idx])
                if not np.isnan(actual):
                    distance_results.append({
                        'split': 'cv', 'fold': fold, 'metric': metric,
                        'actual': actual, 'mean_dist': d[idx].mean(),
                        'n': len(idx), 'prevalence': y[idx].mean(),
                    })

    # CT: per test-set distance bins
    for ts in sorted(ct_data.keys()):
        data = ct_data[ts]
        y, p, d = data['label'], data['pred'], data['distance']
        for idx in _split_dist_indices(d):
            for metric in ['aucroc', 'ap']:
                actual = safe_metric(metric, y[idx], p[idx])
                if not np.isnan(actual):
                    distance_results.append({
                        'split': 'ct', 'test_set': ts, 'metric': metric,
                        'actual': actual, 'mean_dist': d[idx].mean(),
                        'n': len(idx), 'prevalence': y[idx].mean(),
                    })

    dist_df = pd.DataFrame(distance_results)
    dist_df.to_csv(os.path.join(CACHE, 'bcr_fig4_distance_subset.csv'), index=False)
    np.savez(distance_cache, n_rows=len(dist_df))
    print(f"  Saved distance subset ({len(dist_df)} rows)")
else:
    print("  [CACHED] Distance subset")


# ═══════════════════════════════════════════
# Fig 4: CT source identity scatter
# ═══════════════════════════════════════════
print("=== Fig 4: CT source identity scatter ===")

source_cache = os.path.join(CACHE, 'bcr_fig4_source_identity.csv')
if not os.path.exists(source_cache):
    source_results = []
    for ts in sorted(ct_data.keys()):
        data = ct_data[ts]
        y, p = data['label'], data['pred']
        for metric in ['aucroc', 'ap']:
            actual = safe_metric(metric, y, p)
            source_results.append({
                'test_set': ts, 'metric': metric, 'actual': actual,
                'n': len(y), 'prevalence': y.mean(),
            })
    pd.DataFrame(source_results).to_csv(source_cache, index=False)
    print(f"  Saved source identity ({len(source_results)} rows)")
else:
    print("  [CACHED] Source identity")


print(f"\n=== All BCR Fig 3/4 caching complete ({time.time()-t0:.0f}s) ===")
print(f"Cache directory: {CACHE}")
for f in sorted(os.listdir(CACHE)):
    size = os.path.getsize(os.path.join(CACHE, f))
    print(f"  {f} ({size/1024:.1f} KB)")
