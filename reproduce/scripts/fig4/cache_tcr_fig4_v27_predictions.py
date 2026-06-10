#!/usr/bin/env python3
"""Cache v2.7 subset predictions for TCR Fig 4.

Splitting strategies:
  - epitope splitting: group by peptide identity, balance by count
  - distance splitting: equal-size bins by distance

Prediction protocols:
  - CT: LOO across 6 test sets; cal restricted to v3+v4 only
  - CV: within-fold halfsplit (interleaved by distance rank)

S2DD distance strategy: uniform+znorm_sum (per_epitope strategy).
  Uniform wins all 6 TCR per-epitope cells (+0.007 to +0.049 r vs sigma_C).
  See `Manuscript/scripts_manuscript/audit_uniform_vs_sigmaC.py` in the research repo for verification (not bundled in this public release).

Uses predict_subset_metric from calipper.core (v2.7).
"""
import os, sys, time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from calipper.core import predict_subset_metric
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUFFIX_UNIFORM, DIST_SUBDIR

# min_bin_samples inside predict_subset_metric is 30 (the absolute floor).
# For TCR, cal subsets with ≥256 test samples have ~256 cal samples (halfsplit),
# so this threshold is never the bottleneck. The binding constraint is ≥4 cal subsets.
MIN_EP_CAL = 30  # matches min_bin_samples in s2dd_v2_7.py

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig3_fig4_tcr_cache')
os.makedirs(CACHE, exist_ok=True)

TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')

N_SUB = 8
MIN_SAMPLES = 30
METRICS = ['aucroc', 'ap']
MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
CAL_SETS = ['v3_combined', 'v4_combined']  # only v3+v4 for calibration

LABEL_COL = {
    'nettcr': ('binder', 'prediction'),
    'atm_tcr': ('y_true', 'y_prob'),
    'blosum_rf': ('binder', 'prediction'),
    'ergo_ii': ('y_true', 'y_prob'),
    'tcrbert': ('y_true', 'y_prob'),
}
PEPTIDE_COL = {
    'nettcr': 'peptide', 'atm_tcr': 'peptide', 'blosum_rf': 'peptide',
    'ergo_ii': 'peptide', 'tcrbert': 'peptide',
}


def _split_by_distance(distances, n_sub=N_SUB):
    si = np.argsort(distances)
    bs = len(si) // n_sub
    if bs < MIN_SAMPLES:
        return {}
    return {f'dist_bin{i}': si[i*bs:(len(si) if i == n_sub-1 else (i+1)*bs)]
            for i in range(n_sub)}


def _split_by_epitope(epitopes, distances, min_per_epitope=128):
    """Split by individual epitope identity. Each subset = one unique epitope.

    Only includes epitopes with >= min_per_epitope samples.
    Updated 2026-04-27: threshold lowered from 256 to 128 so that epitope-bin
    curve fitting (≥4 regression points) fires in all CV folds.
    """
    result = {}
    unique_eps = pd.Series(epitopes).value_counts()
    valid_eps = unique_eps[unique_eps >= min_per_epitope].index
    for ep in valid_eps:
        mask = epitopes == ep
        idx = np.where(mask)[0]
        result[ep[:30]] = idx  # truncate long peptide names for key
    return result


def load_tcr_ct(model, ts):
    lc, pc = LABEL_COL[model]
    pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                             f'{ts}_predictions_with_label.csv')
    # Load BOTH distance types: uniform for epitope strategy, sigma_C for distance strategy
    dist_uniform = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX_UNIFORM[DIST_TYPE]}')
    dist_sigmac = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
    if not os.path.exists(pred_path) or not os.path.exists(dist_uniform):
        return None
    te = pd.read_csv(pred_path)
    d_un = np.load(dist_uniform)
    n = min(len(d_un), len(te))
    lc_act = lc if lc in te.columns else ('binder' if 'binder' in te.columns else 'y_true')
    pc_act = pc if pc in te.columns else ('prediction' if 'prediction' in te.columns else 'y_prob')
    pep_col = PEPTIDE_COL.get(model, 'peptide')
    result = {
        'label': te[lc_act].values[:n].astype(int),
        'pred': te[pc_act].values[:n].astype(float),
        'dist_epitope': d_un[:n].astype(float),   # uniform for epitope strategy
        'epitope': te[pep_col].values[:n] if pep_col in te.columns else None,
    }
    if os.path.exists(dist_sigmac):
        d_sc = np.load(dist_sigmac)
        result['dist_distance'] = d_sc[:n].astype(float)  # sigma_C for distance strategy
    else:
        result['dist_distance'] = result['dist_epitope']   # fallback
    return result


def load_tcr_cv_fold(model, fold):
    lc, pc = LABEL_COL[model]
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    if not os.path.exists(test_path):
        return None
    parts = [pd.read_csv(test_path)]
    for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fold_dir, vname)
        if os.path.exists(vp):
            parts.append(pd.read_csv(vp))
            break
    df = pd.concat(parts, ignore_index=True)
    n_df = len(df)
    lc_act = lc if lc in df.columns else ('binder' if 'binder' in df.columns else 'y_true')
    pc_act = pc if pc in df.columns else ('prediction' if 'prediction' in df.columns else 'y_prob')
    pep_col = PEPTIDE_COL.get(model, 'peptide')

    # Load uniform distances (for epitope strategy)
    d_epitope = None
    dp_un = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{DIST_SUFFIX_UNIFORM[DIST_TYPE]}')
    if os.path.exists(dp_un):
        d_epitope = np.load(dp_un)

    # Load sigma_C distances (for distance strategy)
    d_distance = None
    if DIST_TYPE == 'blosum-sqrt':
        suffixes = [DIST_SUFFIX[DIST_TYPE].replace('.npy', '')]  # '_blosumsqrt_dist'
    else:
        suffixes = ['_combined_dist', '_dist']
    for suffix in suffixes:
        dp_sc = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dp_sc):
            d_distance = np.load(dp_sc)
            break

    if d_epitope is None and d_distance is None:
        return None
    if d_epitope is None:
        d_epitope = d_distance
    if d_distance is None:
        d_distance = d_epitope

    n = min(len(d_epitope), len(d_distance), n_df)
    result = {
        'label': df[lc_act].values[:n].astype(int),
        'pred': df[pc_act].values[:n].astype(float),
        'dist_epitope': d_epitope[:n].astype(float),
        'dist_distance': d_distance[:n].astype(float),
        'epitope': df[pep_col].values[:n] if pep_col in df.columns else None,
    }
    return result


def make_subsets(data, idx_map, dist_key='dist_epitope'):
    """Build subsets using strategy-specific distance key."""
    return {name: (data['label'][idx], data['pred'][idx], data[dist_key][idx])
            for name, idx in idx_map.items()}


# ── Load data ──
print("Loading TCR data...")
t0 = time.time()

# CT data: all 5 models × 6 test sets
ct_all = {}
for model in MODELS:
    ct_all[model] = {}
    for ts in CT_SETS:
        data = load_tcr_ct(model, ts)
        if data is not None:
            ct_all[model][ts] = data

# CV data: all 5 models × 5 folds
cv_all = {}
for model in MODELS:
    cv_all[model] = {}
    for fold in range(5):
        data = load_tcr_cv_fold(model, fold)
        if data is not None:
            cv_all[model][fold] = data

print(f"  CT: {sum(len(v) for v in ct_all.values())} model-testset combos")
print(f"  CV: {sum(len(v) for v in cv_all.values())} model-fold combos")
print(f"  Loaded in {time.time()-t0:.0f}s\n")


# ── CT subset prediction (v3+v4 as cal) ──
# Epitope strategy: pool ALL test sets, split by epitope (each epitope = one prediction target)
# Distance strategy: per-test-set LOO (each test set split by distance bins)
print("=== TCR CT subset prediction ===")
for strategy in ['epitope', 'distance']:
    for metric in METRICS:
        results = []
        for model in MODELS:
            ct = ct_all[model]
            cal_sets = [s for s in CAL_SETS if s in ct]
            if len(cal_sets) < 1:
                continue

            # Select strategy-specific distance key
            dist_key = 'dist_epitope' if strategy == 'epitope' else 'dist_distance'

            if strategy == 'epitope':
                # Pool NON-CAL test sets only, split by epitope
                # v3+v4 are cal — must NOT appear in test pool (data leakage)
                cal_data = {s: (ct[s]['label'], ct[s]['pred'], ct[s][dist_key])
                            for s in cal_sets}
                test_keys = [s for s in CT_SETS if s in ct and s not in CAL_SETS]
                if not test_keys:
                    continue
                test_y = np.concatenate([ct[s]['label'] for s in test_keys])
                test_p = np.concatenate([ct[s]['pred'] for s in test_keys])
                test_d = np.concatenate([ct[s][dist_key] for s in test_keys])
                test_ep = np.concatenate([ct[s]['epitope'] for s in test_keys])
                # Track source for seen/unseen labeling
                test_src = np.concatenate([np.full(len(ct[s]['label']), s) for s in test_keys])

                idx_map = _split_by_epitope(test_ep, test_d)
                if not idx_map:
                    continue
                test_data = {'label': test_y, 'pred': test_p, dist_key: test_d}
                subsets = make_subsets(test_data, idx_map, dist_key=dist_key)

                # Build cal_subsets from cal data
                cal_y_all = np.concatenate([v[0] for v in cal_data.values()])
                cal_p_all = np.concatenate([v[1] for v in cal_data.values()])
                cal_d_all = np.concatenate([v[2] for v in cal_data.values()])
                cal_ep_all = np.concatenate([ct[s]['epitope'] for s in cal_sets
                                              if ct[s]['epitope'] is not None])
                cal_subs = {}
                for sub_name in idx_map:
                    mask = np.array([str(e)[:30] == sub_name for e in cal_ep_all])
                    if mask.sum() >= MIN_EP_CAL:
                        cal_subs[sub_name] = (cal_y_all[mask], cal_p_all[mask], cal_d_all[mask])
                if len(cal_subs) >= 4:
                    preds = predict_subset_metric(cal_data, subsets, metrics=[metric],
                                                   bin_strategy='subset', cal_subsets=cal_subs)
                else:
                    preds = predict_subset_metric(cal_data, subsets, metrics=[metric])

                # Determine seen/unseen for each epitope subset
                from collections import Counter
                for r in preds:
                    sub_name = r['subset']
                    sub_idx = idx_map[sub_name]
                    sources = test_src[sub_idx]
                    src_counts = Counter(sources)
                    primary_src = src_counts.most_common(1)[0][0]
                    r['source'] = primary_src
                    r['seen'] = 'seen' if primary_src == 'seen_test' else 'unseen'
                    r['model'] = model
                    r['mean_dist'] = float(subsets[r['subset']][2].mean())
                results.extend(preds)

            else:
                # Distance strategy: per-test-set LOO, sigma_C distances
                for held in CT_SETS:
                    if held not in ct:
                        continue
                    cal_keys = [s for s in cal_sets if s != held]
                    if not cal_keys:
                        continue
                    cal_data = {s: (ct[s]['label'], ct[s]['pred'], ct[s][dist_key])
                                for s in cal_keys}
                    test = ct[held]
                    idx_map = _split_by_distance(test[dist_key])
                    if not idx_map:
                        continue
                    subsets = make_subsets(test, idx_map, dist_key=dist_key)
                    preds = predict_subset_metric(cal_data, subsets, metrics=[metric])
                    for r in preds:
                        r['source'] = held
                        r['model'] = model
                        r['mean_dist'] = float(subsets[r['subset']][2].mean())
                    results.extend(preds)

        df = pd.DataFrame(results)
        out = os.path.join(CACHE, f'tcr_fig4_{DIST_TYPE}_ct_{strategy}_{metric}.csv')
        df.to_csv(out, index=False)
        if len(df) >= 3:
            from scipy.stats import pearsonr
            r, _ = pearsonr(df['predicted'], df['actual'])
            mae = np.abs(df['predicted'] - df['actual']).mean()
            print(f"  CT {strategy:8s} {metric:6s}: R={r:.3f}, MAE={mae:.3f}, n={len(df)}")


# ── CV halfsplit prediction ──
print("\n=== TCR CV halfsplit subset prediction ===")
for strategy in ['epitope', 'distance']:
    for metric in METRICS:
        results = []
        # Select strategy-specific distance key
        dist_key = 'dist_epitope' if strategy == 'epitope' else 'dist_distance'

        for model in MODELS:
            for fold in sorted(cv_all[model].keys()):
                data = cv_all[model][fold]
                # Distance-interleaved halfsplit using REFERENCE model (nettcr) sigma_C distances
                # so all models get the same cal/test split for the same fold.
                n_cur = len(data['label'])
                ref_data = cv_all.get('nettcr', {}).get(fold)
                if ref_data is not None and len(ref_data['dist_distance']) == n_cur:
                    si = np.argsort(ref_data['dist_distance'])
                else:
                    si = np.argsort(data['dist_distance'])  # fallback
                cal_idx, val_idx = si[::2], si[1::2]

                cal_data = {'cal': (data['label'][cal_idx], data['pred'][cal_idx],
                                    data[dist_key][cal_idx])}

                val_label = data['label'][val_idx]
                val_pred = data['pred'][val_idx]
                val_dist = data[dist_key][val_idx]
                val_ep = data['epitope'][val_idx] if data['epitope'] is not None else None

                if strategy == 'distance':
                    idx_map = _split_by_distance(val_dist)
                else:
                    if val_ep is None:
                        continue
                    idx_map = _split_by_epitope(val_ep, val_dist)

                if not idx_map:
                    continue

                val_data = {'label': val_label, 'pred': val_pred, dist_key: val_dist}
                subsets = make_subsets(val_data, idx_map, dist_key=dist_key)

                # For epitope strategy: build cal_subsets and use bin_strategy='subset'
                if strategy == 'epitope':
                    cal_ep = data['epitope'][cal_idx] if data['epitope'] is not None else None
                    cal_subs = {}
                    if cal_ep is not None:
                        for sub_name in idx_map:
                            mask = np.array([str(e)[:30] == sub_name for e in cal_ep])
                            if mask.sum() >= MIN_EP_CAL:
                                cal_subs[sub_name] = (data['label'][cal_idx][mask],
                                                       data['pred'][cal_idx][mask],
                                                       data[dist_key][cal_idx][mask])
                    if len(cal_subs) >= 4:
                        preds = predict_subset_metric(cal_data, subsets, metrics=[metric],
                                                       bin_strategy='subset', cal_subsets=cal_subs)
                    else:
                        preds = predict_subset_metric(cal_data, subsets, metrics=[metric])
                else:
                    preds = predict_subset_metric(cal_data, subsets, metrics=[metric])

                for r in preds:
                    r['fold'] = fold
                    r['model'] = model
                    r['mean_dist'] = float(subsets[r['subset']][2].mean())
                results.extend(preds)

        df = pd.DataFrame(results)
        out = os.path.join(CACHE, f'tcr_fig4_{DIST_TYPE}_cv_{strategy}_{metric}.csv')
        df.to_csv(out, index=False)
        if len(df) >= 3:
            from scipy.stats import pearsonr
            r, _ = pearsonr(df['predicted'], df['actual'])
            mae = np.abs(df['predicted'] - df['actual']).mean()
            print(f"  CV {strategy:8s} {metric:6s}: R={r:.3f}, MAE={mae:.3f}, n={len(df)}")


print(f"\nDone in {time.time()-t0:.0f}s")
