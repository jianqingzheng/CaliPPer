#!/usr/bin/env python3
"""Cache BCR Fig 4 subset-level predictions in TCR-compatible format.

Produces CSVs with columns matching TCR cache:
  CT: [subset, metric, predicted, actual, n, prevalence, source, model, mean_dist]
  CV: [subset, metric, predicted, actual, n, prevalence, fold, model, mean_dist]

CT design: per-variant LOO within SARS/flu domains (fold4 model, fold4+external pool)
CV design: within-fold halfsplit from combined_bind_ab_cv

Splitting strategies:
  - distance: 8 equal-sized distance bins per cal set
  - antigen: per-variant groups (analogous to TCR epitope splitting)
"""
import os, sys, time, hashlib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, BCR_DIST_MODE, get_bcr_ct_distance

def vhash(seq):
    """Hash variant sequence to unique short key (avoids [:20] collision)."""
    return hashlib.md5(str(seq).encode()).hexdigest()[:12]
from calipper.core import predict_subset_metric

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
os.makedirs(CACHE, exist_ok=True)

FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
METRICS = ['aucroc', 'ap']
MIN_SAMPLES = 30
N_SUB = 8

t0 = time.time()

# ═══════════════════════════════════════════
# CT: Per-variant LOO within SARS/flu domains
# ═══════════════════════════════════════════
print("=== BCR CT subset prediction (fold4cal, per-variant LOO) ===")

for strategy in ['distance', 'antigen']:
    for metric in METRICS:
        all_results = []

        for model in MODELS:
            # Load cal + external, pool
            cal_path = os.path.join(FOLD4CAL, model, 'cal_predictions.csv')
            if not os.path.exists(cal_path):
                continue
            cal = pd.read_csv(cal_path)
            model_dir = os.path.join(FOLD4CAL, model)
            if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
            cal['source'] = 'fold4_test'

            parts = [cal]
            for ts in ['A1-A11', 'unseen', 'flu']:
                fp = os.path.join(FOLD4CAL, model, f'{ts}_predictions.csv')
                if not os.path.exists(fp):
                    continue
                df = pd.read_csv(fp)
                if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                    df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
                df['source'] = ts
                if 'data_source' not in df.columns:
                    df['data_source'] = 'flu' if ts == 'flu' else 'sars'
                parts.append(df)

            pooled = pd.concat(parts, ignore_index=True)

            for domain in ['sars', 'flu']:
                domain_df = pooled[pooled['data_source'] == domain]
                variants = domain_df.groupby('variant_seq').size()
                valid = variants[variants >= MIN_SAMPLES].index.tolist()

                for held_v in valid:
                    test_mask = domain_df['variant_seq'] == held_v
                    cal_mask = ~test_mask
                    cal_sub = domain_df[cal_mask]
                    test_sub = domain_df[test_mask]

                    if len(test_sub) < 10:
                        continue
                    test_y = test_sub['rbd'].values.astype(int)
                    if test_y.sum() == 0 or test_y.sum() == len(test_y):
                        continue
                    cal_y = cal_sub['rbd'].values.astype(int)
                    if cal_y.sum() < 3 or (len(cal_y) - cal_y.sum()) < 3:
                        continue

                    cal_data = {'cal': (cal_y,
                                        cal_sub['pred_prob'].values.astype(float),
                                        cal_sub['distance'].values.astype(float))}
                    test_p = test_sub['pred_prob'].values.astype(float)
                    test_d = test_sub['distance'].values.astype(float)

                    if strategy == 'distance':
                        # Distance bins within held-out variant
                        n_bins = min(N_SUB, len(test_sub) // MIN_SAMPLES)
                        if n_bins < 2:
                            continue
                        si = np.argsort(test_d)
                        bs = len(si) // n_bins
                        subsets = {}
                        for i in range(n_bins):
                            s = i * bs
                            e = len(si) if i == n_bins - 1 else (i + 1) * bs
                            idx = si[s:e]
                            subsets[f'dist_bin{i}'] = (test_y[idx], test_p[idx], test_d[idx])
                        preds = predict_subset_metric(cal_data, subsets, metrics=[metric])
                    else:
                        # For antigen strategy: held-out variant = one test subset,
                        # remaining variants = cal subsets for bin_strategy='subset'
                        subsets = {f'variant': (test_y, test_p, test_d)}

                        # Build cal_subsets from remaining variants in same domain
                        cal_variants = cal_sub['variant_seq'].unique()
                        cal_subs = {}
                        for cv in cal_variants:
                            cv_mask = cal_sub['variant_seq'] == cv
                            if cv_mask.sum() >= MIN_SAMPLES:
                                cv_df = cal_sub[cv_mask]
                                cv_y = cv_df['rbd'].values.astype(int)
                                if cv_y.sum() == 0 or cv_y.sum() == len(cv_y):
                                    continue
                                cal_subs[vhash(cv)] = (
                                    cv_y,
                                    cv_df['pred_prob'].values.astype(float),
                                    cv_df['distance'].values.astype(float))
                        if len(cal_subs) >= 4:
                            preds = predict_subset_metric(cal_data, subsets, metrics=[metric],
                                                           bin_strategy='subset', cal_subsets=cal_subs)
                        else:
                            preds = predict_subset_metric(cal_data, subsets, metrics=[metric])

                    for r in preds:
                        r['source'] = f'{domain}_{vhash(held_v)}'
                        r['model'] = model
                    all_results.extend(preds)

        df_out = pd.DataFrame(all_results)
        out_path = os.path.join(CACHE, f'bcr_fig4_fold4cal_ct_{strategy}_{metric}.csv')
        df_out.to_csv(out_path, index=False)
        print(f"  CT {strategy} {metric}: {len(df_out)} rows → {out_path}")

# ═══════════════════════════════════════════
# CV: Within-fold halfsplit (from combined_bind_ab_cv)
# ═══════════════════════════════════════════
print("\n=== BCR CV subset prediction (within-fold halfsplit) ===")

for strategy in ['distance', 'antigen']:
    for metric in METRICS:
        all_results = []

        for model in MODELS:
            model_dir = 'xbcr' if model == 'xbcr' else model
            for fold in range(5):
                test_path = os.path.join(RESULTS, model_dir, 'combined_bind_ab_cv',
                                          f'fold{fold}', 'test.csv')
                if not os.path.exists(test_path):
                    continue
                te = pd.read_csv(test_path)
                if 'pred_prob' not in te.columns or 'distance' not in te.columns:
                    continue

                y = te['rbd'].values.astype(int)
                p = te['pred_prob'].values.astype(float)
                if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                    npy_path = os.path.join(RESULTS, 'fig2_cache',
                                            f'{model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
                    if os.path.exists(npy_path):
                        d = np.load(npy_path).astype(float)[:len(te)]
                    else:
                        d = te['distance'].values.astype(float)
                else:
                    d = te['distance'].values.astype(float)

                # Distance-interleaved halfsplit using REFERENCE model (xbcr) distances
                # so all models get the same cal/test split for the same fold.
                ref_path = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv',
                                         f'fold{fold}', 'test.csv')
                if os.path.exists(ref_path) and model != 'xbcr':
                    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                        ref_npy = os.path.join(RESULTS, 'fig2_cache',
                                               f'xbcr_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
                        if os.path.exists(ref_npy):
                            ref_d = np.load(ref_npy).astype(float)
                            si = np.argsort(ref_d)
                        else:
                            ref_df = pd.read_csv(ref_path)
                            ref_d = ref_df['distance'].values.astype(float) if 'distance' in ref_df.columns else d
                            si = np.argsort(ref_d)
                    else:
                        ref_df = pd.read_csv(ref_path)
                        if 'distance' in ref_df.columns:
                            ref_d = ref_df['distance'].values.astype(float)
                            si = np.argsort(ref_d)
                        else:
                            si = np.argsort(d)
                else:
                    si = np.argsort(d)
                mid = len(si) // 2
                cal_idx = si[::2]      # interleaved even indices
                test_idx = si[1::2]    # interleaved odd indices

                cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
                test_y = y[test_idx]
                test_p = p[test_idx]
                test_d = d[test_idx]

                if strategy == 'distance':
                    n_bins = min(N_SUB, len(test_idx) // MIN_SAMPLES)
                    if n_bins < 2:
                        continue
                    si2 = np.argsort(test_d)
                    bs = len(si2) // n_bins
                    subsets = {}
                    for i in range(n_bins):
                        s = i * bs
                        e = len(si2) if i == n_bins - 1 else (i + 1) * bs
                        idx = si2[s:e]
                        subsets[f'dist_bin{i}'] = (test_y[idx], test_p[idx], test_d[idx])
                else:
                    # Antigen splitting: group by variant_seq
                    vs = te['variant_seq'].values[test_idx]
                    unique_vs = pd.Series(vs).value_counts()
                    valid_vs = unique_vs[unique_vs >= MIN_SAMPLES].index
                    subsets = {}
                    for ag in valid_vs:
                        ag_mask = vs == ag
                        if ag_mask.sum() < MIN_SAMPLES:
                            continue
                        subsets[vhash(ag)] = (test_y[ag_mask], test_p[ag_mask], test_d[ag_mask])

                if not subsets:
                    continue

                # For antigen strategy: build cal_subsets and use bin_strategy='subset'
                if strategy == 'antigen':
                    cal_vs = te['variant_seq'].values[cal_idx]
                    cal_subs = {}
                    for ag in subsets:
                        mask = np.array([vhash(v) == ag for v in cal_vs])
                        if mask.sum() >= MIN_SAMPLES:
                            cal_subs[ag] = (y[cal_idx][mask], p[cal_idx][mask], d[cal_idx][mask])
                    if len(cal_subs) >= 4:
                        preds = predict_subset_metric(cal_data, subsets, metrics=[metric],
                                                       bin_strategy='subset', cal_subsets=cal_subs)
                    else:
                        # Fallback to distance-based if too few cal subsets
                        preds = predict_subset_metric(cal_data, subsets, metrics=[metric])
                else:
                    preds = predict_subset_metric(cal_data, subsets, metrics=[metric])

                for r in preds:
                    r['fold'] = fold
                    r['model'] = model
                all_results.extend(preds)

        df_out = pd.DataFrame(all_results)
        out_path = os.path.join(CACHE, f'bcr_fig4_fold4cal_cv_{strategy}_{metric}.csv')
        df_out.to_csv(out_path, index=False)
        print(f"  CV {strategy} {metric}: {len(df_out)} rows → {out_path}")

print(f"\nDone in {time.time()-t0:.0f}s")
