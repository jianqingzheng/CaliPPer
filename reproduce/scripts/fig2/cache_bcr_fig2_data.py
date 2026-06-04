#!/usr/bin/env python3
"""Pre-compute and cache ALL data needed for BCR Fig 2 (4×4).

Caches:
  1. CV degradation bins: 5 models × 5 folds → bcr_cv_{model}_fold{fold}_bins.npz
  2. CT degradation bins: XBCR-net 5 folds × 4 test sets → bcr_ct_fold{fold}_{testset}_bins.npz
  3. UMAP embeddings: antigen + antibody → bcr_umap_antigen.npz, bcr_umap_antibody.npz
  4. Heatmap data: already in cross_model_detail.csv (no caching needed)
  5. CT per-test-set for other models → bcr_ct_{model}_{testset}_bins.npz

All distances: sigma_C 3-chain (Heavy + Light + variant_seq), k=0.1, b=0.03, K=30.
"""
import os, sys, time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ → fig2/ → panels/ → designed_figures/ → Manuscript/ → general_eval/
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from calipper.combine_first_helpers import compute_chain_weights, compute_combine_first_distances
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, DIST_SUBDIR, BCR_DIST_MODE

RESULTS = os.path.join(INPUT_DIR, 'results')
# Separate cache per distance type to avoid overwriting
CACHE = os.path.join(RESULTS, 'fig2_bcr_cache' if DIST_TYPE == 'lev-log' else 'fig2_bcr_cache_blosumsqrt')
os.makedirs(CACHE, exist_ok=True)

CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
k, b, K = 0.1, 0.03, 30
N_BINS = 8
BCR_MODELS = ['xbcr_net', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_MODEL_DIRS = {
    'xbcr_net': 'xbcr', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
    'mint': 'mint', 'rleaai': 'rleaai',
}
CT_SETS = ['A1-A11', 'BNT162b2', 'guoyu', 'unseen', 'flu']  # sources in combined_bind_ab_cv test data


def bin_and_eval(dists, y, p, n_bins=N_BINS):
    """Bin by distance and evaluate AUROC, AP, F1 per bin."""
    si = np.argsort(dists)
    bs = len(si) // n_bins
    bx, by_auc, by_ap, by_f1 = [], [], [], []
    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        bx.append(dists[idx].mean())
        by_auc.append(safe_metric('aucroc', y[idx], p[idx]))
        by_ap.append(safe_metric('ap', y[idx], p[idx]))
        by_f1.append(safe_metric('f1', y[idx], p[idx]))
    return np.array(bx), np.array(by_auc), np.array(by_ap), np.array(by_f1)


# ═══════════════════════════════════════════
# 1. CV degradation bins (5 models × 5 folds)
# ═══════════════════════════════════════════
print("=== Caching CV degradation bins ===")
t0 = time.time()

for model_key in BCR_MODELS:
    model_dir = BCR_MODEL_DIRS[model_key]
    for fold in range(5):
        cache_path = os.path.join(CACHE, f'bcr_cv_{model_key}_fold{fold}_bins.npz')
        if os.path.exists(cache_path):
            print(f'  [CACHED] {model_key} fold{fold}')
            continue

        tp = os.path.join(RESULTS, model_dir, 'combined_bind_ab_cv', f'fold{fold}', 'train.csv')
        tep = os.path.join(RESULTS, model_dir, 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
        if not os.path.exists(tp) or not os.path.exists(tep):
            print(f'  [SKIP] {model_key} fold{fold}: missing data')
            continue

        tr = pd.read_csv(tp)
        te = pd.read_csv(tep)

        # Compute distances (Levenshtein or BLOSUM-sqrt)
        w, _ = compute_chain_weights(tr, CHAIN_COLS, k, b, K, formula='sigma_C')
        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            npy = os.path.join(RESULTS, 'fig2_cache', f'{model_dir}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
            d = np.load(npy).astype(float)[:len(te)] if os.path.exists(npy) else compute_combine_first_distances(te, tr, CHAIN_COLS, w, k, b, K)
        else:
            d = compute_combine_first_distances(te, tr, CHAIN_COLS, w, k, b, K)

        y = te['rbd'].values.astype(int)
        p_vals = te['pred_prob'].values.astype(float)

        bx, by_auc, by_ap, by_f1 = bin_and_eval(d, y, p_vals)

        np.savez(cache_path, bx=bx, by_auroc=by_auc, by_ap=by_ap, by_f1=by_f1,
                 distances=d, n_samples=len(te))
        print(f'  [OK] {model_key} fold{fold} ({len(te)} samples, {time.time()-t0:.0f}s elapsed)')

print(f"CV caching done ({time.time()-t0:.0f}s)\n")


# ═══════════════════════════════════════════
# 2. CT degradation bins (XBCR-net per-source from combined_bind_ab_cv)
# ═══════════════════════════════════════════
# CORRECTED 2026-04-23: Use the SAME 2-pathogen binding model (combined_bind_ab_cv)
# for CT, split by source column. The previous version incorrectly used
# neu_archive_crosstest_f1 which is a DIFFERENT model (neutralization archive).
# The combined_bind_ab_cv test data contains per-source labels (A1-A11, unseen,
# guoyu, BNT162b2, flu, xbcr_train) that serve as independent test subsets.
print("=== Caching CT degradation bins (XBCR-net, per-source from combined_bind_ab_cv) ===")
t1 = time.time()

# Map source names to display names for CT
CT_SOURCE_MAP = {
    'A1-A11': 'A1-A11', 'unseen': 'unseen', 'guoyu': 'guoyu',
    'BNT162b2': 'BNT162b2', 'flu': 'flu',
}

for fold in range(5):
    train_path = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'train.csv')
    test_path = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        continue

    tr = pd.read_csv(train_path)
    te_full = pd.read_csv(test_path)

    # Compute distances (Levenshtein or BLOSUM-sqrt)
    w, _ = compute_chain_weights(tr, CHAIN_COLS, k, b, K, formula='sigma_C')
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        npy = os.path.join(RESULTS, 'fig2_cache', f'xbcr_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        d_full = np.load(npy).astype(float)[:len(te_full)] if os.path.exists(npy) else compute_combine_first_distances(te_full, tr, CHAIN_COLS, w, k, b, K)
    else:
        d_full = compute_combine_first_distances(te_full, tr, CHAIN_COLS, w, k, b, K)

    for source_name, display_name in CT_SOURCE_MAP.items():
        cache_path = os.path.join(CACHE, f'bcr_ct_fold{fold}_{display_name}_bins.npz')
        if os.path.exists(cache_path):
            print(f'  [CACHED] fold{fold} {display_name}')
            continue

        # Filter test data by source
        mask = te_full['source'] == source_name
        if mask.sum() < N_BINS * 5:  # need enough samples for binning
            print(f'  [SKIP] fold{fold} {display_name}: only {mask.sum()} samples')
            continue

        te = te_full[mask].reset_index(drop=True)
        d = d_full[mask.values]
        y = te['rbd'].values.astype(int)
        p_vals = te['pred_prob'].values.astype(float)

        bx, by_auc, by_ap, by_f1 = bin_and_eval(d, y, p_vals)

        np.savez(cache_path, bx=bx, by_auroc=by_auc, by_ap=by_ap, by_f1=by_f1,
                 distances=d, n_samples=len(te))
        print(f'  [OK] fold{fold} {display_name} ({len(te)} samples)')

print(f"CT caching done ({time.time()-t1:.0f}s)\n")


# ═══════════════════════════════════════════
# 3. UMAP embeddings
# ═══════════════════════════════════════════
print("=== Caching UMAP embeddings ===")
t2 = time.time()

# Load XBCR-net fold0 for UMAPs
cv_dir = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv')
train0 = pd.read_csv(os.path.join(cv_dir, 'fold0', 'train.csv'))
test0 = pd.read_csv(os.path.join(cv_dir, 'fold0', 'test.csv'))

# 3a. Antigen UMAP
antigen_cache = os.path.join(CACHE, 'bcr_umap_antigen.npz')
if not os.path.exists(antigen_cache):
    from Levenshtein import ratio as lev_ratio
    all_variants = pd.concat([train0['variant_seq'], test0['variant_seq']]).unique()
    n_var = len(all_variants)
    sim = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            r = lev_ratio(all_variants[i], all_variants[j])
            sim[i, j] = r; sim[j, i] = r

    from umap import UMAP
    emb = UMAP(n_components=2, metric='precomputed', random_state=42,
               n_neighbors=min(15, n_var - 1), min_dist=0.5).fit_transform(1 - sim)

    is_flu = np.array([v[:2] in ('MK', 'ME', 'MN', 'MA') for v in all_variants])
    var_counts = pd.concat([train0, test0]).groupby('variant_seq').size().to_dict()
    sizes = np.array([var_counts.get(v, 5) for v in all_variants])

    np.savez(antigen_cache, emb=emb, is_flu=is_flu, sizes=sizes,
             variants=all_variants, n_var=n_var)
    print(f'  [OK] Antigen UMAP ({n_var} variants)')
else:
    print('  [CACHED] Antigen UMAP')

# 3b. Antibody Heavy chain UMAP
antibody_cache = os.path.join(CACHE, 'bcr_umap_antibody.npz')
if not os.path.exists(antibody_cache):
    from Levenshtein import ratio as lev_ratio
    rng = np.random.RandomState(42)
    n_umap = min(2000, len(test0))
    sub_idx = rng.choice(len(test0), n_umap, replace=False)
    test_sub = test0.iloc[sub_idx].reset_index(drop=True)
    heavies = test_sub['Heavy'].values
    n_fit = min(1200, len(heavies))
    fit_idx = rng.choice(len(heavies), n_fit, replace=False)
    h_sub = heavies[fit_idx]

    sim_h = np.zeros((n_fit, n_fit))
    for i in range(n_fit):
        for j in range(i, n_fit):
            r = lev_ratio(h_sub[i], h_sub[j])
            sim_h[i, j] = r; sim_h[j, i] = r

    from umap import UMAP
    emb_h = UMAP(n_components=2, metric='precomputed', random_state=42,
                 n_neighbors=min(15, n_fit - 1), min_dist=0.3).fit_transform(1 - sim_h)

    sources = test_sub['source'].iloc[fit_idx].values

    np.savez(antibody_cache, emb=emb_h, sources=sources, n_fit=n_fit)
    print(f'  [OK] Antibody UMAP ({n_fit} sequences)')
else:
    print('  [CACHED] Antibody UMAP')

print(f"UMAP caching done ({time.time()-t2:.0f}s)\n")


# ═══════════════════════════════════════════
# 4. Compute per-fold Pearson r for heatmap (cache if not in detail CSV)
# ═══════════════════════════════════════════
print("=== Caching heatmap data ===")
heatmap_cache = os.path.join(CACHE, 'bcr_heatmap_data.npz')
if not os.path.exists(heatmap_cache):
    detail_path = os.path.join(RESULTS, 'bcr_cross_model_comparison', 'cross_model_detail.csv')
    if os.path.exists(detail_path):
        detail = pd.read_csv(detail_path)
        # Build heatmaps for AP and AUROC
        hm_ap = np.full((5, 5), np.nan)
        hm_auroc = np.full((5, 5), np.nan)
        for i, m_key in enumerate(BCR_MODELS):
            for metric, hm in [('ap', hm_ap), ('aucroc', hm_auroc)]:
                sub = detail[(detail['model'] == m_key) &
                             (detail['scope'] == 'combined') &
                             (detail['metric'] == metric)]
                for _, row in sub.iterrows():
                    j = int(row['fold'])
                    if 0 <= j < 5:
                        hm[i, j] = row['pearson_r']
        np.savez(heatmap_cache, hm_ap=hm_ap, hm_auroc=hm_auroc)
        print(f'  [OK] Heatmap data ({np.sum(~np.isnan(hm_ap))}/25 AP, {np.sum(~np.isnan(hm_auroc))}/25 AUROC)')
    else:
        print('  [SKIP] No cross_model_detail.csv')
else:
    print('  [CACHED] Heatmap data')


# ═══════════════════════════════════════════
# 5. Summary statistics cache
# ═══════════════════════════════════════════
print("=== Caching summary stats ===")
summary_cache = os.path.join(CACHE, 'bcr_summary_stats.npz')
if not os.path.exists(summary_cache):
    summary_path = os.path.join(RESULTS, 'bcr_cross_model_comparison', 'comparison_summary.csv')
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        # Extract mean |r| ± std for each model × metric (combined scope)
        model_metric_r = {}
        model_metric_std = {}
        for m_key in BCR_MODELS:
            for metric in ['aucroc', 'ap', 'f1']:
                sub = summary[(summary['model'] == m_key) &
                              (summary['scope'] == 'combined') &
                              (summary['metric'] == metric)]
                if len(sub) > 0:
                    model_metric_r[f'{m_key}_{metric}'] = float(sub.iloc[0]['mean_abs_r'])
                    model_metric_std[f'{m_key}_{metric}'] = float(sub.iloc[0]['std']) if 'std' in sub.columns else 0
        np.savez(summary_cache, **{f'r_{k}': v for k, v in model_metric_r.items()},
                 **{f'std_{k}': v for k, v in model_metric_std.items()})
        print(f'  [OK] Summary stats ({len(model_metric_r)} entries)')
    else:
        print('  [SKIP] No comparison_summary.csv')
else:
    print('  [CACHED] Summary stats')


# ═══════════════════════════════════════════
# 6. Antigen diversity effect data
# ═══════════════════════════════════════════
diversity_cache = os.path.join(CACHE, 'bcr_diversity_effect.npz')
if not os.path.exists(diversity_cache):
    # SARS-only vs SARS+Flu mean |r| from comparison_summary
    # These values are from the existing BCR panels script
    sars_only = {'aucroc': 0.214, 'ap': 0.727, 'f1': 0.711}
    sars_flu = {'aucroc': 0.455, 'ap': 0.897, 'f1': 0.935}
    np.savez(diversity_cache,
             sars_only_auroc=sars_only['aucroc'], sars_only_ap=sars_only['ap'], sars_only_f1=sars_only['f1'],
             sars_flu_auroc=sars_flu['aucroc'], sars_flu_ap=sars_flu['ap'], sars_flu_f1=sars_flu['f1'])
    print('  [OK] Diversity effect data')
else:
    print('  [CACHED] Diversity effect data')


print(f"\n=== All caching complete ===")
print(f"Cache directory: {CACHE}")
print(f"Files: {len(os.listdir(CACHE))}")
for f in sorted(os.listdir(CACHE)):
    size = os.path.getsize(os.path.join(CACHE, f))
    print(f"  {f} ({size/1024:.1f} KB)")
