#!/usr/bin/env python3
"""Fig 2 panels m-n: Distance metric comparison boxplots for TCR-CV and BCR-CV.

TCR CV: 5 models x 5 folds — Lev-log, BLOSUM-sqrt, ESM2
BCR CV: 5 models x 5 folds — Lev-log, BLOSUM-sqrt

Distances are model-independent (same for all models per fold), so ESM2
are computed once per fold and reused. ESM2 uses cached embeddings (no GPU needed).
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style
from calipper.general_evaluator import safe_metric
from calipper.pluggable_distance import compute_s2dd_pluggable, make_sw_blosum62_similarity

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig2_cache')
ESM2_CACHE = os.path.join(RESULTS, 'multi_distance_comparison', 'cache', 'esm2_all_embeddings.npz')
N_BINS = 8
METRIC = 'ap'
K = 50
CHAINS = ['peptide', 'CDR3a', 'CDR3b']


def load_tcr_cv_combined(model, fold):
    """Load TCR CV test+val combined predictions (matching cached distance arrays)."""
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
    return pd.concat(parts, ignore_index=True)


def compute_binned_r(y, p, d, n_bins=8, metric='ap'):
    """Compute Pearson r between binned distance and metric."""
    n = min(len(d), len(y))
    y, p, d = y[:n], p[:n], d[:n]
    si = np.argsort(d)
    bs = len(si) // n_bins
    bv, bd = [], []
    for i in range(n_bins):
        s, e = i * bs, (len(si) if i == n_bins - 1 else (i + 1) * bs)
        idx = si[s:e]
        bv.append(safe_metric(metric, y[idx], p[idx]))
        bd.append(d[idx].mean())
    v, dd = np.array(bv), np.array(bd)
    valid = ~np.isnan(v)
    if valid.sum() >= 4:
        r, _ = pearsonr(dd[valid], v[valid])
        return r
    return None


COL_MAP = {'peptide': 'epitope', 'CDR3a': 'cdr3_a', 'CDR3b': 'cdr3_b'}


def _get_col(df, target):
    """Get column from df, trying target first, then mapped alternative."""
    if target in df.columns:
        return target
    alt = COL_MAP.get(target)
    if alt and alt in df.columns:
        return alt
    return None


def compute_esm2_cv_dist(test_df, train_df, fold, chains=CHAINS):
    """ESM2 S2DD aligned with the canonical Lev S2DD pipeline.

    Mirrors `combine_first_helpers.compute_combine_first_distances`
    (combine_method='weighted_max_znorm', sigma_C weights, top-K reduction,
    full-pair z-norm stats per `compute_pairwise_chain_stats`) but swaps the
    base per-pair distance to the LogDist-transformed ESM2 cosine distance:
    d(q,r) = log(k * (1 - cos_sim(q,r) + b)), with cos_sim on L2-normalised
    mean-pooled embeddings and k=0.1, b=0.1 (same params as the Lev LogDist).
    Cosine is the natural similarity for model embeddings; the log transform
    matches the canonical per-pair LogDist form so ESM2 is treated exactly
    like Lev (per-pair log -> full-pair z-norm -> weighted_max_znorm -> topK).
    The post-hoc global log previously applied downstream is removed (callers
    consume this combined distance directly).
    """
    K_LOG, B_LOG = 0.1, 0.1   # LogDist params, identical to Lev S2DD
    cache_path = os.path.join(CACHE, f'esm2_cv_fold{fold}_dist.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Load cached embeddings
    npz = np.load(ESM2_CACHE, allow_pickle=True)
    embeddings = dict(zip(npz['seqs'].tolist(), npz['embs']))
    dim = 1280
    zero = np.zeros(dim)

    # Per-chain embeddings + train-row expansion + full-pair z-norm stats.
    test_embs_list, train_embs_list, train_idx_list = [], [], []
    chain_stats = []          # (mu, sigma) over ALL train-vs-train pairs
    used_train_cols = []
    rng = np.random.RandomState(42)
    for col in chains:
        test_col = _get_col(test_df, col)
        train_col = _get_col(train_df, col)
        if test_col is None or train_col is None:
            continue
        test_seqs = test_df[test_col].astype(str).tolist()
        train_seqs = train_df[train_col].astype(str).tolist()
        unique_train = list(dict.fromkeys(train_seqs))
        u_index = {s: i for i, s in enumerate(unique_train)}

        test_embs = np.array([embeddings.get(s, zero) for s in test_seqs])
        train_embs = np.array([embeddings.get(s, zero) for s in unique_train])
        train_idx = np.array([u_index[s] for s in train_seqs])

        # Cosine base metric: L2-normalise embeddings, then per-pair distance
        # is 1 - cosine similarity. ESM2 mean-pooled embeddings carry large
        # norm variation that pollutes raw Euclidean; cosine is direction-only.
        def _l2norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            return a / np.maximum(n, 1e-12)
        test_embs = _l2norm(test_embs)
        train_embs = _l2norm(train_embs)

        test_embs_list.append(test_embs)
        train_embs_list.append(train_embs)
        train_idx_list.append(train_idx)
        used_train_cols.append(train_col)

        # Full-pair z-norm stats: subsample <=500 training ROWS (WITH
        # multiplicity, like compute_pairwise_chain_stats which samples from
        # n_train raw rows, NOT from unique sequences) as queries vs ALL
        # train rows, flatten every pair (no top-K).
        n_rows = len(train_idx)
        n_sub = min(500, n_rows)
        sub_row = rng.choice(n_rows, n_sub, replace=False)
        sub_embs = train_embs[train_idx[sub_row]]               # row->unique emb
        sub_cos = 1.0 - sub_embs @ train_embs.T                 # cosine dist
        sub_d = np.log(K_LOG * (sub_cos + B_LOG))               # LogDist form
        sub_full = sub_d[:, train_idx].ravel()
        mu = float(sub_full.mean())
        sigma = max(float(sub_full.std()), 1e-9)
        chain_stats.append((mu, sigma))

    n_chains = len(test_embs_list)

    # sigma_C chain weights (Simpson concentration x z-norm sigma), as
    # selectors only (weighted_max_znorm rescales weights to unit mean).
    esm2_weights = np.zeros(n_chains)
    for c, train_col_name in enumerate(used_train_cols):
        seqs = train_df[train_col_name].astype(str).tolist()
        freq = Counter(seqs)
        n = len(seqs)
        c_val = sum(f * (f - 1) for f in freq.values()) / (n * (n - 1)) if n > 1 else 1.0
        _, sigma = chain_stats[c]
        esm2_weights[c] = sigma * c_val
    esm2_weights = esm2_weights / esm2_weights.sum()
    w_scaled = esm2_weights * n_chains  # mean=1, selection-only

    # Combine-then-reduce: per (query, ref) select the chain with the largest
    # weighted z-score, take its unscaled z; then top-K smallest over refs.
    # Chunked over test rows to bound memory.
    n_test = len(test_df)
    combined = np.zeros(n_test)
    chunk = 512
    for s in range(0, n_test, chunk):
        e = min(s + chunk, n_test)
        z_chains, wz_chains = [], []
        for c in range(n_chains):
            te = test_embs_list[c][s:e]                    # L2-normalised
            tr = train_embs_list[c]                        # L2-normalised
            d_cos = 1.0 - te @ tr.T                        # cosine distance
            d_uniq = np.log(K_LOG * (d_cos + B_LOG))       # LogDist form
            d_full = d_uniq[:, train_idx_list[c]]          # (chunk, n_rows)
            mu, sigma = chain_stats[c]
            z = (d_full - mu) / sigma
            z_chains.append(z)
            wz_chains.append(w_scaled[c] * z)
        z_arr = np.stack(z_chains, axis=0)                 # (C, chunk, n_rows)
        wz_arr = np.stack(wz_chains, axis=0)
        winner = np.argmax(wz_arr, axis=0)                 # (chunk, n_rows)
        D = np.take_along_axis(z_arr, winner[None], axis=0)[0]
        Kc = K if (K is not None and K < D.shape[1]) else D.shape[1]
        combined[s:e] = np.partition(D, Kc - 1, axis=1)[:, :Kc].mean(axis=1)

    np.save(cache_path, combined)
    return combined


# REMOVED 2026-05-19 — compute_tcrdist_cv_dist (normalized SW-BLOSUM62 on
# CDR3b only) was NOT TCRdist (paper-reader, Dash et al. 2017): it was the
# same SW-BLOSUM62 similarity as the BLOSUM column, a misrepresentation.
# The genuine TCRdist-CDR3 metric is compute_tcrdist_cdr3_naive.py (bsd4 +
# fixed-gap CDR3 alignment, CDR3a+b), consumed by the canonical panel
# gen_distance_comparison_cv_with_naive.py. The old function and its
# tcrdist_cv_fold*_dist.npy caches are archived in git history; this script
# is retained only for the TCR ESM2 S2DD cache (compute_esm2_cv_dist).


def plot_boxplot(ax, data, labels, title, colors):
    positions = range(len(labels))
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, c=color, s=25, alpha=0.7,
                   edgecolors='white', linewidth=0.3, zorder=3)
    for i, d in enumerate(data):
        if d:
            ax.text(i, max(d) + 0.03, f'{np.mean(d):.2f}', ha='center',
                    fontsize=7, fontweight='bold', color=colors[i])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Per-model |mean r| (AP)', fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ═══════════════════════════════════════════
# TCR CV: 5 models x 5 folds — 4 distance metrics
# ═══════════════════════════════════════════
print("=== TCR CV Distance Comparison ===")
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
DIST_LABELS = ['Lev-log', 'BLOSUM-log', 'ESM2-log']
COLORS_4 = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Pre-compute ESM2 distances (model-independent, one per fold). TCRdist
# removed — genuine TCRdist is compute_tcrdist_cdr3_naive.py.
print("Pre-computing ESM2 distances per fold...")
esm2_dists = {}  # fold -> distance array
for fold in range(5):
    # Load combined test+val for distance alignment
    te = load_tcr_cv_combined('nettcr', fold)
    if te is None:
        continue
    train_path = os.path.join('Data', 'tcr_seq', 'proc_files',
                               f'tcr_cross_val_fold{fold}', 'train_data.csv')
    if not os.path.exists(train_path):
        continue
    train_df = pd.read_csv(train_path)

    # ESM2
    print(f"  Fold {fold}: ESM2...", end='', flush=True)
    esm2_dists[fold] = compute_esm2_cv_dist(te, train_df, fold,
                                              chains=[c for c in CHAINS if c in te.columns])
    print(f" done ({len(esm2_dists[fold])} samples)")

tcr_cv_rs = {label: [] for label in DIST_LABELS}

for model in TCR_MODELS:
    fold_rs = {label: [] for label in DIST_LABELS}
    for fold in range(5):
        te = load_tcr_cv_combined(model, fold)
        if te is None:
            continue
        lc = 'binder' if 'binder' in te.columns else 'y_true'
        pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
        y = te[lc].values.astype(int)
        p = te[pc].values.astype(float)

        # Lev-log (combined_dist.npy = test+val aligned, already log-transformed)
        lev_path = os.path.join(CACHE, f'{model}_cv_fold{fold}_combined_dist.npy')
        if os.path.exists(lev_path):
            d = np.load(lev_path)
            r = compute_binned_r(y, p, d)
            if r is not None:
                fold_rs['Lev-log'].append(r)

        # BLOSUM-log (apply log to cached sqrt distances to spread compressed range)
        blo_path = os.path.join(CACHE, f'{model}_cv_fold{fold}_blosumsqrt_dist.npy')
        if os.path.exists(blo_path):
            d_sqrt = np.load(blo_path)
            d_log = np.log(d_sqrt - d_sqrt.min() + 0.01)
            r = compute_binned_r(y, p, d_log)
            if r is not None:
                fold_rs['BLOSUM-log'].append(r)

        # ESM2: cache already has the per-pair LogDist transform baked in
        # (log(k*(1-cos+b)) inside compute_esm2_cv_dist). Consume RAW — no
        # post-hoc log (a second log would be a double transform, and Lev is
        # consumed raw too). Matches gen_distance_comparison_cv_with_naive.py.
        if fold in esm2_dists:
            d_raw = esm2_dists[fold]
            r = compute_binned_r(y, p, d_raw)
            if r is not None:
                fold_rs['ESM2-log'].append(r)


    for label in DIST_LABELS:
        rs = fold_rs[label]
        if rs:
            val = abs(np.mean(rs))
            tcr_cv_rs[label].append(val)
    lev_val = abs(np.mean(fold_rs['Lev-log'])) if fold_rs['Lev-log'] else 0
    blo_val = abs(np.mean(fold_rs['BLOSUM-log'])) if fold_rs['BLOSUM-log'] else 0
    esm_val = abs(np.mean(fold_rs['ESM2-log'])) if fold_rs['ESM2-log'] else 0
    print(f"  {model}: Lev={lev_val:.3f}, BLOSUM-log={blo_val:.3f}, ESM2-log={esm_val:.3f}")

# Only include metrics with data
tcr_labels, tcr_data, tcr_colors = [], [], []
for label, color in zip(DIST_LABELS, COLORS_4):
    if tcr_cv_rs[label]:
        tcr_labels.append(label)
        tcr_data.append(tcr_cv_rs[label])
        tcr_colors.append(color)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
plot_boxplot(ax, tcr_data, tcr_labels,
             'Distance comparison\n(TCR CV degradation)', tcr_colors)
out_dir = os.path.join(PANEL_DIR, 'lev-logtransf')
os.makedirs(out_dir, exist_ok=True)
# NOTE: the CANONICAL panel m/n generator is
# gen_distance_comparison_cv_with_naive.py. This standalone script is kept
# ONLY for its S2DD cache generation (compute_esm2_cv_dist, in the
# fold loop above). Its plotting section is SUPERSEDED and its BCR ESM2 bar
# still reads the old unaligned esm2_log_s2dd CSV — so it must NOT write the
# canonical panel filenames. Output redirected to *_DEPRECATED to prevent
# clobbering the canonical panels.
out = os.path.join(out_dir, 'fig2_distance_comparison_tcr_cv_DEPRECATED')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}.png")

# ═══════════════════════════════════════════
# BCR CV: 5 models x 5 folds — Lev-log, BLOSUM-sqrt
# ═══════════════════════════════════════════
print("\n=== BCR CV Distance Comparison ===")
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_MODEL_PRED_DIR = {
    'xbcr': 'xbcr/combined_bind_ab_cv',
    'deepaai': 'deepaai/combined_bind_ab_cv',
    'mambaaai': 'mambaaai/combined_bind_ab_cv',
    'mint': 'mint/combined_bind_ab_cv',
    'rleaai': 'rleaai/3pathogen_bind_ab_cv',
}

bcr_cv_rs = {'Lev-log': [], 'BLOSUM-log': [], 'ESM2-log': []}

# Load pre-computed ESM2 BCR CV correlations
esm2_bcr_csv = os.path.join(RESULTS, 'baselines', 'bcr_esm2', 'esm2_correlation_results.csv')
esm2_bcr_df = None
if os.path.exists(esm2_bcr_csv):
    esm2_bcr_df = pd.read_csv(esm2_bcr_csv)
    esm2_bcr_df = esm2_bcr_df[(esm2_bcr_df['baseline'] == 'esm2_log_s2dd') & (esm2_bcr_df['metric'] == 'ap')]

for model in BCR_MODELS:
    lev_rs, blo_rs = [], []
    pred_dir = BCR_MODEL_PRED_DIR[model]
    for fold in range(5):
        pred_path = os.path.join(RESULTS, pred_dir, f'fold{fold}', 'test.csv')
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        lc = 'rbd' if 'rbd' in df.columns else ('y_true' if 'y_true' in df.columns else 'binder')
        pc = 'pred_prob' if 'pred_prob' in df.columns else ('prediction' if 'prediction' in df.columns else 'y_prob')
        if lc not in df.columns or pc not in df.columns:
            continue
        y = df[lc].values.astype(int)
        p = df[pc].values.astype(float)

        lev_path = os.path.join(CACHE, f'bcr_cv_{model}_fold{fold}_uniform_dist.npy')
        if os.path.exists(lev_path):
            d = np.load(lev_path)
            if len(d) == len(y):
                r = compute_binned_r(y, p, d)
                if r is not None:
                    lev_rs.append(r)

        # BLOSUM-log: apply log to cached sqrt distances
        blo_path = os.path.join(CACHE, f'{model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        if os.path.exists(blo_path):
            d_sqrt = np.load(blo_path)
            if len(d_sqrt) == len(y):
                d_log = np.log(d_sqrt - d_sqrt.min() + 0.01)
                r = compute_binned_r(y, p, d_log)
                if r is not None:
                    blo_rs.append(r)

    if lev_rs:
        bcr_cv_rs['Lev-log'].append(abs(np.mean(lev_rs)))
    if blo_rs:
        bcr_cv_rs['BLOSUM-log'].append(abs(np.mean(blo_rs)))

    # ESM2-log from pre-computed CSV (esm2_log_s2dd = log applied per-chain before combining)
    if esm2_bcr_df is not None:
        esm2_rs = esm2_bcr_df[esm2_bcr_df['model'] == model]['pearson_r'].values
        if len(esm2_rs) >= 3:
            bcr_cv_rs['ESM2-log'].append(abs(np.mean(esm2_rs)))

    lev_str = f"Lev={abs(np.mean(lev_rs)):.3f}" if lev_rs else "Lev=N/A"
    blo_str = f"BLOSUM-log={abs(np.mean(blo_rs)):.3f}" if blo_rs else "BLOSUM=N/A"
    esm2_model_rs = esm2_bcr_df[esm2_bcr_df['model'] == model]['pearson_r'].values if esm2_bcr_df is not None else []
    esm2_str = f"ESM2-log={abs(np.mean(esm2_model_rs)):.3f}" if len(esm2_model_rs) > 0 else "ESM2-log=N/A"
    print(f"  {model}: {lev_str}, {blo_str}, {esm2_str}")

bcr_labels = ['Lev-log', 'BLOSUM-log', 'ESM2-log']
bcr_data = [bcr_cv_rs[l] for l in bcr_labels]
bcr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# Remove empty metrics
bcr_labels_f, bcr_data_f, bcr_colors_f = [], [], []
for l, d, c in zip(bcr_labels, bcr_data, bcr_colors):
    if d:
        bcr_labels_f.append(l)
        bcr_data_f.append(d)
        bcr_colors_f.append(c)
bcr_labels, bcr_data, bcr_colors = bcr_labels_f, bcr_data_f, bcr_colors_f

fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
plot_boxplot(ax, bcr_data, bcr_labels,
             'Distance comparison\n(BCR CV degradation)', bcr_colors)
# SUPERSEDED (see note above): BCR ESM2 here uses the old esm2_log_s2dd CSV.
# Canonical BCR panel comes from gen_distance_comparison_cv_with_naive.py.
out = os.path.join(out_dir, 'fig2_distance_comparison_bcr_cv_DEPRECATED')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}.png")

print("\nDone.")
