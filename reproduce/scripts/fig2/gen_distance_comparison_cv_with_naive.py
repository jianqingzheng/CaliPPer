#!/usr/bin/env python3
"""Fig 2 panels m-n: Distance metric comparison — S2DD vs naive single-chain baseline.

S2DD: multi-chain (peptide+CDR3a+CDR3b for TCR, Heavy+Light+variant_seq for BCR),
      sigma_C weighted, topK=50, z-norm combined, log transform.

Naive: single-chain (CDR3b for TCR, Heavy for BCR),
       min distance (K=1, nearest neighbor), log transform.
       No sigma_C, no z-norm, no multi-chain combine.

Each distance metric shows paired bars: left=S2DD (full color), right=naive (lighter).
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import Levenshtein

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style
from calipper.general_evaluator import safe_metric

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig2_cache')
NAIVE_CACHE = os.path.join(CACHE, 'naive_baseline')
os.makedirs(NAIVE_CACHE, exist_ok=True)

N_BINS = 8
K_S2DD = 50
k_LOG, b_LOG = 0.1, 0.1


# ═══════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════

def compute_binned_r(y, p, d, n_bins=8):
    n = min(len(d), len(y))
    y, p, d = y[:n], p[:n], d[:n]
    si = np.argsort(d)
    bs = len(si) // n_bins
    bv, bd = [], []
    for i in range(n_bins):
        s, e = i * bs, (len(si) if i == n_bins - 1 else (i + 1) * bs)
        idx = si[s:e]
        bv.append(safe_metric('ap', y[idx], p[idx]))
        bd.append(d[idx].mean())
    v, dd = np.array(bv), np.array(bd)
    valid = ~np.isnan(v)
    if valid.sum() >= 4:
        r, _ = pearsonr(dd[valid], v[valid])
        return r
    return None


def load_tcr_cv_combined(model, fold):
    """Load TCR CV test+val combined predictions."""
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


# ═══════════════════════════════════════════
# Naive distance functions — single chain, min (K=1), log transform
# ═══════════════════════════════════════════

# Naive single-chain baseline = the SIMPLE raw distance between the single
# chain and the training set: the MEAN raw per-pair distance over ALL
# training samples (multiplicity-weighted). No log, no z-norm, no top-K, no
# min/nearest selection — uniform definition across ALL metrics (Lev, BLOSUM,
# ESM2) and both domains. (Top-1 min is itself a K=1 selection and is
# degenerate under stratified CV; the mean over all training is the
# definitionally-correct single-chain comparator.)

def naive_lev_min_raw(test_seqs, train_seqs):
    """Mean Levenshtein distance (1-ratio) over ALL training. Single chain."""
    from collections import Counter
    cnt = Counter(train_seqs)
    uq = list(cnt.keys())
    w = np.array([cnt[s] for s in uq], dtype=float)
    w = w / w.sum()
    dists = np.zeros(len(test_seqs))
    for i, q in enumerate(test_seqs):
        dd = np.fromiter((1.0 - Levenshtein.ratio(q, r) for r in uq),
                         dtype=float, count=len(uq))
        dists[i] = float(dd @ w)
    return dists


def naive_blosum_min_raw(test_seqs, train_seqs, n_sub=800, seed=42):
    """Mean BLOSUM-SW distance (1-sim) over training. Single chain.

    Full pairwise SW over all unique training is hours of Smith-Waterman, so
    the mean is estimated on a FIXED random training subsample (multiplicity-
    weighted). This is an unbiased estimator of the mean-over-all-training raw
    distance — every sampled training sequence contributes; NO min/sort/top-K.
    """
    from collections import Counter
    from calipper.pluggable_distance import make_sw_blosum62_similarity
    sw_sim = make_sw_blosum62_similarity()
    cnt = Counter(train_seqs)
    uq = list(cnt.keys())
    w = np.array([cnt[s] for s in uq], dtype=float)
    if len(uq) > n_sub:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(uq), n_sub, replace=False)
        uq = [uq[j] for j in idx]
        w = w[idx]
    w = w / w.sum()
    dists = np.zeros(len(test_seqs))
    for i, q in enumerate(test_seqs):
        dd = np.fromiter((max(1.0 - sw_sim(q, r), 0.0) for r in uq),
                         dtype=float, count=len(uq))
        dists[i] = float(dd @ w)
        if (i + 1) % 2000 == 0:
            print(f"    [{i+1}/{len(test_seqs)}]", flush=True)
    return dists


# ESM2 naive ONLY: the simple raw cosine distance between the single chain
# (CDR3b) of each test sample and the training set, MEAN over ALL training
# samples. No log, no z-norm, no top-K, no min selection. (Top-1 min is
# itself a K=1 selection and is degenerate for ESM2 here: epitope-stratified
# CV makes ~62% of test CDR3b embeddings identical to a training embedding,
# so min->0 and it collapses to a binary train-membership indicator. Mean
# over all training is continuous and is a fair single-chain comparator.)
# Lev/BLOSUM naive intentionally keep the original top-1 min (the auditor
# confirmed their picture is not misrepresented and they are unchanged here).

def naive_esm2_min_raw(test_seqs, train_seqs, embeddings):
    """Mean ESM2 cosine distance (1 - cosine sim) over ALL training.

    Single chain, raw: no log, no z-norm, no top-K, no min selection. Same
    base metric (cosine) as the aligned ESM2 S2DD, so S2DD-vs-naive is
    apples-to-apples in everything except the multi-chain S2DD aggregation.
    """
    from collections import Counter
    dim = 1280
    zero = np.zeros(dim)
    cnt = Counter(train_seqs)
    uq = list(cnt.keys())
    train_embs = np.array([embeddings.get(s, zero) for s in uq])
    tn = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_embs = train_embs / np.maximum(tn, 1e-12)
    w = np.array([cnt[s] for s in uq], dtype=float)
    w = w / w.sum()

    dists = np.zeros(len(test_seqs))
    for i, q in enumerate(test_seqs):
        q_emb = embeddings.get(q, zero)
        qn = np.linalg.norm(q_emb)
        q_emb = q_emb / max(qn, 1e-12)
        cos_d = 1.0 - q_emb @ train_embs.T
        dists[i] = float(cos_d @ w)
    return dists


# ═══════════════════════════════════════════
# Column name helpers
# ═══════════════════════════════════════════

COL_MAP = {'peptide': 'epitope', 'CDR3a': 'cdr3_a', 'CDR3b': 'cdr3_b'}


def get_col(df, target):
    if target in df.columns:
        return target
    alt = COL_MAP.get(target)
    if alt and alt in df.columns:
        return alt
    return None


# ═══════════════════════════════════════════
# TCR CV computation
# ═══════════════════════════════════════════

print("=== TCR CV: S2DD vs Naive ===")
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_LABELS = ['Lev', 'BLOSUM', 'ESM2', 'TCRdist']

# Load ESM2 embeddings (for both S2DD and naive ESM2)
ESM2_CACHE_FILE = os.path.join(RESULTS, 'multi_distance_comparison', 'cache',
                                'esm2_all_embeddings.npz')
esm2_embeddings = None
if os.path.exists(ESM2_CACHE_FILE):
    npz = np.load(ESM2_CACHE_FILE, allow_pickle=True)
    esm2_embeddings = dict(zip(npz['seqs'].tolist(), npz['embs']))

# Pre-compute naive distances per fold (model-independent for Lev, BLOSUM, ESM2, TCRdist)
print("Pre-computing naive single-chain distances per fold...")
naive_tcr_dists = {}  # (metric, fold) -> distance array

for fold in range(5):
    te = load_tcr_cv_combined('nettcr', fold)  # any model, same test set
    if te is None:
        continue
    cdr3b_col = get_col(te, 'CDR3b')
    test_seqs = te[cdr3b_col].astype(str).tolist()

    train_path = os.path.join('Data', 'tcr_seq', 'proc_files',
                               f'tcr_cross_val_fold{fold}', 'train_data.csv')
    train_df = pd.read_csv(train_path)
    train_col = get_col(train_df, 'CDR3b')
    train_seqs = train_df[train_col].astype(str).tolist()

    # Lev naive
    cache_f = os.path.join(NAIVE_CACHE, f'tcr_cv_fold{fold}_lev_naive_raw.npy')
    if os.path.exists(cache_f):
        naive_tcr_dists[('Lev', fold)] = np.load(cache_f)
        print(f"  Fold {fold} Lev: cached")
    else:
        print(f"  Fold {fold} Lev: computing...", end='', flush=True)
        d = naive_lev_min_raw(test_seqs, train_seqs)
        np.save(cache_f, d)
        naive_tcr_dists[('Lev', fold)] = d
        print(f" done")

    # BLOSUM naive
    cache_f = os.path.join(NAIVE_CACHE, f'tcr_cv_fold{fold}_blosum_naive_raw.npy')
    if os.path.exists(cache_f):
        naive_tcr_dists[('BLOSUM', fold)] = np.load(cache_f)
        print(f"  Fold {fold} BLOSUM: cached")
    else:
        print(f"  Fold {fold} BLOSUM: computing...", flush=True)
        d = naive_blosum_min_raw(test_seqs, train_seqs)
        np.save(cache_f, d)
        naive_tcr_dists[('BLOSUM', fold)] = d
        print(f"  Fold {fold} BLOSUM: done")

    # ESM2 naive
    if esm2_embeddings is not None:
        cache_f = os.path.join(NAIVE_CACHE, f'tcr_cv_fold{fold}_esm2_naive_raw.npy')
        if os.path.exists(cache_f):
            naive_tcr_dists[('ESM2', fold)] = np.load(cache_f)
            print(f"  Fold {fold} ESM2: cached")
        else:
            print(f"  Fold {fold} ESM2: computing...", end='', flush=True)
            d = naive_esm2_min_raw(test_seqs, train_seqs, esm2_embeddings)
            np.save(cache_f, d)
            naive_tcr_dists[('ESM2', fold)] = d
            print(f" done")

    # TCRdist = the genuine Dash et al. 2017 / tcrdist3 CDR3 distance
    # (CDR3a+CDR3b, bsd4 + fixed-gap, no normalisation), shown as a single
    # box = its mean-over-training naive (compute_tcrdist_cdr3_naive.py).
    # It is a self-contained receptor-loop metric (no multi-chain S2DD vs
    # single-chain contrast); CDR1/CDR2/CDR2.5 omitted (CDR2.5 unannotated).

# Compute S2DD and naive |mean r| per model
tcr_s2dd_rs = {label: [] for label in TCR_DIST_LABELS}
tcr_naive_rs = {label: [] for label in TCR_DIST_LABELS}

for model in TCR_MODELS:
    s2dd_fold = {l: [] for l in TCR_DIST_LABELS}
    naive_fold = {l: [] for l in TCR_DIST_LABELS}

    for fold in range(5):
        te = load_tcr_cv_combined(model, fold)
        if te is None:
            continue
        lc = 'binder' if 'binder' in te.columns else 'y_true'
        pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
        y = te[lc].values.astype(int)
        p = te[pc].values.astype(float)

        # S2DD: Lev
        path = os.path.join(CACHE, f'{model}_cv_fold{fold}_combined_dist.npy')
        if os.path.exists(path):
            d = np.load(path)
            r = compute_binned_r(y, p, d)
            if r is not None:
                s2dd_fold['Lev'].append(r)

        # S2DD: BLOSUM (log of cached sqrt)
        path = os.path.join(CACHE, f'{model}_cv_fold{fold}_blosumsqrt_dist.npy')
        if os.path.exists(path):
            d_sqrt = np.load(path)
            d_log = np.log(d_sqrt - d_sqrt.min() + 0.01)
            r = compute_binned_r(y, p, d_log)
            if r is not None:
                s2dd_fold['BLOSUM'].append(r)

        # S2DD: ESM2 — aligned with canonical Lev S2DD (weighted_max_znorm +
        # sigma_C + top-K, full-pair z-norm). Raw embedding distance: NO log
        # (per-pair or post-hoc); the cache stores the final combined distance.
        esm2_path = os.path.join(CACHE, f'esm2_cv_fold{fold}_dist.npy')
        if os.path.exists(esm2_path):
            d_raw = np.load(esm2_path)
            r = compute_binned_r(y, p, d_raw)
            if r is not None:
                s2dd_fold['ESM2'].append(r)

        # TCRdist single box = GENUINE TCRdist CDR3 distance (Dash et al. 2017
        # / tcrdist3 CDR3-only), CDR3a+CDR3b, naive = mean over training
        # subsample. NOT the old SW-BLOSUM62 proxy (paper-reader confirmed
        # that was not TCRdist). Computed by compute_tcrdist_cdr3_naive.py.
        tcr_path = os.path.join(NAIVE_CACHE,
                                f'tcrdist_cdr3_naive_fold{fold}.npy')
        if os.path.exists(tcr_path):
            d = np.load(tcr_path)
            r = compute_binned_r(y, p, d)
            if r is not None:
                s2dd_fold['TCRdist'].append(r)

        # Naive: all metrics (model-independent)
        for label in TCR_DIST_LABELS:
            key = (label, fold)
            if key in naive_tcr_dists:
                d = naive_tcr_dists[key]
                r = compute_binned_r(y, p, d)
                if r is not None:
                    naive_fold[label].append(r)

    for label in TCR_DIST_LABELS:
        if s2dd_fold[label]:
            tcr_s2dd_rs[label].append(abs(np.mean(s2dd_fold[label])))
        if naive_fold[label]:
            tcr_naive_rs[label].append(abs(np.mean(naive_fold[label])))

    print(f"  {model}:")
    for label in TCR_DIST_LABELS:
        s = abs(np.mean(s2dd_fold[label])) if s2dd_fold[label] else float('nan')
        n = abs(np.mean(naive_fold[label])) if naive_fold[label] else float('nan')
        print(f"    {label}: S2DD={s:.3f}, naive={n:.3f}")

# ═══════════════════════════════════════════
# BCR CV computation
# ═══════════════════════════════════════════

print("\n=== BCR CV: S2DD vs Naive ===")
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_PRED_DIR = {
    'xbcr': 'xbcr/combined_bind_ab_cv',
    'deepaai': 'deepaai/combined_bind_ab_cv',
    'mambaaai': 'mambaaai/combined_bind_ab_cv',
    'mint': 'mint/combined_bind_ab_cv',
    'rleaai': 'rleaai/combined_bind_ab_cv',
}
BCR_DIST_LABELS = ['Lev', 'BLOSUM', 'ESM2']

# Pre-compute BCR naive distances per fold (model-independent)
print("Pre-computing naive single-chain (Heavy) distances per fold...")
naive_bcr_dists = {}
s2dd_bcr_dists = {}   # ('ESM2', fold) -> aligned canonical S2DD distance

# Load BCR ESM2 pre-computed correlations for S2DD
esm2_bcr_csv = os.path.join(RESULTS, 'baselines', 'bcr_esm2',
                              'esm2_correlation_results.csv')
esm2_bcr_df = None
if os.path.exists(esm2_bcr_csv):
    esm2_bcr_df = pd.read_csv(esm2_bcr_csv)

for fold in range(5):
    train_path = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv',
                               f'fold{fold}', 'train.csv')
    test_path = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv',
                              f'fold{fold}', 'test.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        continue
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_heavy = test_df['Heavy'].astype(str).tolist()
    train_heavy = train_df['Heavy'].astype(str).tolist()

    # Lev naive on Heavy
    cache_f = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_lev_naive_raw.npy')
    if os.path.exists(cache_f):
        naive_bcr_dists[('Lev', fold)] = np.load(cache_f)
        print(f"  Fold {fold} Lev: cached")
    else:
        print(f"  Fold {fold} Lev: computing...", end='', flush=True)
        d = naive_lev_min_raw(test_heavy, train_heavy)
        np.save(cache_f, d)
        naive_bcr_dists[('Lev', fold)] = d
        print(f" done")

    # BLOSUM naive on Heavy — SKIPPED (too slow: ~120 AA × ~4000 unique train)
    # SW alignment on 120 AA chains takes ~2h per fold × 5 folds = ~10h total
    cache_f = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_blosum_naive_raw.npy')
    if os.path.exists(cache_f):
        naive_bcr_dists[('BLOSUM', fold)] = np.load(cache_f)
        print(f"  Fold {fold} BLOSUM: cached")
    else:
        print(f"  Fold {fold} BLOSUM: skipped (too slow for ~120 AA chains)")

    # ESM2 naive on Heavy — true single-chain Heavy mean cosine distance
    # over ALL training (multiplicity-weighted, raw: no log/z-norm/top-K/min),
    # matching the TCR ESM2 naive definition. Pre-computed by
    # compute_bcr_esm2_heavy_naive.py (ESM2 embeddings cached). Replaces the
    # old `esm2_log_naive` CSV row, which was a multi-chain uniform mean
    # (Heavy+Light+variant_seq, top-K=50) mislabeled as single-chain.
    cache_f = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_esm2_naive_raw.npy')
    if os.path.exists(cache_f):
        naive_bcr_dists[('ESM2', fold)] = np.load(cache_f)
        print(f"  Fold {fold} ESM2 naive: cached")
    else:
        print(f"  Fold {fold} ESM2 naive: MISSING "
              f"(run compute_bcr_esm2_heavy_naive.py)")

    # ESM2 S2DD — aligned canonical cosine-log pipeline (weighted_max_znorm +
    # sigma_C + top-K=30 + full-pair z-norm; chains Heavy+Light+variant_seq;
    # k=0.1, b=0.03), model-independent. Pre-computed by
    # compute_bcr_esm2_s2dd.py. Replaces the old `esm2_log_s2dd` CSV row
    # (unaligned multi-chain Euclidean pipeline).
    s2dd_f = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_esm2_s2dd_raw.npy')
    if os.path.exists(s2dd_f):
        s2dd_bcr_dists[('ESM2', fold)] = np.load(s2dd_f)
        print(f"  Fold {fold} ESM2 S2DD: cached")
    else:
        print(f"  Fold {fold} ESM2 S2DD: MISSING "
              f"(run compute_bcr_esm2_s2dd.py)")

# BCR S2DD and naive correlations
bcr_s2dd_rs = {label: [] for label in BCR_DIST_LABELS}
bcr_naive_rs = {label: [] for label in BCR_DIST_LABELS}

for model in BCR_MODELS:
    s2dd_fold = {l: [] for l in BCR_DIST_LABELS}
    naive_fold = {l: [] for l in BCR_DIST_LABELS}

    for fold in range(5):
        pred_path = os.path.join(RESULTS, BCR_PRED_DIR[model], f'fold{fold}', 'test.csv')
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        y = df['rbd'].values.astype(int)
        p = df['pred_prob'].values.astype(float)

        # S2DD: Lev
        lev_path = os.path.join(CACHE, f'bcr_cv_{model}_fold{fold}_uniform_dist.npy')
        if os.path.exists(lev_path):
            d = np.load(lev_path)
            if len(d) == len(y):
                r = compute_binned_r(y, p, d)
                if r is not None:
                    s2dd_fold['Lev'].append(r)

        # S2DD: BLOSUM
        blo_path = os.path.join(CACHE, f'{model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        if os.path.exists(blo_path):
            d_sqrt = np.load(blo_path)
            if len(d_sqrt) == len(y):
                d_log = np.log(d_sqrt - d_sqrt.min() + 0.01)
                r = compute_binned_r(y, p, d_log)
                if r is not None:
                    s2dd_fold['BLOSUM'].append(r)

        # S2DD: ESM2 (aligned canonical cosine-log, model-independent)
        key = ('ESM2', fold)
        if key in s2dd_bcr_dists:
            d = s2dd_bcr_dists[key]
            if len(d) == len(y):
                r = compute_binned_r(y, p, d)
                if r is not None:
                    s2dd_fold['ESM2'].append(r)

        # Naive: Lev, BLOSUM, ESM2 (model-independent, single-chain Heavy)
        for label in ['Lev', 'BLOSUM', 'ESM2']:
            key = (label, fold)
            if key in naive_bcr_dists:
                d = naive_bcr_dists[key]
                if len(d) == len(y):
                    r = compute_binned_r(y, p, d)
                    if r is not None:
                        naive_fold[label].append(r)

    # ESM2 S2DD now from the aligned canonical cosine-log pipeline
    # (compute_bcr_esm2_s2dd.py -> s2dd_fold['ESM2'] above), NOT the old
    # esm2_log_s2dd CSV. ESM2 naive = true single-chain Heavy mean cosine
    # (naive_fold['ESM2'] above). esm2_bcr_df no longer used for the panel.

    for label in ['Lev', 'BLOSUM', 'ESM2']:
        if s2dd_fold[label]:
            bcr_s2dd_rs[label].append(abs(np.mean(s2dd_fold[label])))
    for label in ['Lev', 'BLOSUM', 'ESM2']:
        if naive_fold[label]:
            bcr_naive_rs[label].append(abs(np.mean(naive_fold[label])))

    print(f"  {model}:")
    for label in BCR_DIST_LABELS:
        s_list = s2dd_fold.get(label, [])
        n_list = naive_fold.get(label, [])
        s = abs(np.mean(s_list)) if s_list else float('nan')
        n = abs(np.mean(n_list)) if n_list else float('nan')
        print(f"    {label}: S2DD={s:.3f}, naive={n:.3f}")


# ═══════════════════════════════════════════
# Plotting: paired boxplot (S2DD vs naive side by side)
# ═══════════════════════════════════════════

def plot_paired_boxplot(ax, s2dd_data, naive_data, labels, title, colors,
                         single_labels=None):
    """Paired boxplot: for each metric, left=S2DD (full), right=naive (lighter).

    Labels in ``single_labels`` are drawn as ONE centred box using the
    plain (s2dd_data) value, with no S2DD-vs-naive pairing — used for
    inherently single-chain metrics such as TCRdist (CDR3b only).
    """
    single_labels = set(single_labels or [])
    n_metrics = len(labels)
    width = 0.26          # narrower boxes
    gap = 0.12            # intra-pair (S2DD vs naive) gap
    GROUP_SEP = 0.85      # larger separation between metric groups
    group_width = 2 * width + gap

    for i, (label, color) in enumerate(zip(labels, colors)):
        center = i * (group_width + GROUP_SEP)

        if label in single_labels:
            # Single plain box (no S2DD/naive pairing)
            d = s2dd_data[i] if s2dd_data[i] else [0]
            bp = ax.boxplot([d], positions=[center], widths=width,
                            patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(1.5)
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(d))
            ax.scatter(np.full(len(d), center) + jitter, d, c=color, s=18,
                       alpha=0.8, edgecolors='white', linewidth=0.3, zorder=3)
            if s2dd_data[i]:
                ax.text(center, max(d) + 0.03, f'{np.mean(d):.2f}',
                        ha='center', fontsize=8, fontweight='bold', color=color)
            continue

        # S2DD (left, full color)
        s_data = s2dd_data[i] if s2dd_data[i] else [0]
        bp1 = ax.boxplot([s_data], positions=[center - width/2 - gap/2],
                          widths=width, patch_artist=True, showfliers=False)
        bp1['boxes'][0].set_facecolor(color)
        bp1['boxes'][0].set_alpha(0.7)
        bp1['medians'][0].set_color('black')
        bp1['medians'][0].set_linewidth(1.5)
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(s_data))
        ax.scatter(np.full(len(s_data), center - width/2 - gap/2) + jitter,
                   s_data, c=color, s=18, alpha=0.8, edgecolors='white',
                   linewidth=0.3, zorder=3)

        # Naive (right, lighter color)
        n_data = naive_data[i] if naive_data[i] else [0]
        # Make lighter version of color
        from matplotlib.colors import to_rgba
        rgba = to_rgba(color)
        light_color = (rgba[0]*0.5 + 0.5, rgba[1]*0.5 + 0.5, rgba[2]*0.5 + 0.5, 0.7)
        bp2 = ax.boxplot([n_data], positions=[center + width/2 + gap/2],
                          widths=width, patch_artist=True, showfliers=False)
        bp2['boxes'][0].set_facecolor(light_color)
        bp2['boxes'][0].set_alpha(0.7)
        bp2['medians'][0].set_color('black')
        bp2['medians'][0].set_linewidth(1.5)
        jitter = np.random.default_rng(43).uniform(-0.06, 0.06, len(n_data))
        ax.scatter(np.full(len(n_data), center + width/2 + gap/2) + jitter,
                   n_data, c=light_color, s=18, alpha=0.8, edgecolors='white',
                   linewidth=0.3, zorder=3)

        # Mean annotations
        if s2dd_data[i]:
            ax.text(center - width/2 - gap/2, max(s_data) + 0.03,
                    f'{np.mean(s_data):.2f}', ha='center', fontsize=8,
                    fontweight='bold', color=color)
        if naive_data[i]:
            ax.text(center + width/2 + gap/2, max(n_data) + 0.03,
                    f'{np.mean(n_data):.2f}', ha='center', fontsize=8,
                    fontweight='bold', color=light_color[:3])

    # X-axis (display labels: ESM2 -> ESM-2; single-box metrics tagged "(ref)")
    centers = [i * (group_width + GROUP_SEP) for i in range(n_metrics)]
    disp = []
    for lab in labels:
        d = 'ESM-2' if lab == 'ESM2' else lab
        if lab in single_labels:
            d = d + '\n(ref)'
        disp.append(d)
    ax.set_xticks(centers)
    ax.set_xticklabels(disp, fontsize=9)
    ax.set_ylabel('Per-model |mean r| (AP)', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=colors[0], alpha=0.7),
               Patch(facecolor=(to_rgba(colors[0])[0]*0.5+0.5,
                                to_rgba(colors[0])[1]*0.5+0.5,
                                to_rgba(colors[0])[2]*0.5+0.5, 0.7))],
              ['S2DD', 'Naive'],
              fontsize=6, loc='upper right', framealpha=0.8)


# ═══════════════════════════════════════════
# Generate plots
# ═══════════════════════════════════════════

TCR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
BCR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

out_dir = os.path.join(PANEL_DIR, 'lev-logtransf')
os.makedirs(out_dir, exist_ok=True)

# TCR CV
tcr_s2dd_data = [tcr_s2dd_rs.get(l, []) for l in TCR_DIST_LABELS]
tcr_naive_data = [tcr_naive_rs.get(l, []) for l in TCR_DIST_LABELS]

# Only include metrics with data
tcr_labels_f, tcr_s2dd_f, tcr_naive_f, tcr_colors_f = [], [], [], []
for l, s, n, c in zip(TCR_DIST_LABELS, tcr_s2dd_data, tcr_naive_data, TCR_COLORS):
    if s or n:
        tcr_labels_f.append(l)
        tcr_s2dd_f.append(s)
        tcr_naive_f.append(n)
        tcr_colors_f.append(c)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
plot_paired_boxplot(ax, tcr_s2dd_f, tcr_naive_f, tcr_labels_f,
                     'Distance comparison (TCR CV)', tcr_colors_f,
                     single_labels={'TCRdist'})
out = os.path.join(out_dir, 'fig2_distance_comparison_tcr_cv')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}.png")

# BCR CV
bcr_s2dd_data = [bcr_s2dd_rs.get(l, []) for l in BCR_DIST_LABELS]
bcr_naive_data = [bcr_naive_rs.get(l, []) for l in BCR_DIST_LABELS]

bcr_labels_f, bcr_s2dd_f, bcr_naive_f, bcr_colors_f = [], [], [], []
for l, s, n, c in zip(BCR_DIST_LABELS, bcr_s2dd_data, bcr_naive_data, BCR_COLORS):
    if s or n:
        bcr_labels_f.append(l)
        bcr_s2dd_f.append(s)
        bcr_naive_f.append(n)
        bcr_colors_f.append(c)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
plot_paired_boxplot(ax, bcr_s2dd_f, bcr_naive_f, bcr_labels_f,
                     'Distance comparison (BCR CV)', bcr_colors_f)
out = os.path.join(out_dir, 'fig2_distance_comparison_bcr_cv')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}.png")

print("\nDone.")
