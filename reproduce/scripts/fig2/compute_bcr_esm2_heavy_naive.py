"""BCR ESM2 naive = TRUE single-chain Heavy mean cosine distance.

Mirrors the TCR ESM2 naive fix (naive_esm2_min_raw mean-over-all): for each
BCR CV fold, per test sample the naive distance is the multiplicity-weighted
MEAN cosine distance to ALL training Heavy chains. Raw embedding distance:
NO log, NO z-norm, NO top-K, NO min/nearest selection. Single chain (Heavy).

Replaces the previously-used `esm2_log_naive` CSV row, which the
research-integrity-auditor found to be a uniform MEAN across ALL 3 chains
(Heavy+Light+variant_seq) with top-K=50 -- multi-chain, mislabeled as
"single-chain".

Outputs (consumed by gen_distance_comparison_cv_with_naive.py BCR section):
  results/fig2_cache/naive_baseline/bcr_cv_fold{0-4}_esm2_naive_raw.npy
Embedding cache (reproducibility, avoids re-running ESM2):
  results/fig2_cache/bcr_heavy_esm2_emb.npz
"""
import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
RES = os.path.join(INPUT_DIR, 'results')
NAIVE_CACHE = os.path.join(RES, 'fig2_cache', 'naive_baseline')
EMB_CACHE = os.path.join(RES, 'fig2_cache', 'bcr_heavy_esm2_emb.npz')
os.makedirs(NAIVE_CACHE, exist_ok=True)

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
FOLD_BASE = 'combined_bind_ab_cv'   # shared split (same as panel BCR Lev naive)


def fold_dir(model, fold):
    return os.path.join(RES, model, FOLD_BASE, f'fold{fold}')


def binned_r(y, p, d, nb=8):
    n = min(len(y), len(p), len(d))
    y, p, d = y[:n], p[:n], d[:n]
    si = np.argsort(d, kind='stable')
    bs = len(si) // nb
    bv, bd = [], []
    for i in range(nb):
        s = i * bs
        e = len(si) if i == nb - 1 else (i + 1) * bs
        ix = si[s:e]
        bv.append(average_precision_score(y[ix], p[ix])
                  if len(np.unique(y[ix])) > 1 else np.nan)
        bd.append(d[ix].mean())
    v, dd = np.array(bv), np.array(bd)
    m = ~np.isnan(v)
    # SIGNED pearson r (NOT abs). Aggregation convention must match the panel
    # (gen_distance_comparison_cv_with_naive.py): per-fold signed r, then
    # abs(mean over folds). mean-of-abs would disagree for sign-unstable
    # signals like the weak BCR ESM2-Heavy distance.
    return pearsonr(dd[m], v[m])[0] if m.sum() >= 4 else np.nan


# ── 1. Gather all unique Heavy sequences across folds (xbcr shared split) ──
folds = []
all_heavy = set()
for fold in range(5):
    fd = fold_dir('xbcr', fold)
    tp, rp = os.path.join(fd, 'test.csv'), os.path.join(fd, 'train.csv')
    if not (os.path.exists(tp) and os.path.exists(rp)):
        print(f"fold{fold}: missing data, skip")
        continue
    te = pd.read_csv(tp)
    tr = pd.read_csv(rp)
    th = te['Heavy'].astype(str).tolist()
    rh = tr['Heavy'].astype(str).tolist()
    folds.append((fold, th, rh))
    all_heavy.update(th)
    all_heavy.update(rh)
all_heavy = sorted(all_heavy)
print(f"unique Heavy across all folds: {len(all_heavy)}")

# ── 2. ESM2 embeddings (cached) ──
if os.path.exists(EMB_CACHE):
    z = np.load(EMB_CACHE, allow_pickle=True)
    EMB = dict(zip(z['seqs'].tolist(), z['embs']))
    missing = [s for s in all_heavy if s not in EMB]
    print(f"emb cache hit: {len(EMB)} cached, {len(missing)} missing")
else:
    EMB, missing = {}, all_heavy
if missing:
    from eval_baselines_esm2 import load_esm2_model, compute_embeddings
    print(f"computing ESM2 embeddings for {len(missing)} Heavy chains...")
    model, bc = load_esm2_model(device='cuda')
    new = compute_embeddings(missing, model, bc, device='cuda')
    EMB.update(new)
    seqs = list(EMB.keys())
    embs = np.array([EMB[s] for s in seqs], dtype=np.float32)
    np.savez_compressed(EMB_CACHE, seqs=np.array(seqs, dtype=object), embs=embs)
    print(f"saved {len(seqs)} embeddings -> {EMB_CACHE}")

DIM = len(next(iter(EMB.values())))
ZERO = np.zeros(DIM, dtype=np.float32)


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# ── 3. Per-fold naive = multiplicity-weighted mean cosine over ALL train ──
naive_by_fold = {}
for fold, th, rh in folds:
    cnt = Counter(rh)
    uq = list(cnt.keys())
    Er = l2(np.array([EMB.get(s, ZERO) for s in uq], dtype=np.float64))
    w = np.array([cnt[s] for s in uq], dtype=np.float64)
    w = w / w.sum()
    Et = l2(np.array([EMB.get(s, ZERO) for s in th], dtype=np.float64))
    # mean cosine distance over all training rows (multiplicity via w)
    d = (1.0 - Et @ Er.T) @ w
    out = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_esm2_naive_raw.npy')
    np.save(out, d)
    naive_by_fold[fold] = d
    print(f"fold{fold}: n_test={len(d)} mean_d={d.mean():.4f} -> {os.path.basename(out)}")

# ── 4. Per-model |mean r| (AP) for the record ──
print(f"\n{'model':>10} | naive ESM2-Heavy |mean r| (AP, 5 folds)")
allr = []
for m in BCR_MODELS:
    rs = []
    for fold, th, rh in folds:
        pp = os.path.join(fold_dir(m, fold), 'test.csv')
        if not os.path.exists(pp):
            continue
        df = pd.read_csv(pp)
        if 'rbd' not in df or 'pred_prob' not in df:
            continue
        y = df['rbd'].values.astype(int)
        p = df['pred_prob'].values.astype(float)
        rs.append(binned_r(y, p, naive_by_fold[fold]))
    # Panel convention: abs(mean of signed r across folds)
    mr = abs(np.nanmean(rs)) if rs else float('nan')
    allr.append(mr)
    print(f"{m:>10} | {mr:.3f}")
print(f"{'MEAN':>10} | {np.nanmean(allr):.3f}")
