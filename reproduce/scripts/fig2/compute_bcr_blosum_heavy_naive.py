"""BCR BLOSUM naive = single-chain Heavy MEAN Smith-Waterman distance over
a fixed random training subsample (multiplicity-weighted).

UNIFORM naive definition across ALL metrics/domains: mean raw distance over
all training, single chain, no log/z-norm/top-K/min. SW over all unique
training is hours, so the mean is estimated on a fixed N_SUB=800 random
training subsample (unbiased estimator of the full-training mean; NOT a
selection). Test dedup'd + mapped back; 5 folds parallel. Caches to the
exact filename the panel already loads:
  results/fig2_cache/naive_baseline/bcr_cv_fold{0-4}_blosum_naive_raw.npy
"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
RES = os.path.join(INPUT_DIR, 'results')
NAIVE_CACHE = os.path.join(RES, 'fig2_cache', 'naive_baseline')
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
FOLD_BASE = 'combined_bind_ab_cv'


N_SUB, SEED = 800, 42   # must match naive_blosum_min_raw in the panel script


def _fold_worker(fold):
    """Per-test-sample MEAN (1 - SW_sim) over a fixed random training
    subsample (multiplicity-weighted). Matches naive_blosum_min_raw:
    mean-over-all-training estimator, NO min/top-K. Test dedup'd + mapped
    back (mean is identical for identical test sequences)."""
    from collections import Counter
    from calipper.pluggable_distance import make_sw_blosum62_similarity
    fd = os.path.join(RES, 'xbcr', FOLD_BASE, f'fold{fold}')
    tp, rp = os.path.join(fd, 'test.csv'), os.path.join(fd, 'train.csv')
    if not (os.path.exists(tp) and os.path.exists(rp)):
        return fold, None
    th = pd.read_csv(tp)['Heavy'].astype(str).tolist()
    rh = pd.read_csv(rp)['Heavy'].astype(str).tolist()
    cnt = Counter(rh)
    uq = list(cnt.keys())
    w = np.array([cnt[s] for s in uq], dtype=float)
    if len(uq) > N_SUB:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(uq), N_SUB, replace=False)
        uq = [uq[j] for j in idx]
        w = w[idx]
    w = w / w.sum()
    uniq_test = list(dict.fromkeys(th))
    sw_sim = make_sw_blosum62_similarity()
    cache = {}
    for i, q in enumerate(uniq_test):
        dd = np.fromiter((max(1.0 - sw_sim(q, r), 0.0) for r in uq),
                         dtype=float, count=len(uq))
        cache[q] = float(dd @ w)
        if (i + 1) % 200 == 0:
            print(f"  fold{fold}: {i+1}/{len(uniq_test)} unique test", flush=True)
    d = np.array([cache[s] for s in th])
    out = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{fold}_blosum_naive_raw.npy')
    np.save(out, d)
    print(f"fold{fold}: n_test={len(d)} uniq_test={len(uniq_test)} "
          f"train_sub={len(uq)} -> {os.path.basename(out)}", flush=True)
    return fold, out


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
    return pearsonr(dd[m], v[m])[0] if m.sum() >= 4 else np.nan  # signed


if __name__ == '__main__':
    with Pool(5) as pool:
        results = pool.map(_fold_worker, list(range(5)))
    dist = {}
    for fold, out in results:
        if out:
            dist[fold] = np.load(out)

    print(f"\n{'model':>10} | BCR BLOSUM-naive (Heavy) |mean r| (AP, 5 folds)")
    allr = []
    for m in BCR_MODELS:
        rs = []
        for fold, d in dist.items():
            pp = os.path.join(RES, m, FOLD_BASE, f'fold{fold}', 'test.csv')
            if not os.path.exists(pp):
                continue
            df = pd.read_csv(pp)
            if 'rbd' not in df or 'pred_prob' not in df:
                continue
            rs.append(binned_r(df['rbd'].values.astype(int),
                               df['pred_prob'].values.astype(float), d))
        mr = abs(np.nanmean(rs)) if rs else float('nan')
        allr.append(mr)
        print(f"{m:>10} | {mr:.3f}")
    print(f"{'MEAN':>10} | {np.nanmean(allr):.3f}")
    print("ref: BCR Lev naive ~0.87, BLOSUM S2DD ~0.74")
