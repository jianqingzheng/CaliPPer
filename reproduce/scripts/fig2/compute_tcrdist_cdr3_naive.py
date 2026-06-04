"""Genuine TCRdist CDR3 distance (Dash et al. 2017 / tcrdist3 CDR3-only mode),
on CDR3alpha + CDR3beta, as the panel-m TCRdist naive.

This REPLACES the previous SW-BLOSUM62 proxy (paper-reader 2026-05-18 confirmed
that normalized Smith-Waterman BLOSUM62 on CDR3b is NOT TCRdist). Faithful
TCRdist CDR3 component:

  bsd4(a,b) = 0                       if a == b
            = max(0, 4 - BLOSUM62(a,b)) otherwise
  CDR3 length mismatch -> fixed middle gap at
      gappos = min(6, 3 + (Lshort - 5)//2)
      (Llong - Lshort) gap columns inserted into the shorter CDR3 at gappos
  per gap-aligned position: penalty 12 (the tcrdist CDR3 gap penalty)
  NO normalization (raw integer-valued distance)
  TCRdist(q,r) = tcrdist_cdr3(CDR3a_q,CDR3a_r) + tcrdist_cdr3(CDR3b_q,CDR3b_r)

CDR1/CDR2/CDR2.5 (germline/V-loop) terms are OMITTED — CDR2.5 is not annotated
in the data; this is the recognized tcrdist3 CDR3-only configuration, on both
CDR3 chains to align with the S2DD receptor chains (CDR3a, CDR3b).

Naive = mean TCRdist over a fixed random training subsample (n_sub=2000,
seed=42, multiplicity-weighted) — unbiased estimator of the mean over all
training (full pure-Python pairwise is ~5e9 ops; subsample for tractability,
same rationale as the BLOSUM naive). NO min / top-K / log / z-norm.

Output: results/fig2_cache/naive_baseline/tcrdist_cdr3_naive_fold{0-4}.npy
(consumed by gen_distance_comparison_cv_with_naive.py as the TCRdist box).
"""
import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score
from Bio.Align import substitution_matrices

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
RES = os.path.join(INPUT_DIR, 'results')
NAIVE_CACHE = os.path.join(RES, 'fig2_cache', 'naive_baseline')
os.makedirs(NAIVE_CACHE, exist_ok=True)
N_SUB, SEED = 2000, 42
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']

_B = substitution_matrices.load('BLOSUM62')
_ALPH = set(_B.alphabet)


def _bsd4(x, y):
    if x == y:
        return 0.0
    if x in _ALPH and y in _ALPH:
        return max(0.0, 4.0 - float(_B[x, y]))
    return 4.0   # unknown residue -> max per-position distance


def tcrdist_cdr3(a, b, gap_pen=12.0):
    la, lb = len(a), len(b)
    if la == lb:
        return sum(_bsd4(a[i], b[i]) for i in range(la))
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    gappos = min(6, 3 + (la - 5) // 2)
    gappos = max(0, min(gappos, la))
    ngap = lb - la
    a_al = a[:gappos] + ('-' * ngap) + a[gappos:]
    d = 0.0
    for x, y in zip(a_al, b):
        d += gap_pen if x == '-' else _bsd4(x, y)
    return d


def _get(df, t):
    if t in df.columns:
        return t
    m = {'CDR3a': 'cdr3_a', 'CDR3b': 'cdr3_b'}
    a = m.get(t)
    return a if a and a in df.columns else None


def load_test(fold):
    fd = os.path.join(RES, 'nettcr', 'cv_logdist', f'fold{fold}')
    tp = os.path.join(fd, 'test_predictions_with_label.csv')
    if not os.path.exists(tp):
        return None
    parts = [pd.read_csv(tp)]
    for v in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fd, v)
        if os.path.exists(vp):
            parts.append(pd.read_csv(vp)); break
    return pd.concat(parts, ignore_index=True)


def _fold_worker(fold):
    te = load_test(fold)
    trp = f'Data/tcr_seq/proc_files/tcr_cross_val_fold{fold}/train_data.csv'
    if te is None or not os.path.exists(trp):
        return fold, None
    tr = pd.read_csv(trp)
    ta, tb = _get(te, 'CDR3a'), _get(te, 'CDR3b')
    ra, rb = _get(tr, 'CDR3a'), _get(tr, 'CDR3b')
    test_a = te[ta].astype(str).tolist()
    test_b = te[tb].astype(str).tolist()
    tr_pairs = list(zip(tr[ra].astype(str), tr[rb].astype(str)))
    cnt = Counter(tr_pairs)
    uq = list(cnt.keys())
    w = np.array([cnt[s] for s in uq], dtype=float)
    if len(uq) > N_SUB:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(uq), N_SUB, replace=False)
        uq = [uq[j] for j in idx]
        w = w[idx]
    w = w / w.sum()
    ua = [p[0] for p in uq]
    ub = [p[1] for p in uq]
    # dedup test rows (TCRdist identical for identical (CDR3a,CDR3b))
    seen = {}
    d = np.zeros(len(test_a))
    for i in range(len(test_a)):
        key = (test_a[i], test_b[i])
        if key in seen:
            d[i] = seen[key]
            continue
        qa, qb = test_a[i], test_b[i]
        s = 0.0
        for j in range(len(uq)):
            s += w[j] * (tcrdist_cdr3(qa, ua[j]) + tcrdist_cdr3(qb, ub[j]))
        seen[key] = s
        d[i] = s
        if (len(seen)) % 500 == 0:
            print(f"  fold{fold}: {len(seen)} unique test done", flush=True)
    out = os.path.join(NAIVE_CACHE, f'tcrdist_cdr3_naive_fold{fold}.npy')
    np.save(out, d)
    print(f"fold{fold}: n_test={len(d)} uniq_test={len(seen)} "
          f"train_sub={len(uq)} -> {os.path.basename(out)}", flush=True)
    return fold, out


def binned_r(y, p, dd, nb=8):
    n = min(len(y), len(p), len(dd))
    y, p, dd = y[:n], p[:n], dd[:n]
    si = np.argsort(dd, kind='stable')
    bs = len(si) // nb
    bv, bd = [], []
    for i in range(nb):
        s = i * bs
        e = len(si) if i == nb - 1 else (i + 1) * bs
        ix = si[s:e]
        bv.append(average_precision_score(y[ix], p[ix])
                  if len(np.unique(y[ix])) > 1 else np.nan)
        bd.append(dd[ix].mean())
    v, x = np.array(bv), np.array(bd)
    m = ~np.isnan(v)
    return pearsonr(x[m], v[m])[0] if m.sum() >= 4 else np.nan  # signed


if __name__ == '__main__':
    with Pool(5) as pool:
        res = pool.map(_fold_worker, list(range(5)))
    dist = {f: np.load(o) for f, o in res if o}
    print(f"\n{'model':>10} | TCRdist-CDR3 naive |mean r| (AP, 5 folds)")
    allr = []
    for m in TCR_MODELS:
        rs = []
        for fold, dd in dist.items():
            fd = os.path.join(RES, m, 'cv_logdist', f'fold{fold}')
            tp = os.path.join(fd, 'test_predictions_with_label.csv')
            if not os.path.exists(tp):
                continue
            parts = [pd.read_csv(tp)]
            for v in ['val_predictions_with_label.csv', 'val_predictions.csv']:
                vp = os.path.join(fd, v)
                if os.path.exists(vp):
                    parts.append(pd.read_csv(vp)); break
            df = pd.concat(parts, ignore_index=True)
            yc = 'binder' if 'binder' in df else 'y_true'
            pc = 'prediction' if 'prediction' in df else 'y_prob'
            rs.append(binned_r(df[yc].values.astype(int),
                               df[pc].values.astype(float), dd))
        mr = abs(np.nanmean(rs)) if rs else float('nan')
        allr.append(mr)
        print(f"{m:>10} | {mr:.3f}")
    print(f"{'MEAN':>10} | {np.nanmean(allr):.3f}")
    print("ref: old SW-BLOSUM62 'TCRdist' single box was ~0.60 (not real TCRdist)")
