"""BCR ESM2 **S2DD** aligned to the canonical cosine-log pipeline.

Same recipe as the aligned TCR ESM2 S2DD (`compute_esm2_cv_dist`), adapted to
BCR: chains = [Heavy, Light, variant_seq], BCR params K=30, k=0.1, b=0.03
(matching the canonical BCR Lev S2DD). Pipeline:

  per-pair  d = log(k*(1 - cos_sim + b))   (cosine on L2-normed ESM2 embeds)
  z-norm    full-pair stats from 500 training ROWS (with multiplicity) vs ALL
            train rows, flattened, no top-K (== compute_pairwise_chain_stats)
  weights   sigma_C = z-norm sigma * Simpson concentration, selectors only
  combine   weighted_max_znorm: argmax_chain(w_scaled*z) -> winner z
  reduce    mean of K=30 smallest combined distances over refs (combine->topK)

Replaces the old `esm2_log_s2dd` CSV row (unaligned multi-chain Euclidean
pipeline). Outputs:
  results/fig2_cache/naive_baseline/bcr_cv_fold{0-4}_esm2_s2dd_raw.npy
  results/fig2_cache/bcr_all_esm2_emb.npz   (Heavy+Light+variant embeddings)
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
CACHE = os.path.join(RES, 'fig2_cache')
NAIVE_CACHE = os.path.join(CACHE, 'naive_baseline')
HEAVY_EMB = os.path.join(CACHE, 'bcr_heavy_esm2_emb.npz')   # from naive step
ALL_EMB = os.path.join(CACHE, 'bcr_all_esm2_emb.npz')

CHAINS = ['Heavy', 'Light', 'variant_seq']
K_TOPK, KLOG, BLOG = 30, 0.1, 0.03           # canonical BCR S2DD params
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
FOLD_BASE = 'combined_bind_ab_cv'


def fdir(m, f):
    return os.path.join(RES, m, FOLD_BASE, f'fold{f}')


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
    # SIGNED r; aggregate as abs(mean over folds) to match the panel.
    return pearsonr(dd[m], v[m])[0] if m.sum() >= 4 else np.nan


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# ── gather sequences per chain ──
folds = []
need = {c: set() for c in CHAINS}
for f in range(5):
    fd = fdir('xbcr', f)
    if not (os.path.exists(f'{fd}/train.csv') and os.path.exists(f'{fd}/test.csv')):
        continue
    tr = pd.read_csv(f'{fd}/train.csv')
    te = pd.read_csv(f'{fd}/test.csv')
    rec = {'fold': f}
    for c in CHAINS:
        rec[f'tr_{c}'] = tr[c].astype(str).tolist()
        rec[f'te_{c}'] = te[c].astype(str).tolist()
        need[c].update(rec[f'tr_{c}'])
        need[c].update(rec[f'te_{c}'])
    folds.append(rec)

# ── ESM2 embeddings (reuse Heavy cache; compute Light+variant; combined cache) ──
EMB = {}
if os.path.exists(ALL_EMB):
    z = np.load(ALL_EMB, allow_pickle=True)
    EMB = dict(zip(z['seqs'].tolist(), z['embs']))
elif os.path.exists(HEAVY_EMB):
    z = np.load(HEAVY_EMB, allow_pickle=True)
    EMB = dict(zip(z['seqs'].tolist(), z['embs']))
all_need = set().union(*need.values())
missing = sorted(s for s in all_need if s not in EMB)
print(f"emb: {len(EMB)} cached, {len(missing)} missing")
if missing:
    from eval_baselines_esm2 import load_esm2_model, compute_embeddings
    model, bc = load_esm2_model(device='cuda')
    # short chains big batch; long variant_seq (~560 AA) small batch
    short = [s for s in missing if len(s) <= 160]
    longs = [s for s in missing if len(s) > 160]
    if short:
        print(f"embedding {len(short)} short seqs...")
        EMB.update(compute_embeddings(short, model, bc, 'cuda', batch_size=128))
    if longs:
        print(f"embedding {len(longs)} long seqs (variant_seq)...")
        EMB.update(compute_embeddings(longs, model, bc, 'cuda', batch_size=8))
    seqs = list(EMB.keys())
    np.savez_compressed(ALL_EMB, seqs=np.array(seqs, dtype=object),
                        embs=np.array([EMB[s] for s in seqs], dtype=np.float32))
    print(f"saved {len(seqs)} embeddings -> {ALL_EMB}")

DIM = len(next(iter(EMB.values())))
ZERO = np.zeros(DIM, dtype=np.float32)

# ── per-fold canonical S2DD ──
s2dd_by_fold = {}
for rec in folds:
    f = rec['fold']
    te_e, tr_e, tidx, stats, wraw = [], [], [], [], []
    rng = np.random.RandomState(42)
    for c in CHAINS:
        rseq, tseq = rec[f'tr_{c}'], rec[f'te_{c}']
        uq = list(dict.fromkeys(rseq))
        ui = {s: i for i, s in enumerate(uq)}
        Er = l2(np.array([EMB.get(s, ZERO) for s in uq], dtype=np.float64))
        Et = l2(np.array([EMB.get(s, ZERO) for s in tseq], dtype=np.float64))
        ridx = np.array([ui[s] for s in rseq])
        nrow = len(ridx)
        ns = min(500, nrow)
        srow = rng.choice(nrow, ns, replace=False)            # ROWS, multiplicity
        sub_cos = 1.0 - Er[ridx[srow]] @ Er.T
        sub_d = np.log(KLOG * (sub_cos + BLOG))[:, ridx].ravel()
        mu, sg = float(sub_d.mean()), max(float(sub_d.std()), 1e-9)
        stats.append((mu, sg)); te_e.append(Et); tr_e.append(Er); tidx.append(ridx)
        cnt = Counter(rseq); n = len(rseq)
        cval = sum(x*(x-1) for x in cnt.values())/(n*(n-1)) if n > 1 else 1.0
        wraw.append(sg * cval)
    W = np.array(wraw); W = W / W.sum(); Wsc = W * len(W)
    nT = len(rec['te_Heavy'])
    d_out = np.zeros(nT)
    for s in range(0, nT, 512):
        e = min(s + 512, nT)
        Z = []
        for c in range(len(CHAINS)):
            zc = (np.log(KLOG * ((1.0 - te_e[c][s:e] @ tr_e[c].T) + BLOG))[:, tidx[c]]
                  - stats[c][0]) / stats[c][1]
            Z.append(zc)
        Z = np.stack(Z, 0)
        win = np.argmax(Z * Wsc[:, None, None], 0)
        D = np.take_along_axis(Z, win[None], 0)[0]
        kk = min(K_TOPK, D.shape[1])
        d_out[s:e] = np.partition(D, kk - 1, axis=1)[:, :kk].mean(1)
    out = os.path.join(NAIVE_CACHE, f'bcr_cv_fold{f}_esm2_s2dd_raw.npy')
    np.save(out, d_out)
    s2dd_by_fold[f] = d_out
    print(f"fold{f}: n_test={nT} weights(H,L,V)="
          f"{W[0]:.3f},{W[1]:.3f},{W[2]:.3f} -> {os.path.basename(out)}")

# ── per-model |mean r| (AP), panel convention abs(mean signed) ──
print(f"\n{'model':>10} | ESM2 S2DD |mean r| (AP, 5 folds)")
allr = []
for m in BCR_MODELS:
    rs = []
    for rec in folds:
        f = rec['fold']
        pp = os.path.join(fdir(m, f), 'test.csv')
        if not os.path.exists(pp):
            continue
        df = pd.read_csv(pp)
        if 'rbd' not in df or 'pred_prob' not in df:
            continue
        rs.append(binned_r(df['rbd'].values.astype(int),
                           df['pred_prob'].values.astype(float),
                           s2dd_by_fold[f]))
    mr = abs(np.nanmean(rs)) if rs else float('nan')
    allr.append(mr)
    print(f"{m:>10} | {mr:.3f}")
print(f"{'MEAN':>10} | {np.nanmean(allr):.3f}")
print("ref: BCR ESM2 naive (Heavy) ~0.15 ; old esm2_log_s2dd CSV ~0.35")
