"""Length-controlled BLOSUM naive.

Diagnostic (diag_blosum_naive_why.py) showed the raw BLOSUM naive
(normalized SW-BLOSUM62, single-chain) has an intrinsic CDR3b/Heavy LENGTH
dependence (corr ~+0.33 with length; Lev-naive ~0), inflating its binned_r
and making it sign-unstable across seen/unseen. ESM2/Lev naive have no such
length artifact, so only BLOSUM needs control.

Control = per-fold OLS residual of the raw BLOSUM-naive distance on chain
length: d_ctrl = d - (a*len + b). This removes the length confound while
keeping the baseline model-independent and requiring no re-alignment. The
residual is a length-orthogonalized distributional-distance signal.

Outputs (consumed by the panel as the BLOSUM naive):
  TCR:  tcr_cv_fold{0-4}_blosum_naive_raw.npy   (OVERWRITES with len-ctrl)
  BCR:  bcr_cv_fold{0-4}_blosum_naive_raw.npy   (OVERWRITES with len-ctrl)
Raw versions are first archived to *_blosum_naive_RAWUNCTRL.npy.
Prints per-model |mean r| (signed -> abs(mean)) before vs after control.
"""
import os, sys, shutil
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
RES = os.path.join(INPUT_DIR, 'results')
NC = os.path.join(RES, 'fig2_cache', 'naive_baseline')
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']


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
    v, x = np.array(bv), np.array(bd)
    m = ~np.isnan(v)
    return pearsonr(x[m], v[m])[0] if m.sum() >= 4 else np.nan  # signed


def residualize(d, length):
    A = np.vstack([length.astype(float), np.ones(len(length))]).T
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return d - A @ coef


def load_tcr_test(model, fold):
    fd = os.path.join(RES, model, 'cv_logdist', f'fold{fold}')
    tp = os.path.join(fd, 'test_predictions_with_label.csv')
    if not os.path.exists(tp):
        return None
    parts = [pd.read_csv(tp)]
    vp = os.path.join(fd, 'val_predictions_with_label.csv')
    if os.path.exists(vp):
        parts.append(pd.read_csv(vp))
    return pd.concat(parts, ignore_index=True)


def run(domain):
    if domain == 'tcr':
        models, chain, pred_test = TCR_MODELS, 'CDR3b', load_tcr_test
        pref = 'tcr_cv_fold'
        def lens(fold):
            te = load_tcr_test('nettcr', fold)
            c = 'CDR3b' if 'CDR3b' in te else 'cdr3_b'
            return te[c].astype(str).str.len().values.astype(float)
        def ypred(model, fold):
            df = load_tcr_test(model, fold)
            if df is None:
                return None
            yc = 'binder' if 'binder' in df else 'y_true'
            pc = 'prediction' if 'prediction' in df else 'y_prob'
            return df[yc].values.astype(int), df[pc].values.astype(float)
    else:
        models, pref = BCR_MODELS, 'bcr_cv_fold'
        def lens(fold):
            te = pd.read_csv(f'{RES}/xbcr/combined_bind_ab_cv/fold{fold}/test.csv')
            return te['Heavy'].astype(str).str.len().values.astype(float)
        def ypred(model, fold):
            pp = f'{RES}/{model}/combined_bind_ab_cv/fold{fold}/test.csv'
            if not os.path.exists(pp):
                return None
            df = pd.read_csv(pp)
            return df['rbd'].values.astype(int), df['pred_prob'].values.astype(float)

    raw, ctrl, length = {}, {}, {}
    for f in range(5):
        rawf = os.path.join(NC, f'{pref}{f}_blosum_naive_raw.npy')
        if not os.path.exists(rawf):
            continue
        bak = os.path.join(NC, f'{pref}{f}_blosum_naive_RAWUNCTRL.npy')
        if not os.path.exists(bak):
            shutil.copy(rawf, bak)          # preserve original raw
        d = np.load(bak)                    # always start from original raw
        L = lens(f)
        n = min(len(d), len(L))
        d, L = d[:n], L[:n]
        length[f] = L
        raw[f] = d
        ctrl[f] = residualize(d, L)
        np.save(rawf, ctrl[f])              # panel consumes this filename

    print(f"\n===== {domain.upper()} BLOSUM naive: raw -> length-controlled =====")
    print(f"corr(raw, length) per fold: " +
          ", ".join(f"f{f}={pearsonr(raw[f], length[f])[0]:+.2f}" for f in raw))
    print(f"corr(ctrl,length) per fold: " +
          ", ".join(f"f{f}={pearsonr(ctrl[f], length[f])[0]:+.2f}" for f in ctrl))
    print(f"\n{'model':>10} | raw |mean r| | len-ctrl |mean r|")
    rr, cc = [], []
    for m in models:
        rs_raw, rs_ctrl = [], []
        for f in raw:
            yp = ypred(m, f)
            if yp is None:
                continue
            y, p = yp
            rs_raw.append(binned_r(y, p, raw[f]))
            rs_ctrl.append(binned_r(y, p, ctrl[f]))
        a = abs(np.nanmean(rs_raw)) if rs_raw else float('nan')
        b = abs(np.nanmean(rs_ctrl)) if rs_ctrl else float('nan')
        rr.append(a); cc.append(b)
        print(f"{m:>10} | {a:9.3f}   | {b:9.3f}")
    print(f"{'MEAN':>10} | {np.nanmean(rr):9.3f}   | {np.nanmean(cc):9.3f}")


if __name__ == '__main__':
    run('tcr')
    run('bcr')
