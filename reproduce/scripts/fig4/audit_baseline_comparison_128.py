#!/usr/bin/env python3
"""Fair 3-method comparison: Pure PAPE vs M-CBPE vs S2DD for epitope prediction.

Each method uses ONLY its own features:
  - Pure PAPE: naive Levenshtein avg DRE + pape_eq4
  - M-CBPE: naive Levenshtein avg DRE + logistic calibration
  - S2DD: S2DD LogDist K=50 DRE + pape_eq4 + epitope-bin curve

Design:
  CT: pool non-cal test sets (no v3/v4), split by epitope (≥128)
  CV: halfsplit (interleaved by ref model distance)
  All methods evaluated on the SAME test subsets.

AUDIT CHECKLIST (verify before running):
  [x] PAPE uses cal_lev (naive Lev), NOT cal_d (S2DD distance)
  [x] M-CBPE uses cal_lev, NOT cal_d
  [x] S2DD uses cal_d (pre-computed LogDist K=50 .npy)
  [x] naive_lev_dist: simple average over ALL cal epitopes, no topk, no subsampling
  [x] DRE target: pooled valid test subsets (same for all methods)
  [x] No v3/v4 in CT test pool
  [x] min_per_epitope=128

Usage: python audit_baseline_comparison_128.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import Levenshtein
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUFFIX_UNIFORM
from calipper.core_v2_7 import (
    pape_eq4, fit_best_curve, predict_best_curve, VBIAS_BETA_LAM,
    adaptive_n_bins, MIN_BIN_SAMPLES
)
from PAPE.pape_core import (
    estimate_importance_weights, fit_weighted_calibration, apply_calibration
)
try:
    from MCBPE.mcbpe_core import (
        estimate_density_ratios, fit_weighted_calibrator,
        calibrate_predictions, estimate_metric_from_calibrated
    )
    HAS_MCBPE = True
except ImportError:
    HAS_MCBPE = False
    print("WARNING: M-CBPE not available, skipping")

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')
MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
CAL_SETS = ['v3_combined', 'v4_combined']
MIN_EP = 128
MIN_CAL = 30

# ── Per-sequence Levenshtein cache ──
# Cache: {(test_seq_str, cal_content_hash) -> distance}
# Uses content-based cal hash so recomputed arrays with same content hit cache.
_lev_cache = {}
_cal_hash_cache = {}  # id(cal_seqs) -> content hash (avoid recomputing)


def _cal_content_hash(cal_seqs):
    """Content-based hash for cal array. Cached by id for fast repeat lookups."""
    obj_id = id(cal_seqs)
    if obj_id not in _cal_hash_cache:
        # Use hash of sorted unique sequences (fast, deterministic)
        _cal_hash_cache[obj_id] = hash(tuple(sorted(set(str(s) for s in cal_seqs[:200]))))
    return _cal_hash_cache[obj_id]


def naive_lev_dist_single(test_seqs, cal_seqs):
    """Naive avg Levenshtein for one chain. 1 - mean(ratio) over ALL cal.
    No topk, no subsampling, no log, no k/b scaling. Cached per unique seq."""
    cal_hash = _cal_content_hash(cal_seqs)
    dists = []
    for te in test_seqs:
        key = (str(te), cal_hash)
        if key in _lev_cache:
            dists.append(_lev_cache[key])
        else:
            ratios = [Levenshtein.ratio(str(te), str(ce)) for ce in cal_seqs]
            d = 1.0 - np.mean(ratios)
            _lev_cache[key] = d
            dists.append(d)
    return np.array(dists)


def naive_lev_dist_3chain(test_data, cal_data):
    """3-chain naive avg Levenshtein: uniform mean of per-chain distances.
    Chains: epitope + CDR3a + CDR3b (if available, else epitope only).
    Aligned with eval_baselines.py BL-1 naive_avg."""
    chain_dists = [naive_lev_dist_single(test_data['epitope'], cal_data['epitope'])]
    for chain in ['cdr3a', 'cdr3b']:
        if chain in test_data and chain in cal_data:
            chain_dists.append(naive_lev_dist_single(test_data[chain], cal_data[chain]))
    return np.mean(chain_dists, axis=0)


def load_ct(m, ts):
    pp = os.path.join(RESULTS, m, 'cross_test_logdist', 'predictions',
                      f'{ts}_predictions_with_label.csv')
    # Use uniform distances for S2DD (per_epitope strategy)
    dp = os.path.join(TCR_CACHE, f'{m}_ct_{ts}{DIST_SUFFIX_UNIFORM[DIST_TYPE]}')
    if not os.path.exists(pp) or not os.path.exists(dp):
        return None
    te = pd.read_csv(pp)
    d = np.load(dp)
    n = min(len(d), len(te))
    lc = 'binder' if 'binder' in te.columns else 'y_true'
    pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
    pep_col = 'peptide' if 'peptide' in te.columns else 'Epitope'
    cdr3a_col = 'CDR3a' if 'CDR3a' in te.columns else ('CDR3A' if 'CDR3A' in te.columns else None)
    cdr3b_col = 'CDR3b' if 'CDR3b' in te.columns else ('CDR3B' if 'CDR3B' in te.columns else None)
    result = {
        'label': te[lc].values[:n].astype(int),
        'pred': te[pc].values[:n].astype(float),
        'distance': d[:n].astype(float),
        'epitope': te[pep_col].values[:n],
    }
    if cdr3a_col and cdr3a_col in te.columns:
        result['cdr3a'] = te[cdr3a_col].values[:n].astype(str)
    if cdr3b_col and cdr3b_col in te.columns:
        result['cdr3b'] = te[cdr3b_col].values[:n].astype(str)
    return result


def load_cv(m, fold):
    test = pd.read_csv(os.path.join(RESULTS, m, 'cv_logdist', f'fold{fold}',
                                     'test_predictions_with_label.csv'))
    try:
        val = pd.read_csv(os.path.join(RESULTS, m, 'cv_logdist', f'fold{fold}',
                                        'val_predictions_with_label.csv'))
        df = pd.concat([test, val], ignore_index=True)
    except Exception as _e_baseline:
        import sys as _s_baseline
        print(f"  ⚠ FALLBACK [audit_baseline_comparison_128]: model={m} fold={fold} val_predictions missing ({_e_baseline}); using test only", file=_s_baseline.stderr, flush=True)
        df = test
    lc = 'binder' if 'binder' in df.columns else 'y_true'
    pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
    # Prefer uniform distances (per_epitope strategy)
    if DIST_TYPE == 'blosum-sqrt':
        suffixes = ['_blosumsqrt_uniform_dist', '_blosumsqrt_dist']
    else:
        suffixes = ['_uniform_dist', '_combined_dist', '_dist']
    for suffix in suffixes:
        dp = os.path.join(TCR_CACHE, f'{m}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dp):
            d = np.load(dp)
            break
    else:
        return None
    n = min(len(d), len(df))
    pep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
    cdr3a_col = 'CDR3a' if 'CDR3a' in df.columns else ('CDR3A' if 'CDR3A' in df.columns else None)
    cdr3b_col = 'CDR3b' if 'CDR3b' in df.columns else ('CDR3B' if 'CDR3B' in df.columns else None)
    result = {
        'label': df[lc].values[:n].astype(int),
        'pred': df[pc].values[:n].astype(float),
        'distance': d[:n].astype(float),
        'epitope': df[pep_col].values[:n],
    }
    if cdr3a_col and cdr3a_col in df.columns:
        result['cdr3a'] = df[cdr3a_col].values[:n].astype(str)
    if cdr3b_col and cdr3b_col in df.columns:
        result['cdr3b'] = df[cdr3b_col].values[:n].astype(str)
    return result


def split_ep(eps, min_ep=MIN_EP):
    result = {}
    valid = pd.Series(eps).value_counts()
    valid = valid[valid >= min_ep].index
    for ep in valid:
        result[ep[:30]] = np.where(eps == ep)[0]
    return result


def predict_one_split(cal_y, cal_p, cal_d, cal_ep,
                       test_y, test_p, test_d, test_ep,
                       idx_map, model, split_label, test_src=None,
                       strategy='epitope', metric='aucroc', **kwargs):
    """Run all 3 methods on the same test subsets. Returns list of records.
    kwargs: cal_cdr3a, cal_cdr3b, test_cdr3a, test_cdr3b for 3-chain naive Lev."""
    records = []

    # Pooled valid test samples (for DRE target)
    valid_test_mask = np.zeros(len(test_y), dtype=bool)
    for idx in idx_map.values():
        valid_test_mask[idx] = True
    vt_p = test_p[valid_test_mask]
    vt_d = test_d[valid_test_mask]
    vt_ep = test_ep[valid_test_mask]

    # ── PAPE: prediction-only DRE (NO distance input) ──
    # PAPE and M-CBPE originally do not use distance as input.
    # Distance-based distribution shift detection is S2DD's contribution.
    # Baselines use DRE on [prediction] only — the standard PAPE/M-CBPE design.
    w_pape, _, _ = estimate_importance_weights(
        cal_p.reshape(-1, 1),
        vt_p.reshape(-1, 1))
    cm_pape = fit_weighted_calibration(cal_p, cal_y, w_pape)

    # ── M-CBPE: prediction-only DRE (NO distance input) ──
    if HAS_MCBPE:
        mcbpe_w, _ = estimate_density_ratios(
            cal_p.reshape(-1, 1),
            vt_p.reshape(-1, 1))
        mcbpe_cal = fit_weighted_calibrator(cal_p, cal_y, mcbpe_w)

    # ── S2DD: S2DD distance DRE + epitope-bin curve ──
    # AUDIT: features are [cal_d, cal_p] — uses S2DD distance
    w_s2dd, _, _ = estimate_importance_weights(
        np.stack([cal_d, cal_p], axis=1),
        np.stack([vt_d, vt_p], axis=1))
    cm_s2dd = fit_weighted_calibration(cal_p, cal_y, w_s2dd)

    # Curve fitting: epitope-bin for epitope strategy, distance-bin for distance
    bd, bm, ba, bp = [], [], [], []
    if strategy == 'epitope':
        # Epitope-bin curve from cal subsets
        cal_subs = {}
        for sn in idx_map:
            mask = np.array([str(e)[:30] == sn for e in cal_ep])
            if mask.sum() >= MIN_CAL:
                cal_subs[sn] = (cal_y[mask], cal_p[mask], cal_d[mask])
        if len(cal_subs) >= 4:
            for sn, (sy, sp, sd) in cal_subs.items():
                a = safe_metric(metric, sy, sp)
                cs = apply_calibration(cm_s2dd, sp)
                p = pape_eq4(cs, sp, metric, threshold=0.5)
                if not np.isnan(a) and not np.isnan(p):
                    bd.append(sd.mean()); bm.append(sp.mean())
                    ba.append(a); bp.append(p)
    else:
        # Distance-bin curve
        c_cal_s2dd = apply_calibration(cm_s2dd, cal_p)
        n_bins = adaptive_n_bins(int((cal_y == 1).sum()), int((cal_y == 0).sum()))
        si_c = np.argsort(cal_d)
        bs = max(len(si_c) // n_bins, 1)
        for i in range(n_bins):
            s, e = i * bs, (len(si_c) if i == n_bins - 1 else (i + 1) * bs)
            idx = si_c[s:e]
            if len(idx) < MIN_BIN_SAMPLES: continue
            a = safe_metric(metric, cal_y[idx], cal_p[idx])
            p = pape_eq4(c_cal_s2dd[idx], cal_p[idx], metric, threshold=0.5)
            if not np.isnan(a) and not np.isnan(p):
                bd.append(cal_d[idx].mean()); bm.append(cal_p[idx].mean())
                ba.append(a); bp.append(p)

    if len(bd) >= 4:
        res = np.array(ba) - np.array(bp)
        fr = fit_best_curve(np.array(bd), np.array(bm), res, lam=VBIAS_BETA_LAM)
    else:
        fr = {'params': None}

    # ── Predict each test subset ──
    for sub_name, sub_idx in idx_map.items():
        sub_y = test_y[sub_idx]
        sub_p = test_p[sub_idx]
        sub_d = test_d[sub_idx]
        actual = safe_metric(metric, sub_y, sub_p)
        if np.isnan(actual):
            continue

        # PAPE prediction
        c_pape = apply_calibration(cm_pape, sub_p)
        pape_pred = pape_eq4(c_pape, sub_p, metric, threshold=0.5)

        # M-CBPE prediction
        if HAS_MCBPE:
            mcbpe_c = calibrate_predictions(mcbpe_cal, sub_p)
            mcbpe_pred = estimate_metric_from_calibrated(mcbpe_c, metric)
        else:
            mcbpe_pred = np.nan

        # S2DD prediction
        c_s2dd = apply_calibration(cm_s2dd, sub_p)
        s2dd_base = pape_eq4(c_s2dd, sub_p, metric, threshold=0.5)
        if fr['params'] is not None:
            corr = float(predict_best_curve(
                fr, np.array([sub_d.mean()]), np.array([sub_p.mean()]))[0])
        else:
            corr = 0.0
        s2dd_pred = float(np.clip(s2dd_base + corr, 0, 1))

        # Seen/unseen tag
        if test_src is not None:
            src_mode = pd.Series(test_src[sub_idx]).mode()[0]
            seen = 'seen' if src_mode == 'seen_test' else 'unseen'
        else:
            seen = 'cv'

        records.append({
            'model': model, 'split': split_label, 'subset': sub_name,
            'strategy': strategy, 'metric': metric,
            'actual': actual,
            'pape': pape_pred,
            'mcbpe': mcbpe_pred,
            's2dd': s2dd_pred,
            'seen': seen,
            'n': len(sub_y),
        })

    return records


if __name__ == '__main__':
    t0 = time.time()
    all_records = []

    METRICS_LIST = ['aucroc', 'ap', 'f1']
    for mi, model in enumerate(MODELS):
      for metric in METRICS_LIST:
        if metric == 'aucroc':
            print(f"[{mi+1}/{len(MODELS)}] {model}...", flush=True)

        # ── CT ──
        ct = {ts: load_ct(model, ts) for ts in CT_SETS}
        ct = {k: v for k, v in ct.items() if v is not None}
        cal_sets = [s for s in CAL_SETS if s in ct]
        test_keys = [s for s in CT_SETS if s in ct and s not in CAL_SETS]

        if cal_sets and test_keys:
            cal_y = np.concatenate([ct[s]['label'] for s in cal_sets])
            cal_p = np.concatenate([ct[s]['pred'] for s in cal_sets])
            cal_d = np.concatenate([ct[s]['distance'] for s in cal_sets])
            cal_ep = np.concatenate([ct[s]['epitope'] for s in cal_sets])
            test_y = np.concatenate([ct[s]['label'] for s in test_keys])
            test_p = np.concatenate([ct[s]['pred'] for s in test_keys])
            test_d = np.concatenate([ct[s]['distance'] for s in test_keys])
            test_ep = np.concatenate([ct[s]['epitope'] for s in test_keys])
            test_src = np.concatenate([np.full(len(ct[s]['label']), s) for s in test_keys])
            # 3-chain CDR3 data for naive Lev
            chain_kw = {}
            for ch in ['cdr3a', 'cdr3b']:
                cal_ch = [ct[s].get(ch) for s in cal_sets if ch in ct[s]]
                test_ch = [ct[s].get(ch) for s in test_keys if ch in ct[s]]
                if len(cal_ch) == len(cal_sets) and len(test_ch) == len(test_keys):
                    chain_kw[f'cal_{ch}'] = np.concatenate(cal_ch)
                    chain_kw[f'test_{ch}'] = np.concatenate(test_ch)

            # CT epitope splitting
            idx_map = split_ep(test_ep)
            if idx_map:
                recs = predict_one_split(cal_y, cal_p, cal_d, cal_ep,
                                          test_y, test_p, test_d, test_ep,
                                          idx_map, model, 'CT', test_src,
                                          strategy='epitope', metric=metric,
                                          **chain_kw)
                all_records.extend(recs)

            # CT distance splitting (per held test set LOO)
            for held in sorted(ct.keys()):
                if held not in ct: continue
                held_cal_keys = [s for s in cal_sets if s != held]
                if not held_cal_keys: continue
                hc_y = np.concatenate([ct[s]['label'] for s in held_cal_keys])
                hc_p = np.concatenate([ct[s]['pred'] for s in held_cal_keys])
                hc_d = np.concatenate([ct[s]['distance'] for s in held_cal_keys])
                hc_ep = np.concatenate([ct[s]['epitope'] for s in held_cal_keys])
                test_h = ct[held]
                n_bins = 8
                si_t = np.argsort(test_h['distance'])
                bs = max(len(si_t) // n_bins, 1)
                if bs < MIN_CAL: continue
                dist_idx_map = {}
                for i in range(n_bins):
                    s, e = i * bs, (len(si_t) if i == n_bins - 1 else (i + 1) * bs)
                    dist_idx_map[f'dist_bin{i}'] = si_t[s:e]
                # 3-chain CDR3 data for this LOO split
                held_chain_kw = {}
                for ch in ['cdr3a', 'cdr3b']:
                    hc_ch = [ct[s].get(ch) for s in held_cal_keys if ch in ct[s]]
                    if len(hc_ch) == len(held_cal_keys) and ch in ct[held]:
                        held_chain_kw[f'cal_{ch}'] = np.concatenate(hc_ch)
                        held_chain_kw[f'test_{ch}'] = ct[held][ch]
                recs = predict_one_split(hc_y, hc_p, hc_d, hc_ep,
                                          test_h['label'], test_h['pred'],
                                          test_h['distance'], test_h['epitope'],
                                          dist_idx_map, model, 'CT', None,
                                          strategy='distance', metric=metric,
                                          **held_chain_kw)
                all_records.extend(recs)

        # ── CV ──
        for fold in range(5):
            data = load_cv(model, fold)
            if data is None:
                continue
            ref = load_cv('nettcr', fold)
            if ref and len(ref['distance']) == len(data['label']):
                si = np.argsort(ref['distance'])
            else:
                si = np.argsort(data['distance'])
            cal_idx, test_idx = si[::2], si[1::2]

            # 3-chain CDR3 data for CV halfsplit
            cv_chain_kw = {}
            for ch in ['cdr3a', 'cdr3b']:
                if ch in data:
                    cv_chain_kw[f'cal_{ch}'] = data[ch][cal_idx]
                    cv_chain_kw[f'test_{ch}'] = data[ch][test_idx]

            # CV epitope splitting
            idx_map = split_ep(data['epitope'][test_idx])
            if idx_map:
                recs = predict_one_split(
                    data['label'][cal_idx], data['pred'][cal_idx],
                    data['distance'][cal_idx], data['epitope'][cal_idx],
                    data['label'][test_idx], data['pred'][test_idx],
                    data['distance'][test_idx], data['epitope'][test_idx],
                    idx_map, model, 'CV', strategy='epitope', metric=metric,
                    **cv_chain_kw)
                all_records.extend(recs)

            # CV distance splitting
            test_d_cv = data['distance'][test_idx]
            si_d = np.argsort(test_d_cv)
            n_bins = 8; bs = max(len(si_d) // n_bins, 1)
            if bs >= MIN_CAL:
                dist_idx_map = {}
                for i in range(n_bins):
                    s, e = i * bs, (len(si_d) if i == n_bins - 1 else (i + 1) * bs)
                    dist_idx_map[f'dist_bin{i}'] = si_d[s:e]
                recs = predict_one_split(
                    data['label'][cal_idx], data['pred'][cal_idx],
                    data['distance'][cal_idx], data['epitope'][cal_idx],
                    data['label'][test_idx], data['pred'][test_idx],
                    data['distance'][test_idx], data['epitope'][test_idx],
                    dist_idx_map, model, 'CV', strategy='distance', metric=metric,
                    **cv_chain_kw)
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Total: {len(df)} records.\n")

    # ── Results ──
    methods = ['pape', 's2dd']
    if HAS_MCBPE:
        methods = ['pape', 'mcbpe', 's2dd']

    method_names = {'pape': 'Pure PAPE', 'mcbpe': 'M-CBPE', 's2dd': 'S2DD'}

    for met in METRICS_LIST:
     for strategy in ['epitope', 'distance']:
        strat_df = df[(df['strategy'] == strategy) & (df['metric'] == met)] if 'strategy' in df.columns else df
        for split_label in ['CT', 'CV', 'CT+CV']:
            sub = strat_df if split_label == 'CT+CV' else strat_df[strat_df['split'] == split_label]
            sub = sub.dropna(subset=['actual'])
            if len(sub) < 3:
                continue
            print(f"=== {met} {strategy} {split_label} (n={len(sub)}) ===")
            print(f"  {'Method':<15} {'r':>7} {'MAE':>7}")
            print(f"  {'-'*32}")
            for m in methods:
                valid = sub.dropna(subset=[m])
                if len(valid) < 3:
                    continue
                r, _ = pearsonr(valid[m], valid['actual'])
                mae = np.abs(valid[m] - valid['actual']).mean()
                print(f"  {method_names[m]:<15} {r:>7.3f} {mae:>7.3f}")
            print()

    # CT seen/unseen
    ct = df[df['split'] == 'CT']
    if 'seen' in ct.columns and len(ct) > 0:
        print("=== CT seen/unseen ===")
        for label in ['seen', 'unseen']:
            sub = ct[ct['seen'] == label]
            if len(sub) < 3:
                continue
            print(f"  {label}:")
            for m in methods:
                valid = sub.dropna(subset=[m])
                if len(valid) < 3:
                    continue
                r, _ = pearsonr(valid[m], valid['actual'])
                print(f"    {method_names[m]:<15} r={r:.3f}, n={len(valid)}")
        print()

    # Save
    out = os.path.join(SCRIPT_DIR, '..', f'audit_baseline_comparison_128_{DIST_TYPE}_results.csv')
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
