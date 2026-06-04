#!/usr/bin/env python3
"""BCR 3-method comparison: M-CBPE vs PAPE vs S2DD for antigen subset prediction.

Design:
  CT: per-variant LOO within SARS/flu domains (fold4cal pipeline)
  CV: within-fold halfsplit (from combined_bind_ab_cv)
  Each method uses ONLY its own features.

Baseline features: naive Levenshtein avg over ALL cal variant sequences.
  No topk, no subsampling. Caches per unique sequence.

Usage: python audit_bcr_baseline_comparison.py
"""
import os, sys, time, hashlib
import numpy as np
import pandas as pd
import Levenshtein
from scipy.stats import pearsonr

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
from calipper.core_v2_7 import (
    pape_eq4, fit_best_curve, predict_best_curve, VBIAS_BETA_LAM,
    adaptive_n_bins, MIN_BIN_SAMPLES
)
from PAPE.pape_core import (
    estimate_importance_weights, fit_weighted_calibration, apply_calibration
)
from MCBPE.mcbpe_core import (
    estimate_density_ratios, fit_weighted_calibrator,
    calibrate_predictions, estimate_metric_from_calibrated
)

RESULTS = os.path.join(INPUT_DIR, 'results')
FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
MIN_SAMPLES = 30

_lev_cache = {}
_cal_hash_cache = {}


def _cal_content_hash(cal_seqs):
    """Content-based hash for cal array. Cached by id for fast repeat lookups."""
    obj_id = id(cal_seqs)
    if obj_id not in _cal_hash_cache:
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
    Chains: Heavy + Light + variant_seq.
    Aligned with eval_baselines.py BL-1 naive_avg."""
    chain_dists = []
    for chain in ['heavy', 'light', 'variant_seq']:
        if chain in test_data and chain in cal_data:
            chain_dists.append(naive_lev_dist_single(test_data[chain], cal_data[chain]))
    if not chain_dists:
        raise ValueError("No chain data found for 3-chain naive Lev")
    return np.mean(chain_dists, axis=0)


def predict_subset(cal_y, cal_p, cal_d, cal_vs,
                    test_y, test_p, test_d, test_vs,
                    idx_map, model, split, metric,
                    strategy='antigen', **kwargs):
    """kwargs: cal_heavy, cal_light, test_heavy, test_light for 3-chain naive Lev."""
    records = []
    valid_mask = np.zeros(len(test_y), dtype=bool)
    for idx in idx_map.values():
        valid_mask[idx] = True
    vt_p, vt_d, vt_vs = test_p[valid_mask], test_d[valid_mask], test_vs[valid_mask]

    # Single-chain naive Levenshtein: Heavy only (receptor sequence).
    # Multi-chain combination is S2DD's contribution — baselines should not
    # benefit from it. Using Heavy as the most informative receptor chain.
    # PAPE: prediction-only DRE (NO distance input)
    # Distance-based distribution shift detection is S2DD's contribution.
    w_pape, _, _ = estimate_importance_weights(
        cal_p.reshape(-1, 1), vt_p.reshape(-1, 1))
    cm_pape = fit_weighted_calibration(cal_p, cal_y, w_pape)

    # M-CBPE: prediction-only DRE (NO distance input)
    mcbpe_w, _ = estimate_density_ratios(
        cal_p.reshape(-1, 1), vt_p.reshape(-1, 1))
    mcbpe_cal = fit_weighted_calibrator(cal_p, cal_y, mcbpe_w)

    # S2DD: S2DD dist DRE + curve
    w_s2dd, _, _ = estimate_importance_weights(
        np.stack([cal_d, cal_p], 1), np.stack([vt_d, vt_p], 1))
    cm_s2dd = fit_weighted_calibration(cal_p, cal_y, w_s2dd)

    # Curve fitting
    bd, bm, ba, bp = [], [], [], []
    if strategy == 'antigen':
        cal_subs = {}
        for sn in idx_map:
            mask = np.array([vhash(v) == sn for v in cal_vs])
            if mask.sum() >= MIN_SAMPLES:
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
        c_cal = apply_calibration(cm_s2dd, cal_p)
        nb = adaptive_n_bins(int((cal_y==1).sum()), int((cal_y==0).sum()))
        si_c = np.argsort(cal_d); bs = max(len(si_c)//nb, 1)
        for i in range(nb):
            s, e = i*bs, (len(si_c) if i==nb-1 else (i+1)*bs)
            idx = si_c[s:e]
            if len(idx) < MIN_BIN_SAMPLES: continue
            a = safe_metric(metric, cal_y[idx], cal_p[idx])
            p = pape_eq4(c_cal[idx], cal_p[idx], metric, threshold=0.5)
            if not np.isnan(a) and not np.isnan(p):
                bd.append(cal_d[idx].mean()); bm.append(cal_p[idx].mean())
                ba.append(a); bp.append(p)

    if len(bd) >= 4:
        fr = fit_best_curve(np.array(bd), np.array(bm),
                             np.array(ba)-np.array(bp), lam=VBIAS_BETA_LAM)
    else:
        fr = {'params': None}

    for sub_name, sub_idx in idx_map.items():
        sub_y, sub_p, sub_d = test_y[sub_idx], test_p[sub_idx], test_d[sub_idx]
        actual = safe_metric(metric, sub_y, sub_p)
        if np.isnan(actual): continue

        c_pape = apply_calibration(cm_pape, sub_p)
        pape_pred = pape_eq4(c_pape, sub_p, metric, threshold=0.5)
        mcbpe_c = calibrate_predictions(mcbpe_cal, sub_p)
        mcbpe_pred = estimate_metric_from_calibrated(mcbpe_c, metric)
        c_s2dd = apply_calibration(cm_s2dd, sub_p)
        s2dd_base = pape_eq4(c_s2dd, sub_p, metric, threshold=0.5)
        corr = float(predict_best_curve(fr, np.array([sub_d.mean()]),
                     np.array([sub_p.mean()]))[0]) if fr['params'] is not None else 0.0
        s2dd_pred = float(np.clip(s2dd_base + corr, 0, 1))

        records.append({
            'model': model, 'split': split, 'subset': sub_name,
            'strategy': strategy, 'metric': metric,
            'actual': actual, 'pape': pape_pred,
            'mcbpe': mcbpe_pred, 's2dd': s2dd_pred, 'n': len(sub_y),
        })
    return records


if __name__ == '__main__':
    t0 = time.time()
    all_records = []

    for mi, model in enumerate(MODELS):
        print(f"[{mi+1}/{len(MODELS)}] {model}...", flush=True)

        # Load CT data
        cal_path = os.path.join(FOLD4CAL, model, 'cal_predictions.csv')
        if not os.path.exists(cal_path): continue
        model_dir = os.path.join(FOLD4CAL, model)
        cal = pd.read_csv(cal_path)
        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
        cal['source'] = 'fold4_test'
        parts = [cal]
        for ts in ['A1-A11', 'unseen', 'flu']:
            fp = os.path.join(FOLD4CAL, model, f'{ts}_predictions.csv')
            if not os.path.exists(fp): continue
            df = pd.read_csv(fp)
            if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
            df['source'] = ts
            if 'data_source' not in df.columns:
                df['data_source'] = 'flu' if ts == 'flu' else 'sars'
            parts.append(df)
        pooled = pd.concat(parts, ignore_index=True)

        for metric in ['aucroc', 'ap', 'f1']:
            # CT: per-variant LOO within domains
            for domain in ['sars', 'flu']:
                dom_df = pooled[pooled['data_source'] == domain]
                variants = dom_df.groupby('variant_seq').size()
                valid = variants[variants >= MIN_SAMPLES].index.tolist()

                for held_v in valid:
                    test_mask = dom_df['variant_seq'] == held_v
                    cal_mask = ~test_mask
                    cal_sub = dom_df[cal_mask]
                    test_sub = dom_df[test_mask]
                    if len(test_sub) < 10: continue
                    ty = test_sub['rbd'].values.astype(int)
                    if ty.sum() == 0 or ty.sum() == len(ty): continue
                    cy = cal_sub['rbd'].values.astype(int)
                    if cy.sum() < 3 or (len(cy)-cy.sum()) < 3: continue

                    # 3-chain kwargs for this LOO split
                    ct_chain_kw = {}
                    for ch, col in [('heavy', 'Heavy'), ('light', 'Light')]:
                        if col in cal_sub.columns and col in test_sub.columns:
                            ct_chain_kw[f'cal_{ch}'] = cal_sub[col].values.astype(str)
                            ct_chain_kw[f'test_{ch}'] = test_sub[col].values.astype(str)

                    # Antigen strategy: variant = one test subset
                    test_subsets = {vhash(held_v): np.arange(len(test_sub))}
                    recs = predict_subset(
                        cy, cal_sub['pred_prob'].values.astype(float),
                        cal_sub['distance'].values.astype(float),
                        cal_sub['variant_seq'].values,
                        ty, test_sub['pred_prob'].values.astype(float),
                        test_sub['distance'].values.astype(float),
                        test_sub['variant_seq'].values,
                        test_subsets, model, 'CT', metric, strategy='antigen',
                        **ct_chain_kw)
                    all_records.extend(recs)

                    # Distance strategy: bins within variant
                    n_bins = min(8, len(test_sub) // MIN_SAMPLES)
                    if n_bins >= 2:
                        td = test_sub['distance'].values.astype(float)
                        si = np.argsort(td); bs = len(si) // n_bins
                        dist_map = {}
                        for i in range(n_bins):
                            s, e = i*bs, (len(si) if i==n_bins-1 else (i+1)*bs)
                            dist_map[f'dist_{i}'] = si[s:e]
                        recs = predict_subset(
                            cy, cal_sub['pred_prob'].values.astype(float),
                            cal_sub['distance'].values.astype(float),
                            cal_sub['variant_seq'].values,
                            ty, test_sub['pred_prob'].values.astype(float),
                            td, test_sub['variant_seq'].values,
                            dist_map, model, 'CT', metric, strategy='distance',
                            **ct_chain_kw)
                        all_records.extend(recs)

            # CV: halfsplit
            for fold in range(5):
                test_path = os.path.join(RESULTS, 'xbcr' if model == 'xbcr' else model,
                                          'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
                if not os.path.exists(test_path): continue
                te = pd.read_csv(test_path)
                if 'pred_prob' not in te.columns or 'distance' not in te.columns: continue

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
                vs = te['variant_seq'].values

                si = np.argsort(d)
                cal_idx, test_idx = si[::2], si[1::2]

                # 3-chain kwargs for CV halfsplit
                cv_chain_kw = {}
                for ch, col in [('heavy', 'Heavy'), ('light', 'Light')]:
                    if col in te.columns:
                        cv_chain_kw[f'cal_{ch}'] = te[col].values.astype(str)[cal_idx]
                        cv_chain_kw[f'test_{ch}'] = te[col].values.astype(str)[test_idx]

                # CV antigen splitting
                test_vs = vs[test_idx]
                unique_vs = pd.Series(test_vs).value_counts()
                valid_vs = unique_vs[unique_vs >= MIN_SAMPLES].index
                if len(valid_vs) >= 4:
                    ag_map = {vhash(v): np.where(test_vs == v)[0] for v in valid_vs}
                    recs = predict_subset(
                        y[cal_idx], p[cal_idx], d[cal_idx], vs[cal_idx],
                        y[test_idx], p[test_idx], d[test_idx], test_vs,
                        ag_map, model, 'CV', metric, strategy='antigen',
                        **cv_chain_kw)
                    all_records.extend(recs)

                # CV distance splitting
                td = d[test_idx]
                si_d = np.argsort(td); bs = max(len(si_d)//8, 1)
                if bs >= MIN_SAMPLES:
                    dist_map = {}
                    for i in range(8):
                        s, e = i*bs, (len(si_d) if i==7 else (i+1)*bs)
                        dist_map[f'dist_{i}'] = si_d[s:e]
                    recs = predict_subset(
                        y[cal_idx], p[cal_idx], d[cal_idx], vs[cal_idx],
                        y[test_idx], p[test_idx], d[test_idx], vs[test_idx],
                        dist_map, model, 'CV', metric, strategy='distance',
                        **cv_chain_kw)
                    all_records.extend(recs)

    df = pd.DataFrame(all_records)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Total: {len(df)} records.\n")

    methods = ['pape', 'mcbpe', 's2dd']
    method_names = {'pape': 'PAPE', 'mcbpe': 'M-CBPE', 's2dd': 'S2DD'}

    for met in ['aucroc', 'ap', 'f1']:
        for strategy in ['antigen', 'distance']:
            for split in ['CT', 'CV', 'CT+CV']:
                sub = df[(df['metric']==met) & (df['strategy']==strategy)]
                if split != 'CT+CV':
                    sub = sub[sub['split'] == split]
                sub = sub.dropna(subset=['actual'])
                if len(sub) < 3: continue
                print(f"=== {met} {strategy} {split} (n={len(sub)}) ===")
                for m in methods:
                    valid = sub.dropna(subset=[m])
                    if len(valid) < 3: continue
                    r, _ = pearsonr(valid[m], valid['actual'])
                    mae = np.abs(valid[m] - valid['actual']).mean()
                    print(f"  {method_names[m]:<10} r={r:.3f}, MAE={mae:.3f}")
                print()

    out = os.path.join(SCRIPT_DIR, '..', 'audit_bcr_baseline_results.csv')
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
