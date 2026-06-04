#!/usr/bin/env python3
"""Compare S2DD(sigma_C) vs S2DD(uniform+znorm_sum) for subset prediction.

Tests whether uniform combining strategy improves per-epitope/per-antigen
performance prediction in the v2.7 framework.

Design:
  4 methods: PAPE, M-CBPE, S2DD(sigma_C), S2DD(uniform)
  - PAPE/M-CBPE: 3-chain naive Levenshtein (unchanged from audit scripts)
  - S2DD(sigma_C): degradation strategy (weighted_max_znorm, pre-computed .npy)
  - S2DD(uniform): per_epitope strategy (uniform+znorm_sum)
  - Both S2DD variants use the SAME v2.7 prediction pipeline:
    pape_eq4 DRE + fit_best_curve vbias + predict_best_curve correction

  TCR: pool CT non-cal sets, split by epitope (>=128). CV halfsplit.
  BCR: per-variant LOO within SARS/flu domains. CV halfsplit.

AUDIT CHECKLIST:
  [x] S2DD(sigma_C) uses cal_d_sc (from *_dist.npy or 'distance' column)
  [x] S2DD(uniform) uses cal_d_un (from *_uniform_dist.npy or computed inline)
  [x] PAPE/M-CBPE use cal_lev (3-chain naive Lev) — NO S2DD distance
  [x] Both S2DD variants use identical prediction code (pape_eq4 + fit_best_curve)
  [x] All methods evaluated on the SAME test subsets
  [x] No k/b scaling in naive Lev, no topk, no subsampling

Usage:
  python audit_uniform_vs_sigmaC.py --domain tcr   # TCR per-epitope
  python audit_uniform_vs_sigmaC.py --domain bcr   # BCR per-antigen
  python audit_uniform_vs_sigmaC.py --domain both  # both (default)
"""
import argparse
import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
import Levenshtein
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from calipper.core_v2_7 import (
    pape_eq4, fit_best_curve, predict_best_curve, VBIAS_BETA_LAM,
    adaptive_n_bins, MIN_BIN_SAMPLES,
)
from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
)
from PAPE.pape_core import (
    estimate_importance_weights, fit_weighted_calibration, apply_calibration,
)
from MCBPE.mcbpe_core import (
    estimate_density_ratios, fit_weighted_calibrator,
    calibrate_predictions, estimate_metric_from_calibrated,
)

RESULTS = os.path.join(INPUT_DIR, 'results')

# ── Naive Levenshtein (3-chain, cached) ──────────────────────────────
_lev_cache = {}
_cal_hash_cache = {}


def _cal_content_hash(cal_seqs):
    obj_id = id(cal_seqs)
    if obj_id not in _cal_hash_cache:
        _cal_hash_cache[obj_id] = hash(tuple(sorted(set(str(s) for s in cal_seqs[:200]))))
    return _cal_hash_cache[obj_id]


def naive_lev_dist_single(test_seqs, cal_seqs):
    """1 - mean(ratio) over ALL cal. No topk, no subsampling, no scaling."""
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


def naive_lev_dist_3chain(test_data, cal_data, chain_keys):
    """Uniform mean of per-chain naive Lev distances."""
    chain_dists = []
    for ch in chain_keys:
        if ch in test_data and ch in cal_data:
            chain_dists.append(naive_lev_dist_single(test_data[ch], cal_data[ch]))
    if not chain_dists:
        raise ValueError("No chain data for 3-chain naive Lev")
    return np.mean(chain_dists, axis=0)


# ── S2DD prediction (shared by sigma_C and uniform) ─────────────────
def s2dd_predict_subsets(cal_y, cal_p, cal_d, cal_ep,
                         test_y, test_p, test_d, test_ep,
                         idx_map, metric, min_cal=30):
    """Run S2DD v2.7 prediction pipeline on test subsets.
    Identical code for both sigma_C and uniform — only cal_d/test_d differ.

    Returns list of (sub_name, predicted, actual) tuples.
    """
    # Pooled valid test samples for DRE
    valid_mask = np.zeros(len(test_y), dtype=bool)
    for idx in idx_map.values():
        valid_mask[idx] = True
    vt_p = test_p[valid_mask]
    vt_d = test_d[valid_mask]

    # DRE with S2DD distance
    w_s2dd, _, _ = estimate_importance_weights(
        np.stack([cal_d, cal_p], axis=1),
        np.stack([vt_d, vt_p], axis=1))
    cm_s2dd = fit_weighted_calibration(cal_p, cal_y, w_s2dd)

    # Epitope/antigen-bin curve from cal subsets
    bd, bm, ba, bp = [], [], [], []
    cal_subs = {}
    for sn in idx_map:
        mask = np.array([str(e)[:30] == sn for e in cal_ep])
        if mask.sum() >= min_cal:
            cal_subs[sn] = (cal_y[mask], cal_p[mask], cal_d[mask])
    if len(cal_subs) >= 4:
        for sn, (sy, sp, sd) in cal_subs.items():
            a = safe_metric(metric, sy, sp)
            cs = apply_calibration(cm_s2dd, sp)
            p = pape_eq4(cs, sp, metric, threshold=0.5)
            if not np.isnan(a) and not np.isnan(p):
                bd.append(sd.mean()); bm.append(sp.mean())
                ba.append(a); bp.append(p)

    if len(bd) >= 4:
        res = np.array(ba) - np.array(bp)
        fr = fit_best_curve(np.array(bd), np.array(bm), res, lam=VBIAS_BETA_LAM)
    else:
        fr = {'params': None}

    # Predict each test subset
    results = []
    for sub_name, sub_idx in idx_map.items():
        sub_y = test_y[sub_idx]
        sub_p = test_p[sub_idx]
        sub_d = test_d[sub_idx]
        actual = safe_metric(metric, sub_y, sub_p)
        if np.isnan(actual):
            continue
        c_s2dd = apply_calibration(cm_s2dd, sub_p)
        s2dd_base = pape_eq4(c_s2dd, sub_p, metric, threshold=0.5)
        if fr['params'] is not None:
            corr = float(predict_best_curve(
                fr, np.array([sub_d.mean()]), np.array([sub_p.mean()]))[0])
        else:
            corr = 0.0
        s2dd_pred = float(np.clip(s2dd_base + corr, 0, 1))
        results.append((sub_name, s2dd_pred, actual))
    return results


# ── Baseline prediction (PAPE + M-CBPE) ─────────────────────────────
def baseline_predict_subsets(cal_y, cal_p, cal_lev,
                             test_y, test_p, test_lev,
                             idx_map, metric):
    """PAPE + M-CBPE prediction using naive Lev distances.
    Returns list of (sub_name, pape_pred, mcbpe_pred, actual)."""
    valid_mask = np.zeros(len(test_y), dtype=bool)
    for idx in idx_map.values():
        valid_mask[idx] = True
    vt_p = test_p[valid_mask]
    vt_lev = test_lev[valid_mask]

    # PAPE DRE
    w_pape, _, _ = estimate_importance_weights(
        np.stack([cal_lev, cal_p], axis=1),
        np.stack([vt_lev, vt_p], axis=1))
    cm_pape = fit_weighted_calibration(cal_p, cal_y, w_pape)

    # M-CBPE DRE
    mcbpe_w, _ = estimate_density_ratios(
        np.stack([cal_lev, cal_p], axis=1),
        np.stack([vt_lev, vt_p], axis=1))
    mcbpe_cal = fit_weighted_calibrator(cal_p, cal_y, mcbpe_w)

    results = []
    for sub_name, sub_idx in idx_map.items():
        sub_y = test_y[sub_idx]
        sub_p = test_p[sub_idx]
        actual = safe_metric(metric, sub_y, sub_p)
        if np.isnan(actual):
            continue
        c_pape = apply_calibration(cm_pape, sub_p)
        pape_pred = pape_eq4(c_pape, sub_p, metric, threshold=0.5)
        mcbpe_c = calibrate_predictions(mcbpe_cal, sub_p)
        mcbpe_pred = estimate_metric_from_calibrated(mcbpe_c, metric)
        results.append((sub_name, pape_pred, mcbpe_pred, actual))
    return results


# ══════════════════════════════════════════════════════════════════════
#  TCR: Per-Epitope Comparison
# ══════════════════════════════════════════════════════════════════════
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
CAL_SETS = ['v3_combined', 'v4_combined']
MIN_EP = 128

# TCR params
TCR_CHAIN_COLS = ['peptide', 'CDR3a', 'CDR3b']
TCR_k, TCR_b, TCR_K = 0.1, 0.1, 50
TCR_TRAIN_PATH = os.path.join(INPUT_DIR, 'Data/tcr_seq/proc_files/tcr_ml_v4/train_data.csv')
# Column map from raw train columns to standard names
TCR_TRAIN_RENAME = {'epitope': 'peptide', 'cdr3_a': 'CDR3a', 'cdr3_b': 'CDR3b'}


def load_tcr_ct(model, ts):
    pp = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                      f'{ts}_predictions_with_label.csv')
    dp_sc = os.path.join(TCR_CACHE, f'{model}_ct_{ts}_dist.npy')
    dp_un = os.path.join(TCR_CACHE, f'{model}_ct_{ts}_uniform_dist.npy')
    if not os.path.exists(pp) or not os.path.exists(dp_sc):
        return None
    te = pd.read_csv(pp)
    d_sc = np.load(dp_sc)
    d_un = np.load(dp_un) if os.path.exists(dp_un) else None
    n = min(len(d_sc), len(te))
    lc = 'binder' if 'binder' in te.columns else 'y_true'
    pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
    pep_col = 'peptide' if 'peptide' in te.columns else 'Epitope'
    cdr3a_col = next((c for c in ['CDR3a', 'CDR3A'] if c in te.columns), None)
    cdr3b_col = next((c for c in ['CDR3b', 'CDR3B'] if c in te.columns), None)
    result = {
        'label': te[lc].values[:n].astype(int),
        'pred': te[pc].values[:n].astype(float),
        'dist_sc': d_sc[:n].astype(float),
        'epitope': te[pep_col].values[:n],
    }
    if d_un is not None:
        result['dist_un'] = d_un[:n].astype(float)
    if cdr3a_col:
        result['cdr3a'] = te[cdr3a_col].values[:n].astype(str)
    if cdr3b_col:
        result['cdr3b'] = te[cdr3b_col].values[:n].astype(str)
    return result


def load_tcr_cv(model, fold):
    test = pd.read_csv(os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}',
                                     'test_predictions_with_label.csv'))
    try:
        val = pd.read_csv(os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}',
                                        'val_predictions_with_label.csv'))
        df = pd.concat([test, val], ignore_index=True)
    except Exception as _e_val:
        import sys as _s_audit
        print(f"  ⚠ FALLBACK [audit_uniform_vs_sigmaC]: model={model} fold={fold} val_predictions missing ({_e_val}); using test only", file=_s_audit.stderr, flush=True)
        df = test
    lc = 'binder' if 'binder' in df.columns else 'y_true'
    pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
    for suffix in ['_combined_dist', '_dist']:
        dp = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dp):
            d_sc = np.load(dp)
            break
    else:
        return None
    n = min(len(d_sc), len(df))
    pep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
    cdr3a_col = next((c for c in ['CDR3a', 'CDR3A'] if c in df.columns), None)
    cdr3b_col = next((c for c in ['CDR3b', 'CDR3B'] if c in df.columns), None)
    result = {
        'label': df[lc].values[:n].astype(int),
        'pred': df[pc].values[:n].astype(float),
        'dist_sc': d_sc[:n].astype(float),
        'epitope': df[pep_col].values[:n],
    }
    if cdr3a_col:
        result['cdr3a'] = df[cdr3a_col].values[:n].astype(str)
    if cdr3b_col:
        result['cdr3b'] = df[cdr3b_col].values[:n].astype(str)
    return result


def compute_tcr_cv_uniform_distances(model, fold):
    """Compute uniform+znorm_sum distances for TCR CV fold inline."""
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    train_path = os.path.join(fold_dir, 'train.csv')
    if not os.path.exists(train_path):
        return None
    train_df = pd.read_csv(train_path)
    # Standardize column names
    renames = {}
    for std, alts in [('peptide', ['Epitope', 'epitope']),
                       ('CDR3a', ['CDR3A', 'cdr3_a', 'cdr3a']),
                       ('CDR3b', ['CDR3B', 'cdr3_b', 'cdr3b'])]:
        if std not in train_df.columns:
            for alt in alts:
                if alt in train_df.columns:
                    renames[alt] = std
                    break
    if renames:
        train_df = train_df.rename(columns=renames)
    if not all(c in train_df.columns for c in TCR_CHAIN_COLS):
        return None

    # Load test data
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    test_df = pd.read_csv(test_path)
    try:
        val_path = os.path.join(fold_dir, 'val_predictions_with_label.csv')
        val_df = pd.read_csv(val_path)
        test_df = pd.concat([test_df, val_df], ignore_index=True)
    except Exception as _e_val2:
        import sys as _s_audit2
        print(f"  ⚠ FALLBACK [audit_uniform_vs_sigmaC test+val]: val missing for {fold_dir} ({_e_val2}); using test only", file=_s_audit2.stderr, flush=True)
    for std, alts in [('peptide', ['Epitope', 'epitope']),
                       ('CDR3a', ['CDR3A', 'cdr3_a', 'cdr3a']),
                       ('CDR3b', ['CDR3B', 'cdr3_b', 'cdr3b'])]:
        if std not in test_df.columns:
            for alt in alts:
                if alt in test_df.columns:
                    test_df = test_df.rename(columns={alt: std})
                    break
    if not all(c in test_df.columns for c in TCR_CHAIN_COLS):
        return None

    weights_un, _ = compute_chain_weights(
        train_df, TCR_CHAIN_COLS, TCR_k, TCR_b, TCR_K, formula='uniform')
    dists = compute_combine_first_distances(
        test_df, train_df, TCR_CHAIN_COLS, weights_un,
        TCR_k, TCR_b, TCR_K, combine_method='znorm_sum')
    return dists


def split_ep(eps, min_ep=MIN_EP):
    counts = pd.Series(eps).value_counts()
    valid = counts[counts >= min_ep].index
    return {ep[:30]: np.where(eps == ep)[0] for ep in valid}


def run_tcr():
    """Run TCR per-epitope comparison: PAPE, M-CBPE, S2DD(sigma_C), S2DD(uniform)."""
    print("=" * 70)
    print("TCR PER-EPITOPE: sigma_C vs uniform S2DD")
    print("=" * 70)
    all_records = []

    for mi, model in enumerate(TCR_MODELS):
        for metric in ['aucroc', 'ap']:
            if metric == 'aucroc':
                print(f"[{mi+1}/{len(TCR_MODELS)}] {model}...", flush=True)

            # ── CT ──
            ct = {ts: load_tcr_ct(model, ts) for ts in CT_SETS}
            ct = {k: v for k, v in ct.items() if v is not None}
            cal_keys = [s for s in CAL_SETS if s in ct]
            test_keys = [s for s in CT_SETS if s in ct and s not in CAL_SETS]

            if cal_keys and test_keys:
                # Check if all test sets have uniform distances
                has_uniform = all('dist_un' in ct[s] for s in cal_keys + test_keys)

                if has_uniform:
                    cal_y = np.concatenate([ct[s]['label'] for s in cal_keys])
                    cal_p = np.concatenate([ct[s]['pred'] for s in cal_keys])
                    cal_d_sc = np.concatenate([ct[s]['dist_sc'] for s in cal_keys])
                    cal_d_un = np.concatenate([ct[s]['dist_un'] for s in cal_keys])
                    cal_ep = np.concatenate([ct[s]['epitope'] for s in cal_keys])
                    test_y = np.concatenate([ct[s]['label'] for s in test_keys])
                    test_p = np.concatenate([ct[s]['pred'] for s in test_keys])
                    test_d_sc = np.concatenate([ct[s]['dist_sc'] for s in test_keys])
                    test_d_un = np.concatenate([ct[s]['dist_un'] for s in test_keys])
                    test_ep = np.concatenate([ct[s]['epitope'] for s in test_keys])
                    test_src = np.concatenate([np.full(len(ct[s]['label']), s) for s in test_keys])

                    # 3-chain naive Lev data
                    chain_keys = ['epitope', 'cdr3a', 'cdr3b']
                    cal_chains = {'epitope': cal_ep}
                    test_chains = {'epitope': test_ep}
                    for ch, raw in [('cdr3a', 'cdr3a'), ('cdr3b', 'cdr3b')]:
                        cal_ch = [ct[s].get(raw) for s in cal_keys if raw in ct[s]]
                        test_ch = [ct[s].get(raw) for s in test_keys if raw in ct[s]]
                        if len(cal_ch) == len(cal_keys) and len(test_ch) == len(test_keys):
                            cal_chains[ch] = np.concatenate(cal_ch)
                            test_chains[ch] = np.concatenate(test_ch)

                    idx_map = split_ep(test_ep)
                    if idx_map:
                        # Baselines (PAPE + M-CBPE)
                        cal_lev = naive_lev_dist_3chain(cal_chains, cal_chains, chain_keys)
                        test_lev = naive_lev_dist_3chain(test_chains, cal_chains, chain_keys)
                        bl_results = baseline_predict_subsets(
                            cal_y, cal_p, cal_lev,
                            test_y, test_p, test_lev,
                            idx_map, metric)

                        # S2DD(sigma_C)
                        sc_results = s2dd_predict_subsets(
                            cal_y, cal_p, cal_d_sc, cal_ep,
                            test_y, test_p, test_d_sc, test_ep,
                            idx_map, metric)

                        # S2DD(uniform)
                        un_results = s2dd_predict_subsets(
                            cal_y, cal_p, cal_d_un, cal_ep,
                            test_y, test_p, test_d_un, test_ep,
                            idx_map, metric)

                        # Merge results
                        bl_dict = {r[0]: r for r in bl_results}
                        sc_dict = {r[0]: r for r in sc_results}
                        un_dict = {r[0]: r for r in un_results}
                        for sub_name in idx_map:
                            if sub_name not in bl_dict or sub_name not in sc_dict or sub_name not in un_dict:
                                continue
                            _, pape, mcbpe, actual = bl_dict[sub_name]
                            _, s2dd_sc, _ = sc_dict[sub_name]
                            _, s2dd_un, _ = un_dict[sub_name]
                            sub_idx = idx_map[sub_name]
                            src_mode = pd.Series(test_src[sub_idx]).mode()[0]
                            seen = 'seen' if src_mode == 'seen_test' else 'unseen'
                            all_records.append({
                                'domain': 'TCR', 'model': model, 'split': 'CT',
                                'subset': sub_name, 'metric': metric,
                                'actual': actual, 'pape': pape, 'mcbpe': mcbpe,
                                's2dd_sc': s2dd_sc, 's2dd_un': s2dd_un,
                                'seen': seen, 'n': len(sub_idx),
                            })

            # ── CV ──
            for fold in range(5):
                data = load_tcr_cv(model, fold)
                if data is None:
                    continue
                ref = load_tcr_cv('nettcr', fold)
                if ref and len(ref['dist_sc']) == len(data['label']):
                    si = np.argsort(ref['dist_sc'])
                else:
                    si = np.argsort(data['dist_sc'])
                cal_idx, test_idx = si[::2], si[1::2]

                idx_map = split_ep(data['epitope'][test_idx])
                if not idx_map:
                    continue

                # Compute uniform distances for this fold
                cache_path = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_uniform_dist.npy')
                if os.path.exists(cache_path):
                    d_un_full = np.load(cache_path)
                else:
                    d_un_full = compute_tcr_cv_uniform_distances(model, fold)
                    if d_un_full is not None:
                        np.save(cache_path, d_un_full)
                        print(f"    Cached CV uniform: {cache_path}")

                if d_un_full is None or len(d_un_full) != len(data['label']):
                    continue

                cv_cal_y = data['label'][cal_idx]
                cv_cal_p = data['pred'][cal_idx]
                cv_cal_d_sc = data['dist_sc'][cal_idx]
                cv_cal_d_un = d_un_full[cal_idx]
                cv_cal_ep = data['epitope'][cal_idx]
                cv_test_y = data['label'][test_idx]
                cv_test_p = data['pred'][test_idx]
                cv_test_d_sc = data['dist_sc'][test_idx]
                cv_test_d_un = d_un_full[test_idx]
                cv_test_ep = data['epitope'][test_idx]

                # 3-chain naive Lev
                cv_chain_keys = ['epitope']
                cv_cal_chains = {'epitope': cv_cal_ep}
                cv_test_chains = {'epitope': cv_test_ep}
                for ch in ['cdr3a', 'cdr3b']:
                    if ch in data:
                        cv_cal_chains[ch] = data[ch][cal_idx]
                        cv_test_chains[ch] = data[ch][test_idx]
                        cv_chain_keys.append(ch)

                cv_cal_lev = naive_lev_dist_3chain(cv_cal_chains, cv_cal_chains, cv_chain_keys)
                cv_test_lev = naive_lev_dist_3chain(cv_test_chains, cv_cal_chains, cv_chain_keys)

                bl_results = baseline_predict_subsets(
                    cv_cal_y, cv_cal_p, cv_cal_lev,
                    cv_test_y, cv_test_p, cv_test_lev,
                    idx_map, metric)
                sc_results = s2dd_predict_subsets(
                    cv_cal_y, cv_cal_p, cv_cal_d_sc, cv_cal_ep,
                    cv_test_y, cv_test_p, cv_test_d_sc, cv_test_ep,
                    idx_map, metric)
                un_results = s2dd_predict_subsets(
                    cv_cal_y, cv_cal_p, cv_cal_d_un, cv_cal_ep,
                    cv_test_y, cv_test_p, cv_test_d_un, cv_test_ep,
                    idx_map, metric)

                bl_dict = {r[0]: r for r in bl_results}
                sc_dict = {r[0]: r for r in sc_results}
                un_dict = {r[0]: r for r in un_results}
                for sub_name in idx_map:
                    if sub_name not in bl_dict or sub_name not in sc_dict or sub_name not in un_dict:
                        continue
                    _, pape, mcbpe, actual = bl_dict[sub_name]
                    _, s2dd_sc, _ = sc_dict[sub_name]
                    _, s2dd_un, _ = un_dict[sub_name]
                    all_records.append({
                        'domain': 'TCR', 'model': model, 'split': 'CV',
                        'subset': sub_name, 'metric': metric,
                        'actual': actual, 'pape': pape, 'mcbpe': mcbpe,
                        's2dd_sc': s2dd_sc, 's2dd_un': s2dd_un,
                        'seen': 'cv', 'n': len(idx_map[sub_name]),
                    })

    return all_records


# ══════════════════════════════════════════════════════════════════════
#  BCR: Per-Antigen Comparison
# ══════════════════════════════════════════════════════════════════════
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
BCR_k, BCR_b, BCR_K = 0.1, 0.03, 30
MIN_SAMPLES = 30

# BCR fold4 training data (same for all models)
BCR_TRAIN_PATH = os.path.join(RESULTS, 'xbcr/combined_bind_ab_cv/fold4/train.csv')


def vhash(seq):
    return hashlib.md5(str(seq).encode()).hexdigest()[:12]


def compute_bcr_uniform_distances(test_df, train_df):
    """Compute uniform+znorm_sum distances for BCR."""
    weights_un, _ = compute_chain_weights(
        train_df, BCR_CHAIN_COLS, BCR_k, BCR_b, BCR_K, formula='uniform')
    dists = compute_combine_first_distances(
        test_df, train_df, BCR_CHAIN_COLS, weights_un,
        BCR_k, BCR_b, BCR_K, combine_method='znorm_sum')
    return dists


def run_bcr():
    """Run BCR per-antigen comparison: PAPE, M-CBPE, S2DD(sigma_C), S2DD(uniform)."""
    print("=" * 70)
    print("BCR PER-ANTIGEN: sigma_C vs uniform S2DD")
    print("=" * 70)
    all_records = []

    # Load BCR training data once
    if not os.path.exists(BCR_TRAIN_PATH):
        print(f"ERROR: BCR train not found: {BCR_TRAIN_PATH}")
        return []
    bcr_train = pd.read_csv(BCR_TRAIN_PATH)
    print(f"BCR train: {len(bcr_train)} samples")

    # Precompute uniform distances for all BCR CT data
    bcr_uniform_cache = {}

    for mi, model in enumerate(BCR_MODELS):
        print(f"[{mi+1}/{len(BCR_MODELS)}] {model}...", flush=True)

        # Load CT data
        cal_path = os.path.join(BCR_FOLD4CAL, model, 'cal_predictions.csv')
        if not os.path.exists(cal_path):
            continue
        cal = pd.read_csv(cal_path)
        cal['source'] = 'fold4_test'
        parts = [cal]
        for ts in ['A1-A11', 'unseen', 'flu']:
            fp = os.path.join(BCR_FOLD4CAL, model, f'{ts}_predictions.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            df['source'] = ts
            if 'data_source' not in df.columns:
                df['data_source'] = 'flu' if ts == 'flu' else 'sars'
            parts.append(df)
        pooled = pd.concat(parts, ignore_index=True)

        # Compute uniform distances for this model's pooled CT data
        cache_key = f'bcr_ct_{model}'
        cache_path = os.path.join(RESULTS, 'fig2_cache', f'bcr_ct_{model}_uniform_dist.npy')
        if os.path.exists(cache_path):
            d_un_all = np.load(cache_path)
            print(f"  Loaded cached uniform: {len(d_un_all)}")
        else:
            print(f"  Computing uniform distances ({len(pooled)} samples)...", flush=True)
            t1 = time.time()
            d_un_all = compute_bcr_uniform_distances(pooled, bcr_train)
            np.save(cache_path, d_un_all)
            print(f"  Done ({time.time()-t1:.0f}s), cached to {cache_path}")
        pooled['dist_un'] = d_un_all

        for metric in ['aucroc', 'ap']:
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
                    if len(test_sub) < 10:
                        continue
                    ty = test_sub['rbd'].values.astype(int)
                    if ty.sum() == 0 or ty.sum() == len(ty):
                        continue
                    cy = cal_sub['rbd'].values.astype(int)
                    if cy.sum() < 3 or (len(cy) - cy.sum()) < 3:
                        continue

                    cp = cal_sub['pred_prob'].values.astype(float)
                    tp = test_sub['pred_prob'].values.astype(float)
                    cd_sc = cal_sub['distance'].values.astype(float)
                    td_sc = test_sub['distance'].values.astype(float)
                    cd_un = cal_sub['dist_un'].values.astype(float)
                    td_un = test_sub['dist_un'].values.astype(float)
                    cvs = cal_sub['variant_seq'].values
                    tvs = test_sub['variant_seq'].values

                    idx_map = {vhash(held_v): np.arange(len(test_sub))}

                    # 3-chain naive Lev
                    chain_keys = ['heavy', 'light', 'variant_seq']
                    cal_chains = {'variant_seq': cvs}
                    test_chains = {'variant_seq': tvs}
                    for ch, col in [('heavy', 'Heavy'), ('light', 'Light')]:
                        if col in cal_sub.columns and col in test_sub.columns:
                            cal_chains[ch] = cal_sub[col].values.astype(str)
                            test_chains[ch] = test_sub[col].values.astype(str)

                    cal_lev = naive_lev_dist_3chain(cal_chains, cal_chains, chain_keys)
                    test_lev = naive_lev_dist_3chain(test_chains, cal_chains, chain_keys)

                    bl_results = baseline_predict_subsets(
                        cy, cp, cal_lev, ty, tp, test_lev, idx_map, metric)
                    sc_results = s2dd_predict_subsets(
                        cy, cp, cd_sc, cvs, ty, tp, td_sc, tvs, idx_map, metric)
                    un_results = s2dd_predict_subsets(
                        cy, cp, cd_un, cvs, ty, tp, td_un, tvs, idx_map, metric)

                    vh = vhash(held_v)
                    if vh in {r[0] for r in bl_results} and \
                       vh in {r[0] for r in sc_results} and \
                       vh in {r[0] for r in un_results}:
                        bl = [r for r in bl_results if r[0] == vh][0]
                        sc = [r for r in sc_results if r[0] == vh][0]
                        un = [r for r in un_results if r[0] == vh][0]
                        all_records.append({
                            'domain': 'BCR', 'model': model, 'split': 'CT',
                            'subset': vh, 'metric': metric,
                            'actual': bl[3], 'pape': bl[1], 'mcbpe': bl[2],
                            's2dd_sc': sc[1], 's2dd_un': un[1],
                            'seen': domain, 'n': len(test_sub),
                        })

        # CV: halfsplit
        for fold in range(5):
            test_path = os.path.join(
                RESULTS, 'xbcr' if model == 'xbcr' else model,
                'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
            if not os.path.exists(test_path):
                continue
            te = pd.read_csv(test_path)
            if 'pred_prob' not in te.columns or 'distance' not in te.columns:
                continue

            # Compute uniform distances for CV fold
            cv_cache = os.path.join(RESULTS, 'fig2_cache',
                                     f'bcr_cv_{model}_fold{fold}_uniform_dist.npy')
            if os.path.exists(cv_cache):
                d_un_cv = np.load(cv_cache)
            else:
                cv_train_path = os.path.join(
                    RESULTS, 'xbcr' if model == 'xbcr' else model,
                    'combined_bind_ab_cv', f'fold{fold}', 'train.csv')
                if not os.path.exists(cv_train_path):
                    continue
                cv_train = pd.read_csv(cv_train_path)
                if not all(c in cv_train.columns for c in BCR_CHAIN_COLS):
                    continue
                d_un_cv = compute_bcr_uniform_distances(te, cv_train)
                np.save(cv_cache, d_un_cv)

            if len(d_un_cv) != len(te):
                continue

            y = te['rbd'].values.astype(int)
            p = te['pred_prob'].values.astype(float)
            d_sc = te['distance'].values.astype(float)
            vs = te['variant_seq'].values

            si = np.argsort(d_sc)
            cal_idx, test_idx = si[::2], si[1::2]

            test_vs = vs[test_idx]
            unique_vs = pd.Series(test_vs).value_counts()
            valid_vs = unique_vs[unique_vs >= MIN_SAMPLES].index
            if len(valid_vs) < 4:
                continue

            ag_map = {vhash(v): np.where(test_vs == v)[0] for v in valid_vs}

            # 3-chain naive Lev
            chain_keys = ['heavy', 'light', 'variant_seq']
            cal_chains = {'variant_seq': vs[cal_idx]}
            test_chains = {'variant_seq': test_vs}
            for ch, col in [('heavy', 'Heavy'), ('light', 'Light')]:
                if col in te.columns:
                    cal_chains[ch] = te[col].values.astype(str)[cal_idx]
                    test_chains[ch] = te[col].values.astype(str)[test_idx]

            cal_lev = naive_lev_dist_3chain(cal_chains, cal_chains, chain_keys)
            test_lev = naive_lev_dist_3chain(test_chains, cal_chains, chain_keys)

            for metric in ['aucroc', 'ap']:
                bl_results = baseline_predict_subsets(
                    y[cal_idx], p[cal_idx], cal_lev,
                    y[test_idx], p[test_idx], test_lev,
                    ag_map, metric)
                sc_results = s2dd_predict_subsets(
                    y[cal_idx], p[cal_idx], d_sc[cal_idx], vs[cal_idx],
                    y[test_idx], p[test_idx], d_sc[test_idx], vs[test_idx],
                    ag_map, metric)
                un_results = s2dd_predict_subsets(
                    y[cal_idx], p[cal_idx], d_un_cv[cal_idx], vs[cal_idx],
                    y[test_idx], p[test_idx], d_un_cv[test_idx], vs[test_idx],
                    ag_map, metric)

                bl_dict = {r[0]: r for r in bl_results}
                sc_dict = {r[0]: r for r in sc_results}
                un_dict = {r[0]: r for r in un_results}
                for sub_name in ag_map:
                    if sub_name not in bl_dict or sub_name not in sc_dict or sub_name not in un_dict:
                        continue
                    _, pape, mcbpe, actual = bl_dict[sub_name]
                    _, s2dd_sc, _ = sc_dict[sub_name]
                    _, s2dd_un, _ = un_dict[sub_name]
                    all_records.append({
                        'domain': 'BCR', 'model': model, 'split': 'CV',
                        'subset': sub_name, 'metric': metric,
                        'actual': actual, 'pape': pape, 'mcbpe': mcbpe,
                        's2dd_sc': s2dd_sc, 's2dd_un': s2dd_un,
                        'seen': 'cv', 'n': len(ag_map[sub_name]),
                    })

    return all_records


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def print_results(df, domain):
    methods = ['pape', 'mcbpe', 's2dd_sc', 's2dd_un']
    names = {'pape': 'PAPE', 'mcbpe': 'M-CBPE',
             's2dd_sc': 'S2DD(σC)', 's2dd_un': 'S2DD(uni)'}

    for met in ['aucroc', 'ap']:
        for split in ['CT', 'CV', 'CT+CV']:
            sub = df[df['metric'] == met]
            if split != 'CT+CV':
                sub = sub[sub['split'] == split]
            sub = sub.dropna(subset=['actual'])
            if len(sub) < 3:
                continue
            print(f"  {domain} {met} {split} (n={len(sub)}):")
            print(f"    {'Method':<15} {'r':>7} {'MAE':>7}")
            for m in methods:
                valid = sub.dropna(subset=[m])
                if len(valid) < 3:
                    continue
                r, _ = pearsonr(valid[m], valid['actual'])
                mae = np.abs(valid[m] - valid['actual']).mean()
                winner = ' ◄' if m in ('s2dd_sc', 's2dd_un') and \
                    r == max(pearsonr(valid['s2dd_sc'].dropna(), valid['actual'])[0],
                             pearsonr(valid['s2dd_un'].dropna(), valid['actual'])[0]) else ''
                print(f"    {names[m]:<15} {r:>7.3f} {mae:>7.3f}{winner}")
            # Head-to-head sigma_C vs uniform
            both = sub.dropna(subset=['s2dd_sc', 's2dd_un'])
            if len(both) >= 3:
                r_sc, _ = pearsonr(both['s2dd_sc'], both['actual'])
                r_un, _ = pearsonr(both['s2dd_un'], both['actual'])
                delta = r_un - r_sc
                winner = 'uniform' if delta > 0.001 else ('sigma_C' if delta < -0.001 else 'tie')
                print(f"    → Δ(uniform−σC) = {delta:+.3f} → {winner}")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='both', choices=['tcr', 'bcr', 'both'])
    args = parser.parse_args()

    t0 = time.time()
    all_records = []

    if args.domain in ('tcr', 'both'):
        tcr_records = run_tcr()
        all_records.extend(tcr_records)

    if args.domain in ('bcr', 'both'):
        bcr_records = run_bcr()
        all_records.extend(bcr_records)

    df = pd.DataFrame(all_records)
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {len(df)} records in {elapsed/60:.1f} min")
    print(f"{'=' * 70}\n")

    if 'TCR' in df['domain'].values:
        print_results(df[df['domain'] == 'TCR'], 'TCR')
    if 'BCR' in df['domain'].values:
        print_results(df[df['domain'] == 'BCR'], 'BCR')

    # Per-model breakdown
    print("--- Per-model sigma_C vs uniform r (CT+CV, AUROC) ---")
    for domain in df['domain'].unique():
        dom = df[(df['domain'] == domain) & (df['metric'] == 'aucroc')]
        for model in sorted(dom['model'].unique()):
            sub = dom[dom['model'] == model].dropna(subset=['s2dd_sc', 's2dd_un', 'actual'])
            if len(sub) < 3:
                continue
            r_sc, _ = pearsonr(sub['s2dd_sc'], sub['actual'])
            r_un, _ = pearsonr(sub['s2dd_un'], sub['actual'])
            delta = r_un - r_sc
            w = '◄uni' if delta > 0.01 else ('◄σC' if delta < -0.01 else '≈')
            print(f"  {domain} {model:<12} σC={r_sc:.3f} uni={r_un:.3f} Δ={delta:+.3f} {w}")
    print()

    # Save
    out_path = os.path.join(SCRIPT_DIR, '..', 'audit_uniform_vs_sigmaC_results.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
