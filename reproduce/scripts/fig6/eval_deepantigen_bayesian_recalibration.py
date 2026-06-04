#!/usr/bin/env python3
"""Bayesian recalibration of deepAntigen predictions on ImmuneCODE using S2DD.

Methodology (matches our manuscript Fig 6 pipeline):
  1. Fit PPV/NPV curves on zero-shot (1,714 pairs, labeled) binned by S2DD distance
  2. Apply bin-level PPV/NPV to ImmuneCODE predictions (with S2DD distances)
  3. Compare AUROC/AP/F1 before vs after recalibration
  4. At θ=0.5 threshold: count TP/FP/FN before vs after
  5. Extrapolate to full 1.1M ImmuneCODE: estimated additional TP recovered

Key claim: "Our framework, applied to deepAntigen predictions on 1.1M SARS-CoV-2
TCR-epitope pairs, recovers N more true positive immunogenic pairs while
reducing M false positives, improving AUROC from 0.71 → X."
"""

import os, sys, warnings
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.general_evaluator import safe_metric

# Import the PROVEN unified calibration functions from fig6 by loading the module
# directly (fig6_practical_usage is a script, not a module — load via spec).
# However, fig6 runs side-effect code at import. To avoid that, we extract just
# the two key functions by re-defining them here from a verified AST diff against
# fig6 (verified 2026-04-16: identical body). Source of truth: fig6_practical_usage.py
# lines 113-234 (fit_ppv_npv_from_cal_sets + calibrate_with_curves).
from calipper.combine_first_helpers import (
    fit_ridge_vbias, CALIBRATION_LAM, CALIBRATION_THRESHOLD,
)

RESULTS_DIR = Path(OUTPUT_DIR) / 'deepantigen_retrospective' / 'bayesian_recalibration'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Read existing distance CSVs from OUTPUT_DIR (where phase 3 wrote them)
# Fall back to INPUT_DIR (canonical cached) if not in OUTPUT_DIR.
_OUTPUT_S2DD = Path(OUTPUT_DIR) / 'deepantigen_retrospective' / 's2dd_degradation'
_INPUT_S2DD = Path(INPUT_DIR) / 'results' / 'deepantigen_retrospective' / 's2dd_degradation'
S2DD_DIR = _OUTPUT_S2DD if (_OUTPUT_S2DD / 'immunecode_with_distances.csv').exists() else _INPUT_S2DD

# Unified constants from library — MUST match fig6 pipeline
N_BINS = 8                            # N_SUB=8 bins per fig6 + guidance docs
THRESHOLD = CALIBRATION_THRESHOLD      # 0.5, model-independent, from library
# lam = CALIBRATION_LAM = 0.0 (used inside fit_ridge_vbias)

# ── Unified calibration primitives (EXACT match to fig6_practical_usage.py) ─
_logit = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) /
                           (1 - np.clip(p, 1e-12, 1 - 1e-12)))
_sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_ppv_npv_from_cal_sets(cal_data, n_sub=N_BINS, min_samples=30, threshold=THRESHOLD):
    """EXACT COPY of fig6_practical_usage.fit_ppv_npv_from_cal_sets.

    cal_data: dict[ts_name -> (y, p, dists)] — calibration sources.
    Returns (ppv_params, npv_params, p_pos, p_neg).
    """
    cal_d, cal_mp, cal_ppv, cal_npv = [], [], [], []
    cal_preds = []
    for ts_name, (y_ts, p_ts, d_ts) in cal_data.items():
        si = np.argsort(d_ts)
        bs = len(si) // n_sub
        if bs < min_samples:
            continue
        cal_preds.extend(p_ts.tolist())
        for i in range(n_sub):
            s = i * bs
            e = len(si) if i == n_sub - 1 else (i + 1) * bs
            idx = si[s:e]
            yi, pi = y_ts[idx], p_ts[idx]
            pp = pi >= threshold
            tp = ((pp) & (yi == 1)).sum()
            fp = ((pp) & (yi == 0)).sum()
            tn = ((~pp) & (yi == 0)).sum()
            fn = ((~pp) & (yi == 1)).sum()
            cal_d.append(d_ts[idx].mean())
            cal_mp.append(pi.mean())
            cal_ppv.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
            cal_npv.append(tn / (tn + fn) if tn + fn > 0 else np.nan)

    cal_d = np.array(cal_d)
    cal_mp = np.array(cal_mp)
    cal_ppv = np.array(cal_ppv)
    cal_npv = np.array(cal_npv)
    cal_preds = np.array(cal_preds)

    ppv_params = npv_params = None
    for vals, name in [(cal_ppv, 'ppv'), (cal_npv, 'npv')]:
        v = ~np.isnan(vals)
        if v.sum() >= 4:
            params = fit_ridge_vbias(cal_d[v], cal_mp[v], vals[v], lam=CALIBRATION_LAM)
            if name == 'ppv':
                ppv_params = params
            else:
                npv_params = params

    pos_p = cal_preds[cal_preds >= threshold]
    neg_p = cal_preds[cal_preds < threshold]
    p_pos = np.quantile(pos_p, 0.25) if len(pos_p) > 0 else 0.75
    p_neg = np.quantile(neg_p, 0.75) if len(neg_p) > 0 else 0.25
    return ppv_params, npv_params, p_pos, p_neg, cal_d, cal_ppv, cal_npv


def calibrate_with_curves(y, p, d, ppv_params, npv_params, p_pos, p_neg,
                          n_bins=N_BINS):
    """EXACT COPY of fig6_practical_usage.calibrate_with_curves."""
    prev = y.mean()
    si = np.argsort(d)
    bs = max(len(si) // n_bins, 1)
    mp_per_sample = np.zeros(len(d))
    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        mp_per_sample[idx] = p[idx].mean()

    if ppv_params is not None:
        a, bx, c, beta = ppv_params
        ppv_ps = np.clip(a * np.exp(-bx * d) + c + beta * mp_per_sample, 0.01, 0.99)
    else:
        ppv_ps = np.full(len(d), 0.5)
    if npv_params is not None:
        a, bx, c, beta = npv_params
        npv_ps = np.clip(a * np.exp(-bx * d) + c + beta * mp_per_sample, 0.01, 0.99)
    else:
        npv_ps = np.full(len(d), 0.5)

    logit_pp = _logit(p_pos)
    logit_pn = _logit(p_neg)
    denom = logit_pp - logit_pn
    b_raw = (_logit(ppv_ps) - _logit(1.0 - npv_ps)) / np.where(
        np.abs(denom) > 1e-12, denom, 1e-12)
    a_raw = _logit(ppv_ps) - _logit(prev) - b_raw * logit_pp

    w_ps = np.clip(ppv_ps + npv_ps - 1.0, 0.0, 1.0)
    w_ps = np.clip(w_ps, 0.1, 1.0)
    b_eff = w_ps * b_raw
    a_eff = w_ps * a_raw

    cal = _sigmoid_fn(_logit(prev) + a_eff + b_eff * _logit(p))
    cal = np.where(np.isfinite(cal), cal, 0.5)
    return cal, ppv_ps, npv_ps


def compute_immunecode_sw_topk(k=0.1, b=0.1, K=50):
    """Compute S2DD-SW (BLOSUM + topK) distances for ImmuneCODE.

    Mirrors the zero-shot computation in fig_s2dd_vs_sw_comparison.py.
    Uses BLOSUM62 SW similarity, topK=50 mean, expanded via peptide frequency.
    """
    import parasail
    import time

    cache_path = S2DD_DIR / 'immunecode_sw_topk_distances.csv'
    if cache_path.exists():
        print(f"  Loading cached ImmuneCODE S2DD-SW distances")
        return pd.read_csv(cache_path)['distance'].values

    print(f"  Computing S2DD-SW distances for ImmuneCODE (this takes ~2-3 min)...")
    train_path = Path(INPUT_DIR) / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data' / 'train.csv'
    train = pd.read_csv(train_path)
    ic = pd.read_csv(S2DD_DIR / 'immunecode_with_distances.csv')

    train_peps_rows = train['peptide'].values
    unique_train_peps = np.unique(train_peps_rows)
    test_peps = ic['peptide'].values

    # Self-scores
    train_self = {s: parasail.sw_stats(s, s, 10, 1, parasail.blosum62).score
                  for s in unique_train_peps}

    # Unique test peptides → similarity against all unique train peptides
    unique_test_peps = np.unique(test_peps)
    print(f"    Unique test peptides: {len(unique_test_peps)}, "
          f"unique train peptides: {len(unique_train_peps)}")
    t0 = time.time()
    sim_by_test_pep = {}
    for i, q in enumerate(unique_test_peps):
        q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
        if q_self <= 0:
            sim_by_test_pep[q] = np.zeros(len(unique_train_peps))
            continue
        sims = np.array([
            parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score /
            np.sqrt(q_self * train_self[r]) if train_self[r] > 0 else 0.0
            for r in unique_train_peps
        ])
        sim_by_test_pep[q] = sims
        if (i + 1) % 50 == 0 or i == len(unique_test_peps) - 1:
            print(f"    [{i+1:>3}/{len(unique_test_peps)}] {(i+1)/len(unique_test_peps)*100:.0f}% "
                  f"({time.time()-t0:.0f}s)")

    # Expand: each training row has the sim corresponding to its peptide
    pep_to_rows = {p: np.where(train_peps_rows == p)[0] for p in unique_train_peps}
    distances = np.zeros(len(test_peps))
    for i, q in enumerate(test_peps):
        pep_sims = sim_by_test_pep[q]
        row_sims = np.zeros(len(train_peps_rows))
        for pep_idx, p in enumerate(unique_train_peps):
            row_sims[pep_to_rows[p]] = pep_sims[pep_idx]
        topk_sims = np.sort(row_sims)[::-1][:K]
        topk_dists = np.log(k * (1.0 - topk_sims + b))
        distances[i] = topk_dists.mean()

    pd.DataFrame({'distance': distances}).to_csv(cache_path, index=False)
    print(f"    Cached to {cache_path}")
    return distances


def run_recalibration(dist_metric_name, y_zs, p_zs, d_zs, y_ic, p_ic, d_ic):
    """Run full recalibration pipeline for a given distance metric."""
    print(f"\n{'='*70}")
    print(f"Bayesian Recalibration with {dist_metric_name}")
    print('='*70)

    print(f"\nFitting PPV/NPV on zero-shot ({dist_metric_name})...")
    cal_data = {'zero_shot': (y_zs, p_zs, d_zs)}
    ppv_params, npv_params, p_pos, p_neg, cal_d, cal_ppv, cal_npv = fit_ppv_npv_from_cal_sets(
        cal_data, n_sub=N_BINS, min_samples=30, threshold=THRESHOLD)
    print(f"  PPV per bin: {[f'{v:.3f}' for v in cal_ppv]}")
    print(f"  NPV per bin: {[f'{v:.3f}' for v in cal_npv]}")
    print(f"  p_pos={p_pos:.4f}, p_neg={p_neg:.4f}")
    if ppv_params is not None:
        print(f"  PPV curve: a={ppv_params[0]:.3f} b={ppv_params[1]:.3f} "
              f"c={ppv_params[2]:.3f} beta={ppv_params[3]:.3f}")
    if npv_params is not None:
        print(f"  NPV curve: a={npv_params[0]:.3f} b={npv_params[1]:.3f} "
              f"c={npv_params[2]:.3f} beta={npv_params[3]:.3f}")

    print(f"\nApplying recalibration to ImmuneCODE...")
    cal_ic, ppv_ic, npv_ic = calibrate_with_curves(
        y_ic, p_ic, d_ic, ppv_params, npv_params, p_pos, p_neg)

    print(f"\n{'Metric':<10} {'Original':>12} {'Calibrated':>12} {'Δ':>8}")
    print('-'*50)
    results = {'metric': [], 'original': [], 'calibrated': [], 'delta': []}
    for m in ['aucroc', 'ap', 'f1']:
        orig = safe_metric(m, y_ic, p_ic)
        cal = safe_metric(m, y_ic, cal_ic)
        print(f"{m:<10} {orig:>12.4f} {cal:>12.4f} {cal-orig:>+8.4f}")
        results['metric'].append(m)
        results['original'].append(orig)
        results['calibrated'].append(cal)
        results['delta'].append(cal - orig)

    # Confusion at θ
    print(f"\n--- Confusion at θ={THRESHOLD} ---")
    cm = {}
    for name, probs in [('original', p_ic), ('calibrated', cal_ic)]:
        pred = probs >= THRESHOLD
        tp = int(((pred) & (y_ic == 1)).sum())
        fp = int(((pred) & (y_ic == 0)).sum())
        tn = int(((~pred) & (y_ic == 0)).sum())
        fn = int(((~pred) & (y_ic == 1)).sum())
        cm[name] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                    'precision': tp/(tp+fp) if tp+fp else 0,
                    'recall': tp/(tp+fn) if tp+fn else 0}
        print(f"  {name:<12}: TP={tp}, FP={fp}, TN={tn}, FN={fn} | "
              f"precision={cm[name]['precision']:.3f}, recall={cm[name]['recall']:.3f}")

    delta_tp = cm['calibrated']['TP'] - cm['original']['TP']
    delta_fp = cm['calibrated']['FP'] - cm['original']['FP']
    print(f"\n  Δ TP = {delta_tp:+d}, Δ FP = {delta_fp:+d}")

    scale = 1_129_028 / len(y_ic)
    print(f"\n  Extrapolated to 1.1M (scale={scale:.2f}):")
    print(f"    Δ TP = {delta_tp * scale:+.0f}, Δ FP = {delta_fp * scale:+.0f}")

    return {
        'metric_name': dist_metric_name,
        'results': results, 'cm': cm,
        'delta_tp': delta_tp, 'delta_fp': delta_fp,
        'scale_factor': scale,
        'cal_ppv': cal_ppv, 'cal_npv': cal_npv,
        'ppv_params': ppv_params, 'npv_params': npv_params,
        'calibrated_predictions': cal_ic,
    }


def main():
    # ── Load base data ───────────────────────────────────────────────────
    print("Loading data...")
    zs = pd.read_csv(S2DD_DIR / 'zero_shot_with_distances.csv')
    ic = pd.read_csv(S2DD_DIR / 'immunecode_with_distances.csv')
    print(f"  Zero-shot: {len(zs)} pairs (calibration source)")
    print(f"  ImmuneCODE: {len(ic)} pairs (held-out)")

    y_zs = zs['label'].values.astype(int)
    p_zs = zs['prediction'].values.astype(float)
    y_ic = ic['label'].values.astype(int)
    p_ic = ic['prediction'].values.astype(float)

    # ── Distance variants ────────────────────────────────────────────────
    # S2DD-Levenshtein (existing pipeline, sigma_C weights, weighted_max_znorm)
    d_zs_lev = zs['s2dd_distance'].values.astype(float)
    d_ic_lev = ic['s2dd_distance'].values.astype(float)

    # S2DD-BLOSUM (topK=50 mean of BLOSUM62 SW similarity, log-transformed)
    print("\n--- Loading S2DD-BLOSUM distances ---")
    d_zs_sw = pd.read_csv(S2DD_DIR / 'zero_shot_sw_topk_distances.csv')['distance'].values
    d_ic_sw = compute_immunecode_sw_topk()

    # ── Run recalibration for both distance metrics ──────────────────────
    all_results = {}
    for name, d_zs, d_ic in [
        ('S2DD-Levenshtein', d_zs_lev, d_ic_lev),
        ('S2DD-BLOSUM-topK', d_zs_sw, d_ic_sw),
    ]:
        all_results[name] = run_recalibration(name, y_zs, p_zs, d_zs, y_ic, p_ic, d_ic)

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY: S2DD-Levenshtein vs S2DD-BLOSUM for Bayesian Recalibration")
    print("="*80)
    print(f"{'Distance':<22} {'Metric':<8} {'Original':>10} {'Calibrated':>12} {'Δ':>8}")
    print("-"*80)
    for name, res in all_results.items():
        for m, o, c, d in zip(res['results']['metric'], res['results']['original'],
                              res['results']['calibrated'], res['results']['delta']):
            print(f"{name:<22} {m:<8} {o:>10.4f} {c:>12.4f} {d:>+8.4f}")

    print(f"\n{'Distance':<22} {'ΔTP':>8} {'ΔFP':>8} {'Scale×ΔTP':>12} {'Scale×ΔFP':>12}")
    print("-"*80)
    for name, res in all_results.items():
        s = res['scale_factor']
        print(f"{name:<22} {res['delta_tp']:>+8d} {res['delta_fp']:>+8d} "
              f"{res['delta_tp']*s:>+12.0f} {res['delta_fp']*s:>+12.0f}")

    # Save combined CSV
    rows = []
    for name, res in all_results.items():
        for m, o, c, d in zip(res['results']['metric'], res['results']['original'],
                              res['results']['calibrated'], res['results']['delta']):
            rows.append({'distance': name, 'metric': m, 'original': o,
                         'calibrated': c, 'delta': d})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / 'recalibration_comparison_lev_vs_blosum.csv',
                                index=False)

    cm_rows = []
    for name, res in all_results.items():
        for method, cm in res['cm'].items():
            cm_rows.append({'distance': name, 'method': method, **cm})
    pd.DataFrame(cm_rows).to_csv(RESULTS_DIR / 'confusion_lev_vs_blosum.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR}")
    return all_results


if __name__ == '__main__':
    results = main()
