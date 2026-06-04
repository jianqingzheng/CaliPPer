#!/usr/bin/env python3
"""Bayesian recalibration of deepAntigen neoantigen predictions.

Clinical application: improve ranking of 100 candidate neoantigens (5 patients)
using S2DD-based Bayesian recalibration. ELISPOT validation determines the
ground truth (15/100 confirmed immunogenic).

Implementation matches fig6_practical_usage.py (unified method):
  - Calibration source: zero-shot (1,714 pairs with labels)
  - Distance metric: both S2DD-Lev and S2DD-BLOSUM compared
  - θ = CALIBRATION_THRESHOLD = 0.5
  - λ = CALIBRATION_LAM = 0.0
  - bin-level mp at fit and predict time
  - Linear floor blend: w = max(0.1, clip(PPV+NPV-1, 0, 1))

Metrics:
  - TDR at top-k (Top Discovery Rate — fraction of confirmed among top-k ranked)
  - Top-k precision/recall
  - Comparison: original ranking vs Lev-recalibrated vs BLOSUM-recalibrated
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import parasail

warnings.filterwarnings('ignore')

import os, sys
# CaliPPer self-contained path bootstrap (writes into INPUT_DIR so Stage 1 reads fresh distances)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR
from pathlib import Path
PROJECT_ROOT = Path(INPUT_DIR)
ROOT = PROJECT_ROOT


from calipper.general_evaluator import safe_metric
from calipper.combine_first_helpers import (
    fit_ridge_vbias, CALIBRATION_LAM, CALIBRATION_THRESHOLD,
)

RESULTS_DIR = PROJECT_ROOT / 'results' / 'deepantigen_retrospective' / 'neoantigen_recalibration'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NEOANTIGEN_CSV = PROJECT_ROOT / 'results' / 'deepantigen_retrospective' / \
                 'neoantigen_confidence' / 'neoantigen_s2dd_confidence.csv'
ZS_DIR = PROJECT_ROOT / 'results' / 'deepantigen_retrospective' / 's2dd_degradation'
TRAIN_CSV = PROJECT_ROOT / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data' / 'train.csv'

N_BINS = 8
THRESHOLD = CALIBRATION_THRESHOLD

# ── Unified calibration functions (EXACT copy of fig6, audited 2026-04-16) ──
_logit = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) /
                           (1 - np.clip(p, 1e-12, 1 - 1e-12)))
_sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_ppv_npv_from_cal_sets(cal_data, n_sub=N_BINS, min_samples=30, threshold=THRESHOLD):
    """EXACT COPY of fig6_practical_usage.fit_ppv_npv_from_cal_sets."""
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

    cal_d = np.array(cal_d); cal_mp = np.array(cal_mp)
    cal_ppv = np.array(cal_ppv); cal_npv = np.array(cal_npv)
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


def calibrate_with_curves(y, p, d, ppv_params, npv_params, p_pos, p_neg, n_bins=N_BINS):
    """EXACT COPY of fig6_practical_usage.calibrate_with_curves.

    Uses bin-level mp; requires n_bins of held-out samples. For small cohorts
    (N<30), we use per-sample mp (mp_per_sample = p) as a fallback since
    binning isn't meaningful.
    """
    prev = y.mean()
    # For small samples, use per-sample mp instead of bin-level mp
    if len(d) < n_bins * 5:
        mp_per_sample = p.copy()  # each sample's own prediction
    else:
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

    logit_pp = _logit(p_pos); logit_pn = _logit(p_neg)
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


def compute_neoantigen_sw_topk(k=0.1, b=0.1, K=50):
    """Compute S2DD-BLOSUM distances for 100 neoantigens.

    Same methodology as fig_s2dd_vs_sw_comparison.py and
    eval_deepantigen_bayesian_recalibration.py.
    """
    cache = RESULTS_DIR / 'neoantigen_sw_topk_distances.csv'
    if cache.exists():
        print("  Loading cached neoantigen S2DD-BLOSUM")
        return pd.read_csv(cache)['distance'].values

    print(f"  Computing S2DD-BLOSUM for neoantigens...")
    neo = pd.read_csv(NEOANTIGEN_CSV)
    train = pd.read_csv(TRAIN_CSV)

    train_peps_rows = train['peptide'].values
    unique_train_peps = np.unique(train_peps_rows)
    neo_peps = neo['peptide'].values
    unique_neo_peps = np.unique(neo_peps)

    print(f"    Unique neo: {len(unique_neo_peps)}, unique train: {len(unique_train_peps)}")
    train_self = {s: parasail.sw_stats(s, s, 10, 1, parasail.blosum62).score
                  for s in unique_train_peps}

    sim_by_neo = {}
    t0 = time.time()
    for i, q in enumerate(unique_neo_peps):
        q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
        if q_self <= 0:
            sim_by_neo[q] = np.zeros(len(unique_train_peps))
            continue
        sims = np.array([
            parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score /
            np.sqrt(q_self * train_self[r]) if train_self[r] > 0 else 0.0
            for r in unique_train_peps
        ])
        sim_by_neo[q] = sims

    pep_to_rows = {p: np.where(train_peps_rows == p)[0] for p in unique_train_peps}
    distances = np.zeros(len(neo_peps))
    for i, q in enumerate(neo_peps):
        pep_sims = sim_by_neo[q]
        row_sims = np.zeros(len(train_peps_rows))
        for pep_idx, p in enumerate(unique_train_peps):
            row_sims[pep_to_rows[p]] = pep_sims[pep_idx]
        topk_sims = np.sort(row_sims)[::-1][:K]
        topk_dists = np.log(k * (1.0 - topk_sims + b))
        distances[i] = topk_dists.mean()

    pd.DataFrame({'distance': distances}).to_csv(cache, index=False)
    print(f"    Done ({time.time()-t0:.0f}s), cached to {cache}")
    return distances


def compute_tdr_at_k(scores, labels, k_values):
    """TDR at top-k: fraction of confirmed immunogenic among top-k ranked."""
    order = np.argsort(-scores)  # descending
    results = []
    for k in k_values:
        top_k = order[:k]
        tp = labels[top_k].sum()
        tdr = tp / k
        results.append({'k': k, 'tp': int(tp), 'tdr': tdr,
                       'recall': tp / labels.sum() if labels.sum() > 0 else 0})
    return pd.DataFrame(results)


def main():
    # ── Load neoantigens ─────────────────────────────────────────────────
    print("Loading neoantigens...")
    neo = pd.read_csv(NEOANTIGEN_CSV)
    print(f"  {len(neo)} neoantigens, {neo['confirmed'].sum()} confirmed immunogenic")

    # Neoantigen data
    y_neo = neo['confirmed'].astype(int).values
    p_neo = neo['score'].astype(float).values  # deepAntigen prediction
    d_neo_lev = neo['s2dd_distance'].astype(float).values  # Levenshtein
    d_neo_blosum = compute_neoantigen_sw_topk()  # BLOSUM-topK

    # ── Load zero-shot (calibration source) ──────────────────────────────
    print("\nLoading zero-shot (calibration source)...")
    zs = pd.read_csv(ZS_DIR / 'zero_shot_with_distances.csv')
    y_zs = zs['label'].values.astype(int)
    p_zs = zs['prediction'].values.astype(float)
    d_zs_lev = zs['s2dd_distance'].values.astype(float)
    d_zs_blosum = pd.read_csv(ZS_DIR / 'zero_shot_sw_topk_distances.csv')['distance'].values
    print(f"  Zero-shot: {len(zs)} pairs (calibration source)")

    # ── Run recalibration for both distance metrics ──────────────────────
    all_results = {}
    for name, d_zs, d_neo in [
        ('S2DD-Levenshtein', d_zs_lev, d_neo_lev),
        ('S2DD-BLOSUM', d_zs_blosum, d_neo_blosum),
    ]:
        print(f"\n{'='*70}")
        print(f"Recalibration with {name}")
        print('='*70)

        # Fit on zero-shot
        cal_data = {'zero_shot': (y_zs, p_zs, d_zs)}
        ppv_params, npv_params, p_pos, p_neg, cal_d, cal_ppv, cal_npv = \
            fit_ppv_npv_from_cal_sets(cal_data)
        print(f"  PPV per bin: {[f'{v:.3f}' for v in cal_ppv]}")
        print(f"  NPV per bin: {[f'{v:.3f}' for v in cal_npv]}")
        if ppv_params is not None:
            print(f"  PPV curve: a={ppv_params[0]:.3f} b={ppv_params[1]:.3f} "
                  f"c={ppv_params[2]:.3f} beta={ppv_params[3]:.3f}")
        if npv_params is not None:
            print(f"  NPV curve: a={npv_params[0]:.3f} b={npv_params[1]:.3f} "
                  f"c={npv_params[2]:.3f} beta={npv_params[3]:.3f}")

        # Apply to neoantigens (per-sample mp since N=100 < 40 bins*5)
        cal_scores, ppv_neo, npv_neo = calibrate_with_curves(
            y_neo, p_neo, d_neo, ppv_params, npv_params, p_pos, p_neg)

        # Rank metrics
        print(f"\n  Ranking comparison:")
        print(f"    {'Method':<20} {'AUROC':>8} {'AP':>8}")
        auroc_orig = safe_metric('aucroc', y_neo, p_neo)
        ap_orig = safe_metric('ap', y_neo, p_neo)
        auroc_cal = safe_metric('aucroc', y_neo, cal_scores)
        ap_cal = safe_metric('ap', y_neo, cal_scores)
        print(f"    {'Original':<20} {auroc_orig:>8.4f} {ap_orig:>8.4f}")
        print(f"    {'Calibrated':<20} {auroc_cal:>8.4f} {ap_cal:>8.4f}")
        print(f"    {'Δ':<20} {auroc_cal-auroc_orig:>+8.4f} {ap_cal-ap_orig:>+8.4f}")

        # TDR at top-k
        print(f"\n  TDR at top-k (baseline = 15/100 = 15%):")
        tdr_orig = compute_tdr_at_k(p_neo, y_neo, [5, 10, 15, 20, 25, 30, 50])
        tdr_cal = compute_tdr_at_k(cal_scores, y_neo, [5, 10, 15, 20, 25, 30, 50])
        print(f"    {'k':<4} {'Orig TP':>8} {'Orig TDR':>10} {'Cal TP':>8} {'Cal TDR':>10} {'ΔTP':>6}")
        for _, (ro, rc) in enumerate(zip(tdr_orig.itertuples(), tdr_cal.itertuples())):
            print(f"    {ro.k:<4} {ro.tp:>8d} {ro.tdr*100:>9.1f}% "
                  f"{rc.tp:>8d} {rc.tdr*100:>9.1f}% {rc.tp-ro.tp:>+6d}")

        all_results[name] = {
            'calibrated_scores': cal_scores,
            'auroc_orig': auroc_orig, 'auroc_cal': auroc_cal,
            'ap_orig': ap_orig, 'ap_cal': ap_cal,
            'tdr_orig': tdr_orig, 'tdr_cal': tdr_cal,
            'ppv_params': ppv_params, 'npv_params': npv_params,
            'ppv_neo': ppv_neo, 'npv_neo': npv_neo,
        }

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY: Original vs Bayesian Recalibrated Ranking (100 neoantigens)")
    print("="*80)
    print(f"{'Method':<24} {'AUROC':>8} {'Δ':>8} {'AP':>8} {'Δ':>8}")
    print("-"*80)
    orig_auroc = safe_metric('aucroc', y_neo, p_neo)
    orig_ap = safe_metric('ap', y_neo, p_neo)
    print(f"{'deepAntigen (original)':<24} {orig_auroc:>8.4f} {'—':>8} {orig_ap:>8.4f} {'—':>8}")
    for name, res in all_results.items():
        print(f"{name:<24} {res['auroc_cal']:>8.4f} {res['auroc_cal']-orig_auroc:>+8.4f} "
              f"{res['ap_cal']:>8.4f} {res['ap_cal']-orig_ap:>+8.4f}")

    print(f"\n{'Method':<24} {'Top-15 TDR':>12} {'Top-20 TDR':>12} {'Top-30 TDR':>12}")
    print("-"*80)
    for name, res in all_results.items():
        t = res['tdr_cal']
        t15 = t[t['k']==15]['tdr'].iloc[0] * 100
        t20 = t[t['k']==20]['tdr'].iloc[0] * 100
        t30 = t[t['k']==30]['tdr'].iloc[0] * 100
        print(f"{name+' (cal)':<24} {t15:>11.1f}% {t20:>11.1f}% {t30:>11.1f}%")

    # Save comparison
    out_rows = [{'method': 'Original', 'auroc': orig_auroc, 'ap': orig_ap}]
    for name, res in all_results.items():
        out_rows.append({'method': name + ' (calibrated)',
                         'auroc': res['auroc_cal'], 'ap': res['ap_cal']})
    pd.DataFrame(out_rows).to_csv(RESULTS_DIR / 'recalibration_summary.csv', index=False)

    # Per-neoantigen calibrated scores
    neo_out = neo.copy()
    neo_out['score_lev_calibrated'] = all_results['S2DD-Levenshtein']['calibrated_scores']
    neo_out['score_blosum_calibrated'] = all_results['S2DD-BLOSUM']['calibrated_scores']
    neo_out['s2dd_blosum'] = d_neo_blosum
    neo_out.to_csv(RESULTS_DIR / 'neoantigen_recalibrated.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR}")
    return all_results


if __name__ == '__main__':
    main()
