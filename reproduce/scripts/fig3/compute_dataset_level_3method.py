#!/usr/bin/env python3
"""Compute DATASET-LEVEL predictions for 3 methods: S2DD, PAPE, M-CBPE.

For fig3 panels l (TCR) and p (BCR) — method comparison boxplots.

Protocol: LOO across test sets (TCR CT). Only v3+v4 as calibration
(excludes IEDB, McPAS, seen, unseen — real-case simulation).

Baselines use prediction-only DRE (no distance input at all).
Distance-based distribution shift detection is S2DD's contribution.

S2DD uses full multi-chain LogDist K=50 with DRE + vbias curve correction.

AUDIT CHECKLIST (verify before running):
  [ ] PAPE DRE features: [cal_p] only — NO distance input
  [ ] M-CBPE DRE features: [cal_p] only — NO distance input
  [ ] S2DD: uses cal_d from .npy files — completely separate from cal_lev
  [ ] No test labels used during fitting
  [ ] Same test sets for all 3 methods
  [ ] predict_metric called with adaptive_bins=True (v2.7 default)

Output: fig3/fig3_dataset_level_3method.csv

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:Manuscript/designed_figures/panels:. \
        python Manuscript/designed_figures/panels/fig3/scripts/compute_dataset_level_3method.py
"""
import sys, os, time
import numpy as np
import pandas as pd
import Levenshtein

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.general_evaluator import safe_metric
from calipper.core import predict_metric

# PAPE and M-CBPE
from PAPE.pape_core import estimate_importance_weights, fit_weighted_calibration, apply_calibration
from PAPE.pape_core import estimate_metric as pape_eq4
try:
    from MCBPE.mcbpe_core import (estimate_density_ratios, fit_weighted_calibrator,
                                   calibrate_predictions, estimate_metric_from_calibrated)
    HAS_MCBPE = True
except ImportError:
    HAS_MCBPE = False
    print("WARNING: M-CBPE not available")

RESULTS = os.path.join(INPUT_DIR, 'results')
OUT_CSV = os.path.join(SCRIPT_DIR, '..', 'fig3_dataset_level_3method.csv')
METRICS = ['aucroc', 'ap', 'f1']

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
# Only v3+v4 as calibration (real-case simulation: standard validation splits)
EXCLUDE_FROM_CAL = {'iedb_sars', 'mcpas', 'seen_test', 'unseen_fold34'}

records = []
t0 = time.time()


def naive_lev_fast(test_seqs, cal_seqs, subsample=500, seed=42):
    """Naive 1-chain Levenshtein: 1 - mean(ratio) over subsampled cal.

    Subsamples cal to 500 for computational feasibility.
    No log, no topK, no sigma_C. Pure naive baseline.
    """
    rng = np.random.RandomState(seed)
    if len(cal_seqs) > subsample:
        idx = rng.choice(len(cal_seqs), subsample, replace=False)
        cal_sub = [cal_seqs[i] for i in idx]
    else:
        cal_sub = list(cal_seqs)
    dists = []
    for te in test_seqs:
        ratios = [Levenshtein.ratio(str(te), str(ce)) for ce in cal_sub]
        dists.append(1.0 - np.mean(ratios))
    return np.array(dists)


# ═══════════════════════════════════════════
# TCR CT: LOO across test sets
# ═══════════════════════════════════════════
print("=== TCR CT: dataset-level LOO ===", flush=True)

for mi, model in enumerate(TCR_MODELS):
    print(f"  [{mi+1}/{len(TCR_MODELS)}] {model}...", flush=True)

    all_data = {}
    for ts in TCR_CT_SETS:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(RESULTS, 'fig2_cache', f'{model}_ct_{ts}_dist.npy')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)[:len(df)]
        y_col = 'binder' if 'binder' in df.columns else 'y_true'
        p_col = 'prediction' if 'prediction' in df.columns else 'y_prob'
        cdr3b_col = next((c for c in ['CDR3b', 'CDR3B', 'cdr3b'] if c in df.columns), None)
        if y_col not in df.columns or p_col not in df.columns:
            continue
        all_data[ts] = {
            'y': df[y_col].values.astype(int),
            'p': df[p_col].values.astype(float),
            'd': d,
            'cdr3b': df[cdr3b_col].astype(str).values if cdr3b_col else None
        }

    for held_ts in TCR_CT_SETS:
        if held_ts not in all_data:
            continue
        test = all_data[held_ts]
        # Cal: only v3+v4 (exclude external + seen/unseen)
        cal_sets = {k: v for k, v in all_data.items()
                    if k != held_ts and k not in EXCLUDE_FROM_CAL}
        if not cal_sets:
            continue

        cal_y = np.concatenate([v['y'] for v in cal_sets.values()])
        cal_p = np.concatenate([v['p'] for v in cal_sets.values()])
        cal_d = np.concatenate([v['d'] for v in cal_sets.values()])

        # ── S2DD: multi-chain LogDist K=50 + DRE + vbias ──
        # AUDIT: uses cal_d from .npy (S2DD distance), NOT cal_lev
        cal_data_s2dd = {cn: (cv['y'], cv['p'], cv['d']) for cn, cv in cal_sets.items()}
        s2dd_result = predict_metric(cal_data_s2dd, test['p'], test['d'],
                                      metrics=METRICS, adaptive_bins=True)

        # ── PAPE + M-CBPE: prediction-only DRE (NO distance input) ──
        # Distance-based distribution shift detection is S2DD's contribution.
        # PAPE/M-CBPE use DRE on [prediction] only — their standard design.
        has_cdr3b = True  # no longer need CDR3b for baselines

        # PAPE: DRE on [prediction] only
        w_pape, _, _ = estimate_importance_weights(
            cal_p.reshape(-1, 1), test['p'].reshape(-1, 1))
        cm_pape = fit_weighted_calibration(cal_p, cal_y, w_pape)
        c_pape = apply_calibration(cm_pape, test['p'])

        # M-CBPE: DRE on [prediction] only
        if HAS_MCBPE:
            mcbpe_w, _ = estimate_density_ratios(
                cal_p.reshape(-1, 1), test['p'].reshape(-1, 1))
            mcbpe_cal = fit_weighted_calibrator(cal_p, cal_y, mcbpe_w)
            mcbpe_c = calibrate_predictions(mcbpe_cal, test['p'])

        for metric in METRICS:
            actual = safe_metric(metric, test['y'], test['p'])
            if np.isnan(actual):
                continue

            # S2DD prediction
            sp = s2dd_result['estimated'].get(metric, np.nan)
            records.append(dict(domain='TCR', model=model, test_set=held_ts, metric=metric,
                                method='S2DD', predicted=sp, actual=actual,
                                abs_error=abs(sp - actual)))

            # PAPE prediction — AUDIT: uses c_pape (from cal_lev DRE), NOT c_s2dd
            if has_cdr3b:
                pp = pape_eq4(c_pape, test['p'], metric, threshold=0.5)
                records.append(dict(domain='TCR', model=model, test_set=held_ts, metric=metric,
                                    method='PAPE', predicted=pp, actual=actual,
                                    abs_error=abs(pp - actual)))

                # M-CBPE prediction
                if HAS_MCBPE:
                    mp = estimate_metric_from_calibrated(mcbpe_c, metric)
                    records.append(dict(domain='TCR', model=model, test_set=held_ts, metric=metric,
                                        method='M-CBPE', predicted=mp, actual=actual,
                                        abs_error=abs(mp - actual)))

print(f"  TCR done ({time.time()-t0:.0f}s)", flush=True)


# ═══════════════════════════════════════════
# BCR CT: fold4-as-cal → predict each test set
# ═══════════════════════════════════════════
print("\n=== BCR CT: fold4-as-cal ===", flush=True)

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_CT_SETS = ['A1-A11', 'unseen', 'flu']
BCR_CAL_DIR = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')

for mi, model in enumerate(BCR_MODELS):
    print(f"  [{mi+1}/{len(BCR_MODELS)}] {model}...", flush=True)
    model_dir = os.path.join(BCR_CAL_DIR, model)
    if not os.path.exists(model_dir):
        print(f"    Skip: {model_dir} not found")
        continue

    # Load fold4 cal data (cal_predictions.csv has distance column)
    cal_path = os.path.join(model_dir, 'cal_predictions.csv')
    if not os.path.exists(cal_path):
        print(f"    Skip: cal_predictions.csv not found")
        continue
    cal_df = pd.read_csv(cal_path)
    cal_y = cal_df['rbd'].values.astype(int)
    cal_p = cal_df['pred_prob'].values.astype(float)
    cal_d = cal_df['distance'].values.astype(float)  # Lev-log S2DD distance from CSV

    for ts in BCR_CT_SETS:
        test_path = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(test_path):
            continue
        test_df = pd.read_csv(test_path)
        test_y = test_df['rbd'].values.astype(int)
        test_p = test_df['pred_prob'].values.astype(float)
        test_d = test_df['distance'].values.astype(float)

        # S2DD prediction
        cal_data_s2dd = {'fold4': (cal_y, cal_p, cal_d)}
        s2dd_result = predict_metric(cal_data_s2dd, test_p, test_d,
                                      metrics=METRICS, adaptive_bins=True)

        # PAPE/M-CBPE: prediction-only DRE (NO distance input)
        w_p, _, _ = estimate_importance_weights(
            cal_p.reshape(-1, 1), test_p.reshape(-1, 1))
        cm_p = fit_weighted_calibration(cal_p, cal_y, w_p)
        c_p = apply_calibration(cm_p, test_p)
        if HAS_MCBPE:
            mw, _ = estimate_density_ratios(
                cal_p.reshape(-1, 1), test_p.reshape(-1, 1))
            mc = fit_weighted_calibrator(cal_p, cal_y, mw)
            mc_c = calibrate_predictions(mc, test_p)

        for metric in METRICS:
            actual = safe_metric(metric, test_y, test_p)
            if np.isnan(actual): continue
            sp = s2dd_result['estimated'].get(metric, np.nan)
            records.append(dict(domain='BCR', model=model, test_set=ts, metric=metric,
                                method='S2DD', predicted=sp, actual=actual,
                                abs_error=abs(sp - actual)))
            pp = pape_eq4(c_p, test_p, metric, threshold=0.5)
            records.append(dict(domain='BCR', model=model, test_set=ts, metric=metric,
                                method='PAPE', predicted=pp, actual=actual,
                                abs_error=abs(pp - actual)))
            if HAS_MCBPE:
                mp = estimate_metric_from_calibrated(mc_c, metric)
                records.append(dict(domain='BCR', model=model, test_set=ts, metric=metric,
                                    method='M-CBPE', predicted=mp, actual=actual,
                                    abs_error=abs(mp - actual)))

print(f"  BCR done ({time.time()-t0:.0f}s)", flush=True)


# Save
df_out = pd.DataFrame(records)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV} ({len(df_out)} rows)")

# Summary
for domain in ['TCR', 'BCR']:
    dom = df_out[df_out['domain'] == domain]
    if len(dom) == 0: continue
    print(f"\n=== {domain} CT dataset-level MAE ===")
    for metric in METRICS:
        sub = dom[dom['metric'] == metric]
        for method in ['PAPE', 'M-CBPE', 'S2DD']:
            ms = sub[sub['method'] == method]
            if len(ms) > 0:
                print(f"  {metric:>6s} {method:<7s}: MAE={ms['abs_error'].mean():.4f} (n={len(ms)})")

print(f"\nTotal time: {time.time()-t0:.0f}s")
