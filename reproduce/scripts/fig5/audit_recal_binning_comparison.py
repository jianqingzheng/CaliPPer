#!/usr/bin/env python3
"""Audit: Compare recalibration binning strategies (distance vs subset) × distance weighting (sigma_C vs uniform).

Three methods compared on TCR CT (5 models × 4 test sets):
  A: sigma_C distances + distance-binning (current fig5 default)
  B: sigma_C distances + subset-binning (epitope groups)
  C: uniform distances + subset-binning (fig4-aligned setting)

Evaluation at two levels:
  - Dataset-level: ΔAUROC and ΔAP per test set
  - Per-epitope: mean ΔAUROC and ΔAP across epitopes with ≥30 samples

All methods use v2.7 fit_recalibration/apply_recalibration with adaptive defaults.

Output: audit_recal_binning_comparison_results.csv + summary to stdout
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from calipper.core import fit_recalibration, apply_recalibration
from calipper.general_evaluator import safe_metric

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
MODEL_DISPLAY = {'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
                 'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT'}
TCR_CAL = ['v3_combined', 'v4_combined']
TCR_TEST = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
MIN_SAMPLES = 30

METHODS = [
    ('A_sigC_distbin', '_dist.npy', 'distance'),
    ('B_sigC_subbin',  '_dist.npy', 'subset'),
    ('C_uni_subbin',   '_uniform_dist.npy', 'subset'),
]


def load_model_data(model, dist_suffix):
    """Load predictions + distances for all cal+test sets."""
    ct = {}
    for ts in TCR_CAL + TCR_TEST:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{dist_suffix}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
        ct[ts] = {
            'y': df[lc].values[:n].astype(int),
            'p': df[pc].values[:n].astype(float),
            'd': d[:n].astype(float),
            'ep': df[ep_col].values[:n] if ep_col in df.columns else None,
        }
    return ct


def run_recalibration(ct, bin_strategy):
    """Fit recalibration and apply to all test sets. Returns per-test and per-epitope results."""
    cal_data = {s: (ct[s]['y'], ct[s]['p'], ct[s]['d']) for s in TCR_CAL if s in ct}
    if not cal_data:
        return [], []

    if bin_strategy == 'subset':
        cal_y = np.concatenate([ct[s]['y'] for s in TCR_CAL if s in ct])
        cal_p = np.concatenate([ct[s]['p'] for s in TCR_CAL if s in ct])
        cal_d = np.concatenate([ct[s]['d'] for s in TCR_CAL if s in ct])
        cal_ep = np.concatenate([ct[s]['ep'] for s in TCR_CAL if s in ct])
        cal_subsets = {}
        for ep in np.unique(cal_ep):
            mask = cal_ep == ep
            if mask.sum() >= MIN_SAMPLES:
                cal_subsets[ep] = (cal_y[mask], cal_p[mask], cal_d[mask])
        if len(cal_subsets) < 4:
            return [], []
        ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(
            cal_data, bin_strategy='subset', cal_subsets=cal_subsets)
    else:
        ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(cal_data)

    ds_rows = []
    ep_rows = []

    for ts in TCR_TEST:
        if ts not in ct:
            continue
        test = ct[ts]
        cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                    ppv_p, npv_p, pp, pn, prev=cal_prev)

        # Dataset-level
        for metric in ['aucroc', 'ap']:
            b = safe_metric(metric, test['y'], test['p'])
            a = safe_metric(metric, test['y'], cal_s)
            ds_rows.append({'test_set': ts, 'metric': metric, 'before': b, 'after': a, 'delta': a - b})

        # Per-epitope
        if test['ep'] is not None:
            for ep in np.unique(test['ep']):
                mask = test['ep'] == ep
                if mask.sum() < MIN_SAMPLES:
                    continue
                yi = test['y'][mask]
                if yi.sum() == 0 or yi.sum() == len(yi):
                    continue
                for metric in ['aucroc', 'ap']:
                    b = safe_metric(metric, yi, test['p'][mask])
                    a = safe_metric(metric, yi, cal_s[mask])
                    ep_rows.append({
                        'test_set': ts, 'epitope': ep, 'metric': metric,
                        'n_samples': int(mask.sum()), 'prevalence': float(yi.mean()),
                        'mean_distance': float(test['d'][mask].mean()),
                        'before': b, 'after': a, 'delta': a - b,
                        # Diagnostic: how many predictions above threshold for this epitope
                        'n_above_theta': int((test['p'][mask] >= 0.4).sum()),
                        'pct_above_theta': float((test['p'][mask] >= 0.4).mean()),
                    })

    return ds_rows, ep_rows


# ═══════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════
all_rows = []

for model in TCR_MODELS:
    display = MODEL_DISPLAY[model]
    print(f"\n=== {display} ===")

    for method_name, dist_suffix, bin_strat in METHODS:
        ct = load_model_data(model, dist_suffix)
        if 'v3_combined' not in ct or 'v4_combined' not in ct:
            print(f"  {method_name}: missing data, skipping")
            continue

        ds_rows, ep_rows = run_recalibration(ct, bin_strat)

        for r in ds_rows:
            r['model'] = model; r['method'] = method_name; r['level'] = 'dataset'
            all_rows.append(r)
        for r in ep_rows:
            r['model'] = model; r['method'] = method_name; r['level'] = 'per_epitope'
            all_rows.append(r)

        ds_auc = np.mean([r['delta'] for r in ds_rows if r['metric'] == 'aucroc'])
        ep_auc = np.mean([r['delta'] for r in ep_rows if r['metric'] == 'aucroc']) if ep_rows else np.nan
        ep_ap = np.mean([r['delta'] for r in ep_rows if r['metric'] == 'ap']) if ep_rows else np.nan
        n_ep = len([r for r in ep_rows if r['metric'] == 'aucroc'])
        print(f"  {method_name:20s} DS_ΔAUC={ds_auc:+.4f}  EP_ΔAUC={ep_auc:+.4f}  EP_ΔAP={ep_ap:+.4f}  n_ep={n_ep}")


# Save full results
df = pd.DataFrame(all_rows)
out_path = os.path.join(os.path.dirname(__file__), '..', 'audit_recal_binning_comparison_results.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path}")

# ═══════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════
print("\n" + "=" * 80)
print("GRAND SUMMARY")
print("=" * 80)

for level in ['dataset', 'per_epitope']:
    print(f"\n--- {level} ---")
    for metric in ['aucroc', 'ap']:
        print(f"  {metric.upper()}:")
        for method_name in ['A_sigC_distbin', 'B_sigC_subbin', 'C_uni_subbin']:
            sub = df[(df['method'] == method_name) & (df['level'] == level) & (df['metric'] == metric)]
            mean_d = sub['delta'].mean()
            n_improved = (sub['delta'] > 0).sum()
            n_total = len(sub)
            print(f"    {method_name:20s} mean_Δ={mean_d:+.4f}  improved={n_improved}/{n_total}")

# Per-epitope diagnostic: threshold sparsity by method
print("\n--- Diagnostic: threshold sparsity (per-epitope, AUROC) ---")
ep_df = df[(df['level'] == 'per_epitope') & (df['metric'] == 'aucroc')]
if 'pct_above_theta' in ep_df.columns:
    for method_name in ['A_sigC_distbin', 'B_sigC_subbin', 'C_uni_subbin']:
        sub = ep_df[ep_df['method'] == method_name]
        if len(sub) == 0:
            continue
        mean_pct = sub['pct_above_theta'].mean()
        pct_zero = (sub['pct_above_theta'] == 0).mean() * 100
        print(f"  {method_name:20s} mean_pct_above_theta={mean_pct:.3f}  pct_epitopes_with_zero={pct_zero:.1f}%")
