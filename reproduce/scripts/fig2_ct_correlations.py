#!/usr/bin/env python3
"""Fig 2 cross-test correlations extractor (TCR side).

Computes per-(model, test_set, metric) Pearson |r| between binned S2DD
distance and binned AUROC/AP. Mirrors the panel e/f computation from
`generate_all_panels.py:200-227` exactly: 8 equal-sized distance bins,
per-bin metric, Pearson r over (bin_distance, bin_metric).

Distance metric is controlled by the module constant ``DIST_TYPE`` (see
below). The default is ``'lev-log'`` (matching reproduce_fig2.sh's
default), which reads lev-log distances written by Stage 1
(``compute_fig2_levlog_distances.py``). Setting ``DIST_TYPE='blosum-sqrt'``
switches to the alternate distance variant.

Inputs (under INPUT_DIR / OUTPUT_DIR):
    results/{model}/cross_test_logdist/predictions/{test_set}_predictions_with_label.csv
        - per-model cached prediction CSVs (the canonical artifact)

    Distance arrays — path depends on DIST_TYPE:
      DIST_TYPE='lev-log'     (DEFAULT, used by reproduce_fig2.sh):
        OUTPUT_DIR/fig2_cache_lev/{ts}_dist.npy
        (Stage 1 output of compute_fig2_levlog_distances.py — note that
         the file naming convention is {ts}_dist.npy with no model prefix,
         since the array is the same shared train pool for all 5 models)
      DIST_TYPE='blosum-sqrt' (alternate, not used by default pipeline):
        INPUT_DIR/results/fig2_cache/{model}_ct_{test_set}_blosumsqrt_dist.npy
        (per-model staged blosum-sqrt arrays)

Outputs (under OUTPUT_DIR):
    fig2_ct_correlations_{lev,blosumsqrt}.csv: per-cell |r| with columns
        model, test_set, n_samples, n_valid_bins, abs_r_auroc, abs_r_ap
    (The suffix matches DIST_TYPE. The default pipeline writes
     fig2_ct_correlations_lev.csv; there is no file with the bare name
     fig2_ct_correlations.csv unless a separate ad-hoc run produced it.)

Scope (cross-test only, per the 2026-05-20 CV data gap and user's
Option A decision; see published_repo/BUILD_PROGRESS.md P4b.systematic.fig2):
    - 5 TCR models: nettcr, atm_tcr, blosum_rf, ergo_ii, tcrbert
    - 6 CT test sets: seen_test, unseen_fold34, v3_combined, v4_combined,
      mcpas, iedb_sars
    - 30 cells total

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/fig2_ct_correlations.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.general_evaluator import safe_metric

RESULTS = os.path.join(INPUT_DIR, 'results')
BLOSUM_CACHE = os.path.join(RESULTS, 'fig2_cache')
LEVLOG_CACHE = os.path.join(OUTPUT_DIR, 'fig2_cache_lev')

# Default: lev-log (matches manuscript canonical fig2 panels in lev-logtransf/).
# Set DIST_TYPE='blosum-sqrt' env var to switch.
DIST_TYPE = os.environ.get('DIST_TYPE', 'lev-log')
assert DIST_TYPE in ('lev-log', 'blosum-sqrt'), \
    f"DIST_TYPE must be 'lev-log' or 'blosum-sqrt', got '{DIST_TYPE}'"

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined',
               'mcpas', 'iedb_sars']
N_BINS = 8


def get_distance_path(model: str, test_set: str) -> str:
    """Resolve distance .npy path for the current DIST_TYPE.

    For lev-log: distances are model-independent (one .npy per test set)
    because they only depend on test sequences vs the canonical
    tcr_ml_v4 training reference.

    For blosum-sqrt: distances are per-(model, test_set), reading from
    the pre-computed cache in INPUT_DIR/results/fig2_cache/.
    """
    if DIST_TYPE == 'lev-log':
        return os.path.join(LEVLOG_CACHE, f'{test_set}_dist.npy')
    return os.path.join(BLOSUM_CACHE, f'{model}_ct_{test_set}_blosumsqrt_dist.npy')


def per_cell_correlations(model: str, test_set: str) -> dict | None:
    """Compute |r| AUROC and |r| AP for one (model, test_set) cell."""
    pred_path = os.path.join(RESULTS, model, 'cross_test_logdist',
                              'predictions', f'{test_set}_predictions_with_label.csv')
    dist_path = get_distance_path(model, test_set)
    if not os.path.exists(pred_path) or not os.path.exists(dist_path):
        return None
    df = pd.read_csv(pred_path)
    d = np.load(dist_path)
    if len(d) != len(df):
        d = d[:len(df)]
    lc = 'binder' if 'binder' in df.columns else 'y_true'
    pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
    y = df[lc].values.astype(int)
    p = df[pc].values.astype(float)
    si = np.argsort(d)
    bs = len(si) // N_BINS
    if bs < 5:
        return None
    bx = np.empty(N_BINS)
    by_auc = np.empty(N_BINS)
    by_ap = np.empty(N_BINS)
    for i in range(N_BINS):
        s = i * bs
        e = len(si) if i == N_BINS - 1 else (i + 1) * bs
        idx = si[s:e]
        bx[i] = d[idx].mean()
        by_auc[i] = safe_metric('aucroc', y[idx], p[idx])
        by_ap[i] = safe_metric('ap', y[idx], p[idx])
    # Pearson r between bin distance and per-bin metric; skip NaN bins
    valid_auc = ~np.isnan(by_auc)
    valid_ap = ~np.isnan(by_ap)
    r_auc = pearsonr(bx[valid_auc], by_auc[valid_auc])[0] if valid_auc.sum() >= 3 else np.nan
    r_ap = pearsonr(bx[valid_ap], by_ap[valid_ap])[0] if valid_ap.sum() >= 3 else np.nan
    return {
        'model': model,
        'test_set': test_set,
        'n_samples': len(df),
        'n_valid_bins_auroc': int(valid_auc.sum()),
        'n_valid_bins_ap': int(valid_ap.sum()),
        'abs_r_auroc': abs(r_auc) if not np.isnan(r_auc) else np.nan,
        'abs_r_ap': abs(r_ap) if not np.isnan(r_ap) else np.nan,
        'signed_r_auroc': r_auc,
        'signed_r_ap': r_ap,
    }


def main() -> int:
    print(f"[fig2_ct] INPUT_DIR={INPUT_DIR}")
    print(f"[fig2_ct] DIST_TYPE={DIST_TYPE}")
    cache = LEVLOG_CACHE if DIST_TYPE == 'lev-log' else BLOSUM_CACHE
    print(f"[fig2_ct] CACHE={cache}")
    print(f"[fig2_ct] Computing per-cell |r| for {len(TCR_MODELS)} models x {len(TCR_CT_SETS)} test sets...")
    print()
    rows = []
    for model in TCR_MODELS:
        for ts in TCR_CT_SETS:
            row = per_cell_correlations(model, ts)
            if row is None:
                print(f"  SKIP  {model:<10} {ts:<18} (inputs absent)")
                continue
            rows.append(row)
            print(f"  OK    {model:<10} {ts:<18}  n={row['n_samples']:>5d}  "
                  f"|r| AUROC={row['abs_r_auroc']:.3f}  |r| AP={row['abs_r_ap']:.3f}")

    df = pd.DataFrame(rows)
    suffix = 'lev' if DIST_TYPE == 'lev-log' else 'blosumsqrt'
    out = os.path.join(OUTPUT_DIR, f'fig2_ct_correlations_{suffix}.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(out, index=False, float_format='%.6f')
    print()
    print(f"Wrote: {out}")
    print()

    # Aggregate statistics
    print("=== Aggregate (per-model) ===")
    by_model = df.groupby('model').agg(
        mean_abs_r_auroc=('abs_r_auroc', 'mean'),
        mean_abs_r_ap=('abs_r_ap', 'mean'),
        n_cells=('test_set', 'count'),
    ).round(4)
    print(by_model)
    print()
    print("=== Aggregate (overall) ===")
    print(f"  n_cells:             {len(df)}")
    print(f"  mean |r| AUROC:      {df['abs_r_auroc'].mean():.4f}")
    print(f"  mean |r| AP:         {df['abs_r_ap'].mean():.4f}")
    print(f"  median |r| AUROC:    {df['abs_r_auroc'].median():.4f}")
    print(f"  median |r| AP:       {df['abs_r_ap'].median():.4f}")
    print(f"  |r| > 0.6 on AP:     {(df['abs_r_ap'] > 0.6).sum()}/{len(df)}")
    print(f"  signed r AP < 0:     {(df['signed_r_ap'] < 0).sum()}/{len(df)}  (manuscript: 'declined with distance')")
    print(f"  signed r AUROC < 0:  {(df['signed_r_auroc'] < 0).sum()}/{len(df)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
