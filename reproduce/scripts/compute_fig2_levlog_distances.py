#!/usr/bin/env python3
"""Precompute Levenshtein-log S2DD distances for fig2 TCR cross-test sets.

Pipeline:
1. Build training pool from ``tcr_ml_stratified_v3/`` using the
   seen-pool 50/50 stratified split (canonical ``build_pools()`` from
   ``eval_cross_test_logdist.py``).
2. Compute sigma_C chain weights (Simpson concentration based), the
   canonical method for degradation curves per CLAUDE.md and
   ``feedback_weighting_tradeoff.md`` memory: "Degradation → sigma_C
   both" (best for bin-level degradation, bin_R²=0.791).
3. Per test set: compute 3-chain LogDist via
   ``compute_combine_first_distances()``.

**Known limitation (2026-05-29)**: fig2 CT |r| values reproduce
qualitatively (sign + broad magnitude) but cannot bit-exactly match the
manuscript ATM-TCR panel e legend (-0.98, -0.79, -0.87, -0.94, -0.96,
-0.83). 3/6 cells reproduce well (Unseen, v4, IEDB), 3/6 diverge (Seen,
v3, McPAS). The divergence pattern is per-cell (not uniformly off),
which rules out an S2DD method/parameter mismatch — the same chain
weight formula gives 3 matching cells. The cause is the 2026-05-20
model-deletion incident: prediction CSVs we read are from post-retrain
models that differ from the manuscript canonical pre-retrain models.
Same Option A scenario as Fig 3 and Fig 5.

Inputs (under INPUT_DIR):
    Data/tcr_seq/proc_files/tcr_ml_stratified_v3/{train,validation,test}_data.csv
    Data/tcr_seq/proc_files/epitope_fold_assignments.csv
    results/nettcr/cross_test_logdist/predictions/{ts}_predictions_with_label.csv

Outputs (gitignored, regenerated each run):
    OUTPUT_DIR/fig2_cache_lev/{ts}_dist.npy

Parameters (canonical):
    k = 0.1, b = 0.1, K = 50
    chain_cols = ['peptide', 'CDR3a', 'CDR3b']
    weight_formula = sigma_C (per CLAUDE.md "Degradation → sigma_C" rule;
        this is what the code actually calls at compute_chain_weights(...,
        formula='sigma_C'). An earlier docstring revision incorrectly said
        sigma_H — that was a non-canonical detour documented in
        BUILD_PROGRESS.md "Fig 2 Stage 3 method-side iterations" and is
        NOT what the code does.)
    seed = 42 (for seen-pool 50/50 split)

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/compute_fig2_levlog_distances.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.combine_first_helpers import (
    compute_chain_weights,
    compute_combine_first_distances,
)

RESULTS = os.path.join(INPUT_DIR, 'results')
DATA_DIR = os.path.join(INPUT_DIR, 'Data', 'tcr_seq', 'proc_files')

K_PARAM = 0.1
B_PARAM = 0.1
TOPK = 50
CHAIN_COLS = ['peptide', 'CDR3a', 'CDR3b']
SUBSAMPLE = 500
SEED = 42

TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined',
               'mcpas', 'iedb_sars']

OUT_DIR = os.path.join(OUTPUT_DIR, 'fig2_cache_lev')


def build_train_pool() -> pd.DataFrame:
    """Mirror eval_cross_test_logdist.py:build_pools train_df construction.

    1. Concatenate v3 train + val + test.
    2. Add fold column from epitope_fold_assignments.csv.
    3. Seen pool = folds 0,1,2.
    4. 50/50 stratified split by epitope (singletons -> training).
    5. train_df = train_main + singleton_pool.
    """
    from sklearn.model_selection import train_test_split

    v3_dir = os.path.join(DATA_DIR, 'tcr_ml_stratified_v3')
    v3_train = pd.read_csv(os.path.join(v3_dir, 'train_data.csv'))
    v3_val = pd.read_csv(os.path.join(v3_dir, 'validation_data.csv'))
    v3_test = pd.read_csv(os.path.join(v3_dir, 'test_data.csv'))
    v3_all = pd.concat([v3_train, v3_val, v3_test], ignore_index=True)

    fold_path = os.path.join(DATA_DIR, 'epitope_fold_assignments.csv')
    fold_df = pd.read_csv(fold_path)
    ep_to_fold = dict(zip(fold_df['epitope'], fold_df['fold']))
    v3_all['fold'] = v3_all['epitope'].map(ep_to_fold)
    assert v3_all['fold'].notna().all(), "Some epitopes not in fold assignments!"

    seen_pool = v3_all[v3_all['fold'].isin([0, 1, 2])].copy()

    ep_counts = seen_pool['epitope'].value_counts()
    singleton_eps = set(ep_counts[ep_counts == 1].index)
    singleton_mask = seen_pool['epitope'].isin(singleton_eps)

    main_pool = seen_pool[~singleton_mask]
    singleton_pool = seen_pool[singleton_mask]

    train_main, _test_main = train_test_split(
        main_pool, test_size=0.5, random_state=SEED,
        stratify=main_pool['epitope'])

    if len(singleton_pool) > 0:
        train_df = pd.concat([train_main, singleton_pool], ignore_index=True)
    else:
        train_df = train_main.reset_index(drop=True)

    # Rename to match cross-test prediction CSV column convention
    rename = {'epitope': 'peptide', 'cdr3_a': 'CDR3a', 'cdr3_b': 'CDR3b'}
    train_df = train_df.rename(columns=rename)
    for c in CHAIN_COLS:
        train_df[c] = train_df[c].fillna('').astype(str)
    return train_df.reset_index(drop=True)


# compute_chain_weights with formula='sigma_C' from calipper is canonical
# for degradation. See memory feedback_weighting_tradeoff.md.


def load_test(test_set: str) -> pd.DataFrame:
    path = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist',
                        'predictions', f'{test_set}_predictions_with_label.csv')
    df = pd.read_csv(path)
    for c in CHAIN_COLS:
        df[c] = df[c].fillna('').astype(str)
    return df


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[compute_fig2_levlog] OUT_DIR={OUT_DIR}")
    print()

    t0 = time.time()
    train_df = build_train_pool()
    print(f"[1/3] Built train pool from tcr_ml_stratified_v3 (seen-pool 50/50): "
          f"n={len(train_df)} rows ({time.time()-t0:.1f}s)")

    t1 = time.time()
    print(f"[2/3] Computing sigma_C chain weights (canonical for degradation)...")
    weights, _ = compute_chain_weights(
        train_df, CHAIN_COLS, K_PARAM, B_PARAM, TOPK, formula='sigma_C')
    weights_str = ', '.join(f"{c}={w:.4f}" for c, w in zip(CHAIN_COLS, weights))
    print(f"      Chain weights (sigma_C): {weights_str}  ({time.time()-t1:.1f}s)")
    print()

    print(f"[3/3] Computing per-row LogDist distances for {len(TCR_CT_SETS)} test sets:")
    for ts in TCR_CT_SETS:
        t2 = time.time()
        out_path = os.path.join(OUT_DIR, f'{ts}_dist.npy')
        if os.path.exists(out_path):
            print(f"    SKIP  {ts:<18} (already cached at {out_path})")
            continue
        test_df = load_test(ts)
        d = compute_combine_first_distances(
            test_df, train_df, CHAIN_COLS, weights,
            K_PARAM, B_PARAM, TOPK,
        )
        np.save(out_path, d)
        print(f"    OK    {ts:<18}  n={len(test_df):>5d}  "
              f"d_range=[{d.min():.3f}, {d.max():.3f}]  ({time.time()-t2:.1f}s)")

    print()
    print(f"Total: {time.time()-t0:.1f}s")
    print(f"Cache: {OUT_DIR}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
