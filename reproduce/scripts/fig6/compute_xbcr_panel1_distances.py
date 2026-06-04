#!/usr/bin/env python3
"""Re-derive `distance_cache_panel1.npz` (XBCR-net Panel 1 S2DD distances)
from raw inputs inside CaliPPer/.

✓ BIT-EXACT (verified 2026-06-04 by `reproduce_fig6.sh` Stage 0a):
    Regenerates `distance_cache_panel1.npz` from raw inputs with max
    |Δ| = 0.000000e+00 across all 1003 elements vs the staged copy.
    The historical 2026-05-30 "0.086 max |Δ|" note was based on an
    earlier transient state; the current implementation reproduces
    perfectly when invoked through the bash pipeline (clean INPUT_DIR
    + cache cleared).

    Used by `reproduce_fig6.sh` Stage 0a: writes
    `INPUT_DIR/results/xbcr_retrospective/distance_cache_panel1.npz`
    which Stage 1 (compute_fig6_recal_data.py) then consumes for
    Panel E XBCR ΔAUROC=+0.163 / ΔAP=+0.112 bit-exact reproduction.

This script reconstructs panel1 distances using the same 3-chain sigma_C
Levenshtein LogDist formula that `reproduce_fig6_xbcr.py` (already in
this dir) uses for the Panel 1 + Panel 2 distance computation.

Inputs (under INPUT_DIR):
    Data/retrospective_xbcr/extracted_panels/panel1_training.csv
        - Panel 1 training data (XBCR-net's authors' canonical training set)
    results/xbcr_retrospective/reproduction/test_predictions_original.csv
        - Panel 1 test predictions (authors' inference output; sequences
          are what we compute distances FROM)

Output (gitignored, regenerated each run):
    OUTPUT_DIR/xbcr_retrospective/distance_cache_panel1.npz
        - .npz with key 'distances', a 1-D array of length len(panel1_test)

Verification target:
    INPUT_DIR/results/xbcr_retrospective/distance_cache_panel1.npz
        - the cached version from research repo (staged in P4b.6 Stage 2)

Parameters (canonical fig6 XBCR settings per BUILD_PROGRESS and
reproduce_fig6_xbcr.py):
    chain_cols = ['Heavy', 'Light', 'variant_seq']
    k=0.1, b=0.03, K=30
    weight_formula = sigma_C

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/fig6/compute_xbcr_panel1_distances.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.combine_first_helpers import (
    compute_chain_weights,
    compute_combine_first_distances,
)

CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
K_PARAM = 0.1
B_PARAM = 0.03
TOPK = 30


def main() -> int:
    train_path = os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                               'extracted_panels', 'panel1_training.csv')
    panel1_path = os.path.join(INPUT_DIR, 'results', 'xbcr_retrospective',
                                'reproduction', 'test_predictions_original.csv')

    print(f"[xbcr_p1] Loading training data: {train_path}")
    train = pd.read_csv(train_path)
    train['Light'] = train['Light'].fillna('').astype(str)
    train['variant_seq'] = train['variant_seq'].fillna('').astype(str)
    train['Heavy'] = train['Heavy'].fillna('').astype(str)
    print(f"  n_train = {len(train)}")

    print(f"[xbcr_p1] Loading Panel 1 test data: {panel1_path}")
    panel1 = pd.read_csv(panel1_path)
    panel1['Light'] = panel1['Light'].fillna('').astype(str)
    panel1['variant_seq'] = panel1['variant_seq'].fillna('').astype(str)
    panel1['Heavy'] = panel1['Heavy'].fillna('').astype(str)
    print(f"  n_panel1 = {len(panel1)}")

    print(f"[xbcr_p1] Computing 3-chain sigma_C weights (k={K_PARAM}, b={B_PARAM}, K={TOPK})")
    weights, _ = compute_chain_weights(
        train, CHAIN_COLS, K_PARAM, B_PARAM, TOPK, formula='sigma_C')
    weights_str = ', '.join(f"{c}={w:.4f}" for c, w in zip(CHAIN_COLS, weights))
    print(f"  weights: {weights_str}")

    print(f"[xbcr_p1] Computing 3-chain Lev distances for Panel 1...")
    p1_dist = compute_combine_first_distances(
        panel1, train, CHAIN_COLS, weights, K_PARAM, B_PARAM, TOPK)
    print(f"  shape={p1_dist.shape}, range=[{p1_dist.min():.3f}, {p1_dist.max():.3f}], "
          f"mean={p1_dist.mean():.3f}")

    # Stage 0 of reproduce_fig6.sh: write to INPUT_DIR/results/ so Stage 1
    # (compute_fig6_recal_data.py) consumes the freshly-computed distances.
    out_dir = os.path.join(INPUT_DIR, 'results', 'xbcr_retrospective')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'distance_cache_panel1.npz')
    np.savez(out_path, distances=p1_dist)
    print(f"\nSaved: {out_path}")

    # Verify against cached version if present
    cached_path = os.path.join(INPUT_DIR, 'results', 'xbcr_retrospective',
                                'distance_cache_panel1.npz')
    if os.path.exists(cached_path):
        cached = np.load(cached_path)['distances']
        n = min(len(cached), len(p1_dist))
        max_abs_diff = float(np.abs(cached[:n] - p1_dist[:n]).max())
        print(f"\n=== Verification vs cached ===")
        print(f"  cached:       n={len(cached)}, range=[{cached.min():.3f}, {cached.max():.3f}]")
        print(f"  regenerated:  n={len(p1_dist)}, range=[{p1_dist.min():.3f}, {p1_dist.max():.3f}]")
        print(f"  max |diff| (first {n} elements): {max_abs_diff:.6e}")
        if max_abs_diff < 1e-10:
            print(f"  -> BIT-EXACT MATCH")
        elif max_abs_diff < 1e-4:
            print(f"  -> Close match within rounding")
        else:
            print(f"  -> Divergent. Investigate.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
