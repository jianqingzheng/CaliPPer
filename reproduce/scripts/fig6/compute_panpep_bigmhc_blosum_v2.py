#!/usr/bin/env python3
"""Compute BLOSUM-sqrt distances for PanPep + BigMHC with ACTUAL training data.

Fixes self-reference bug: uses train→test instead of test→test.

PanPep: majority_training_dataset (23K) → majority_test (5K) + zeroshot_test (857)
BigMHC: im_train (6185) → manafest (834)
"""
import os, sys, time
import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.pluggable_distance import compute_s2dd_pluggable, make_sw_blosum62_similarity
from calipper.combine_first_helpers import compute_chain_weights

# Stage 0 of reproduce_fig6.sh: read predictions from INPUT_DIR AND write the
# freshly-computed distance arrays back into INPUT_DIR/results/ so Stage 1
# (compute_fig6_recal_data.py) consumes the from-scratch distances.
RESULTS_IN = os.path.join(INPUT_DIR, 'results')
RESULTS = os.path.join(INPUT_DIR, 'results')
DATA = os.path.join(INPUT_DIR, 'Data')
sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)


def compute_and_save(train_df, test_df, chain_cols, k, b, K, out_dir, name, sim_cache):
    """Compute BLOSUM-sqrt distances from train→test and save."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(sim_cache, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_blosumsqrt_dist.npy')

    w, _ = compute_chain_weights(train_df, chain_cols, k, b, K, formula='sigma_C')
    print(f'  sigma_C: {", ".join(f"{c}={v:.4f}" for c, v in zip(chain_cols, w))}')
    print(f'  {name}: {len(test_df)} test × {len(train_df)} train, computing...', flush=True)
    t0 = time.time()

    d = compute_s2dd_pluggable(
        test_df, train_df, chain_cols, w,
        similarity_fn=sw_sim, k=k, b=b, K=K,
        cache_prefix=name, cache_dir=sim_cache,
        transform='sqrt', verbose=True)
    np.save(out_path, d)
    print(f'  Saved: {out_path} ({len(d)} samples, {time.time()-t0:.0f}s)')
    return d


def compute_panpep():
    """PanPep: majority_training (23K) → majority_test (5K) + zeroshot (857)."""
    print("=" * 60)
    print("PanPep: BLOSUM-sqrt (train → test, 2-chain)")
    print("=" * 60)

    train_path = os.path.join(DATA, 'tcr_seq', 'proc_files', 'deepantigen_data',
                              'majority', 'majority_training_dataset.csv')
    train_df = pd.read_csv(train_path)
    if len(train_df.columns) == 3:
        train_df.columns = ['peptide', 'binding_TCR', 'label']
    print(f'  Train: {len(train_df)} samples, {train_df["peptide"].nunique()} unique peptides, '
          f'{train_df["binding_TCR"].nunique()} unique CDR3b')

    chain_cols = ['peptide', 'binding_TCR']
    k, b, K = 0.1, 0.1, 50
    out_dir = os.path.join(RESULTS, 'panpep_retrospective', 'blosum_sqrt')
    sim_cache = os.path.join(out_dir, 'sim_cache_v2')

    # Majority test
    test_path = os.path.join(RESULTS_IN, 'panpep_retrospective', 'reproduction',
                             'majority_test_predictions.csv')
    test_df = pd.read_csv(test_path)
    print(f'  Test (majority): {len(test_df)} samples, '
          f'{test_df["peptide"].nunique()} unique peptides, '
          f'{test_df["binding_TCR"].nunique()} unique CDR3b')
    compute_and_save(train_df, test_df, chain_cols, k, b, K, out_dir,
                     'majority_test', sim_cache)

    # Zeroshot test (positive + negative combined)
    # Negatives are random peptide-CDR3 reshuffles (same CDR3, different peptide)
    zs_pos_path = os.path.join(RESULTS_IN, 'panpep_retrospective', 'reproduction',
                               'zeroshot_test_predictions.csv')
    zs_neg_path = os.path.join(RESULTS_IN, 'panpep_retrospective', 'reproduction',
                               'zeroshot_neg_predictions.csv')
    if os.path.exists(zs_pos_path) and os.path.exists(zs_neg_path):
        zs_pos = pd.read_csv(zs_pos_path)
        zs_neg = pd.read_csv(zs_neg_path)
        zs_combined = pd.concat([zs_pos, zs_neg], ignore_index=True)
        if all(c in zs_combined.columns for c in chain_cols):
            print(f'  Test (zeroshot combined): {len(zs_combined)} samples '
                  f'({len(zs_pos)} pos + {len(zs_neg)} neg)')
            compute_and_save(train_df, zs_combined, chain_cols, k, b, K, out_dir,
                             'zeroshot_test', sim_cache)

    print('PanPep complete.\n')


def compute_bigmhc():
    """BigMHC: im_train (6185) → manafest (834), 1-chain peptide."""
    print("=" * 60)
    print("BigMHC: BLOSUM-sqrt (im_train → manafest, 1-chain)")
    print("=" * 60)

    bm_dir = os.path.join(DATA, 'retrospective_bigmhc', 'mendeley_data', 'extracted',
                          'BigMHC Training and Evaluation Data')
    train_path = os.path.join(bm_dir, 'im_train.csv')
    train_df = pd.read_csv(train_path)
    print(f'  Train: {len(train_df)} samples, {train_df["pep"].nunique()} unique peptides')

    test_path = os.path.join(RESULTS_IN, 'bigmhc_retrospective', 'reproduction',
                             'manafest_with_distances.csv')
    test_df = pd.read_csv(test_path)
    print(f'  Test: {len(test_df)} samples, {test_df["pep"].nunique()} unique peptides')

    chain_cols = ['pep']
    k, b, K = 0.1, 0.1, 50
    out_dir = os.path.join(RESULTS, 'bigmhc_retrospective', 'blosum_sqrt')
    sim_cache = os.path.join(out_dir, 'sim_cache_v2')

    compute_and_save(train_df, test_df, chain_cols, k, b, K, out_dir,
                     'manafest', sim_cache)
    print('BigMHC complete.\n')


if __name__ == '__main__':
    compute_bigmhc()   # Fast (834 test × 6K train peptides)
    compute_panpep()   # Slower (5K test × 23K train, 2-chain)
    print('All done.')
