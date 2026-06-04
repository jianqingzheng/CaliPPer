#!/usr/bin/env python3
"""
BLOSUM-RF Cross-Test-Set LogDist Consistency Experiment.

Adapts eval_cross_test_logdist.py for the BLOSUM62 + Random Forest baseline.
In-process training (no subprocess, no conda).
Uses sigma_H chain weights for 3-chain LogDist.

Single model trained on combined training set, predictions on 5 test sets.
"""

import os
import sys
import time
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from Model.BLOSUM_RF.blosum_rf import train_and_predict

# Reuse from NetTCR cross-test pipeline
from eval_cross_test_logdist import (
    build_pools,
    compute_sigma_h_weights,
    compute_logdist_per_testset,
    evaluate_and_plot,
    verify_splits,
)


# ============================================================================
# Step 2: Prepare standard-format files
# ============================================================================

def prepare_standard_files(train_df, test_sets, splits_dir):
    """Convert to standard format, save CSVs.

    Returns:
        train_path, mega_test_path
    """
    print("\n[Step 2] Preparing standard-format files...")

    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }
    os.makedirs(splits_dir, exist_ok=True)

    # Train file
    train_std = train_df.rename(columns=col_map).copy()
    train_path = os.path.join(splits_dir, 'train.csv')
    train_std.to_csv(train_path, index=False)
    print(f"  train.csv: {len(train_std)} rows")

    # Per-test-set files + mega_test
    mega_parts = []
    for name, df in test_sets.items():
        test_std = df.rename(columns=col_map).copy()
        # Ensure CDR3a exists
        if 'CDR3a' not in test_std.columns:
            if 'cdr3_a' in df.columns:
                test_std['CDR3a'] = df['cdr3_a']
            else:
                test_std['CDR3a'] = ''
        # Ensure CDR3b exists
        if 'CDR3b' not in test_std.columns:
            if 'cdr3_b' in df.columns:
                test_std['CDR3b'] = df['cdr3_b']
            else:
                test_std['CDR3b'] = ''
        test_std['test_set_id'] = name

        test_path = os.path.join(splits_dir, f'{name}.csv')
        test_std.to_csv(test_path, index=False)
        print(f"  {name}.csv: {len(test_std)} rows")

        mega_parts.append(test_std)

    mega_test = pd.concat(mega_parts, ignore_index=True)
    mega_path = os.path.join(splits_dir, 'mega_test.csv')
    mega_test.to_csv(mega_path, index=False)
    print(f"  mega_test.csv: {len(mega_test)} rows")

    return train_path, mega_path


# ============================================================================
# Step 3: Train BLOSUM-RF ONCE + predict on mega_test
# ============================================================================

def train_and_predict_blosum_rf(train_path, mega_test_path, predictions_dir,
                                 train_epitopes, n_estimators, pca_variance,
                                 random_state):
    """Train BLOSUM-RF once on train, predict on mega_test, split by test_set_id.

    Returns:
        dict of test_set_name -> DataFrame with predictions
    """
    print("\n[Step 3] Training BLOSUM-RF (single run, all test sets)...")

    os.makedirs(predictions_dir, exist_ok=True)

    mega_pred_path = os.path.join(predictions_dir,
                                  'mega_test_predictions.csv')

    t0 = time.time()
    mega_pred, _ = train_and_predict(
        train_path, mega_test_path, mega_pred_path,
        pep_col='peptide', tcr_col='CDR3b', label_col='binder',
        n_estimators=n_estimators, pca_variance=pca_variance,
        random_state=random_state)
    elapsed = time.time() - t0

    print(f"  Training + prediction done in {elapsed:.1f}s")
    print(f"  mega_test_predictions.csv: {len(mega_pred)} rows")

    # Verify row count
    mega_test = pd.read_csv(mega_test_path)
    assert len(mega_pred) == len(mega_test), \
        (f"Row count mismatch: predictions={len(mega_pred)} "
         f"vs mega_test={len(mega_test)}")

    # Split by test_set_id
    pred_sets = {}
    for name, group in mega_pred.groupby('test_set_id'):
        group = group.copy().reset_index(drop=True)
        group['epitope_seen'] = group['peptide'].apply(
            lambda x: 1 if x in train_epitopes else 0)
        out_path = os.path.join(predictions_dir,
                                f'{name}_predictions_with_label.csv')
        group.to_csv(out_path, index=False)
        n_seen = (group['epitope_seen'] == 1).sum()
        n_unseen = (group['epitope_seen'] == 0).sum()
        print(f"  {name}: {len(group)} rows "
              f"({n_seen} seen + {n_unseen} unseen epitopes)")
        pred_sets[name] = group

    return pred_sets


def load_existing_predictions(predictions_dir, test_set_names, train_epitopes):
    """Load previously generated predictions (--skip-training mode)."""
    print("\n[Step 3] Loading existing predictions (skip-training mode)...")
    pred_sets = {}
    for name in test_set_names:
        path = os.path.join(predictions_dir,
                            f'{name}_predictions_with_label.csv')
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Remove --skip-training.")
            sys.exit(1)
        df = pd.read_csv(path)
        pred_sets[name] = df
        print(f"  {name}: {len(df)} rows")
    return pred_sets


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BLOSUM-RF Cross-Test-Set LogDist Consistency Experiment")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing data files')
    parser.add_argument('--output-dir', type=str,
                        default='results/blosum_rf/cross_test_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--pca-variance', type=float, default=0.90)
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, use existing predictions')
    args = parser.parse_args()

    t_start = time.time()
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    splits_dir = os.path.join(output_dir, 'splits')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    chain_cols = ['peptide', 'CDR3a', 'CDR3b']

    print(f"{'='*80}")
    print(f"BLOSUM-RF Cross-Test-Set LogDist Consistency Experiment")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  RF trees: {args.n_estimators}, PCA var: {args.pca_variance}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")

    # Step 1: Build pools (reused)
    train_df, test_sets, train_epitopes = build_pools(data_dir, args.seed)

    # Step 2: Prepare standard files
    train_path, mega_test_path = prepare_standard_files(
        train_df, test_sets, splits_dir)

    # Step 3: Train BLOSUM-RF + predict
    test_set_names = list(test_sets.keys())
    if not args.skip_training:
        pred_sets = train_and_predict_blosum_rf(
            train_path, mega_test_path, predictions_dir,
            train_epitopes, args.n_estimators, args.pca_variance,
            args.seed)
    else:
        pred_sets = load_existing_predictions(
            predictions_dir, test_set_names, train_epitopes)

    # Step 4: Compute sigma_H weights (reused)
    weights = compute_sigma_h_weights(
        train_path, chain_cols, args.k, args.b, args.K,
        subsample=500, seed=args.seed)

    # Step 5: Compute LogDist per test set (reused)
    result_sets = compute_logdist_per_testset(
        pred_sets, train_path, chain_cols, weights,
        args.k, args.b, args.K)

    # Step 6: Evaluate and plot (reused)
    test_results, corr_df = evaluate_and_plot(
        result_sets, train_epitopes, chain_cols, weights,
        args.K, args.k, args.b, args.bin_num, args.eval_metrics, output_dir)

    # Verification
    verify_splits(test_results, train_epitopes)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")
    print(f"  Output: {output_dir}")
    print(f"  Files:")
    print(f"    splits/                        — train, mega_test, per-test CSVs")
    print(f"    predictions/                   — per-test-set prediction CSVs")
    print(f"    cross_test_summary.csv         — all test set x metric correlations")
    print(f"    per_epitope_metrics.csv        — per-epitope performance")
    print(f"    cross_test_aucroc_overlay.png  — AUCROC curves overlay")
    print(f"    cross_test_bars.png            — bar chart comparison")
    print(f"    cross_test_all_metrics_grid.png — full metrics grid")
    print(f"    split_statistics.txt           — experiment statistics")


if __name__ == '__main__':
    main()
