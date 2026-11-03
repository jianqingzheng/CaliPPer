#!/usr/bin/env python3
"""
BLOSUM-RF 5-Fold Cross-Validation LogDist Evaluation Pipeline.

Adapts eval_cv_folds_logdist.py for the BLOSUM62 + Random Forest baseline.
Much simpler than ATM_TCR/NetTCR: in-process training (no subprocess, no conda).

For each fold:
  1. Convert fold data to standard format (peptide, CDR3a, CDR3b, binder)
  2. Train BLOSUM-RF in-process -> predict on val and test
  3. Add epitope_seen labels
  4. Compute 3-chain LogDist at fixed (K, k, b) = (50, 0.1, 0.1)
  5. Compute binned and per-epitope correlations
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

from Model.BLOSUM_RF.blosum_rf import train_and_predict_multiple

# Reuse evaluation and plotting from the NetTCR pipeline
from eval_cv_folds_logdist import (
    evaluate_fold_logdist,
    compare_folds,
)


# ============================================================================
# Step 1: Prepare standard-format files for a fold
# ============================================================================

def prepare_standard_format(fold_dir, output_dir):
    """Convert fold CSVs to standard format (peptide, CDR3a, CDR3b, binder)."""
    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }
    paths = {}
    for src_name, dst_name in [('train_data.csv', 'train.csv'),
                                ('validation_data.csv', 'val.csv'),
                                ('test_data.csv', 'test.csv')]:
        df = pd.read_csv(os.path.join(fold_dir, src_name))
        std_df = df.rename(columns=col_map)
        out_path = os.path.join(output_dir, dst_name)
        std_df.to_csv(out_path, index=False)
        paths[dst_name] = out_path
        print(f"    {src_name} -> {dst_name} ({len(std_df)} rows)")
    return paths


# ============================================================================
# Step 2: Train BLOSUM-RF and get predictions (in-process)
# ============================================================================

def train_blosum_rf_fold(data_files, output_dir, n_estimators, pca_variance,
                         random_state):
    """Train BLOSUM-RF ONCE on fold's train, predict on val and test."""
    train_path = data_files['train.csv']

    test_csvs = {
        'val': data_files['val.csv'],
        'test': data_files['test.csv'],
    }
    output_csvs = {
        'val': os.path.join(output_dir, 'val_predictions.csv'),
        'test': os.path.join(output_dir, 'test_predictions.csv'),
    }

    print(f"    Training BLOSUM-RF once, predicting on val + test...")
    t0 = time.time()
    train_and_predict_multiple(
        train_path, test_csvs, output_csvs,
        pep_col='peptide', tcr_col='CDR3b', label_col='binder',
        n_estimators=n_estimators, pca_variance=pca_variance,
        random_state=random_state)
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s")


def add_epitope_seen_labels(output_dir, train_path):
    """Add epitope_seen column based on training epitopes."""
    train = pd.read_csv(train_path)
    train_epitopes = set(train['peptide'].unique())

    for name in ['val', 'test']:
        pred_path = os.path.join(output_dir, f'{name}_predictions.csv')
        pred_df = pd.read_csv(pred_path)
        pred_df['epitope_seen'] = pred_df['peptide'].apply(
            lambda x: 1 if x in train_epitopes else 0)
        out_path = os.path.join(output_dir,
                                f'{name}_predictions_with_label.csv')
        pred_df.to_csv(out_path, index=False)
        n_seen = (pred_df['epitope_seen'] == 1).sum()
        n_unseen = (pred_df['epitope_seen'] == 0).sum()
        print(f"    {name}: {n_seen} seen + {n_unseen} unseen = {len(pred_df)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BLOSUM-RF 5-Fold CV LogDist Evaluation Pipeline")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing tcr_cross_val_fold{0-4}/')
    parser.add_argument('--output-dir', type=str,
                        default='results/blosum_rf/cv_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Random Forest number of trees')
    parser.add_argument('--pca-variance', type=float, default=0.90,
                        help='PCA variance to retain')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'])
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, use existing predictions')
    args = parser.parse_args()

    t_start = time.time()
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)

    chain_cols = ['peptide', 'CDR3a', 'CDR3b']

    print(f"{'='*80}")
    print(f"BLOSUM-RF 5-Fold CV LogDist Evaluation")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  RF trees: {args.n_estimators}, PCA var: {args.pca_variance}")
    print(f"  Output: {output_dir}")

    all_fold_results = []

    for fold_id in range(args.n_folds):
        fold_dir = os.path.join(data_dir, f'tcr_cross_val_fold{fold_id}')
        fold_output = os.path.join(output_dir, f'fold{fold_id}')
        os.makedirs(fold_output, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"FOLD {fold_id}")
        print(f"{'='*80}")

        if not args.skip_training:
            # Step 1: Convert to standard format
            print(f"\n  [Step 1] Preparing standard format...")
            data_files = prepare_standard_format(fold_dir, fold_output)

            # Step 2: Train and predict (in-process)
            print(f"\n  [Step 2] Training BLOSUM-RF...")
            train_blosum_rf_fold(data_files, fold_output,
                                 args.n_estimators, args.pca_variance,
                                 args.seed)

            # Step 3: Add epitope_seen labels
            print(f"\n  [Step 3] Adding epitope_seen labels...")
            add_epitope_seen_labels(fold_output, data_files['train.csv'])
        else:
            for name in ['val_predictions_with_label.csv',
                         'test_predictions_with_label.csv', 'train.csv']:
                path = os.path.join(fold_output, name)
                if not os.path.exists(path):
                    print(f"  ERROR: {path} not found. "
                          f"Remove --skip-training to generate.")
                    sys.exit(1)

        # Step 4: LogDist evaluation (reused)
        res = evaluate_fold_logdist(
            fold_id, fold_output, chain_cols,
            args.K, args.k, args.b, args.bin_num, args.eval_metrics)
        all_fold_results.append(res)

        # Save per-fold results
        res['results_df'].to_csv(
            os.path.join(fold_output, 'logdist_correlations.csv'), index=False)
        res['ep_df'].to_csv(
            os.path.join(fold_output, 'per_epitope_metrics.csv'), index=False)

    # Step 5: Cross-fold comparison (reused)
    all_results = compare_folds(
        all_fold_results, args.eval_metrics,
        args.K, args.k, args.b, output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")
    print(f"  Output: {output_dir}")
    print(f"  Files:")
    print(f"    all_folds_correlations.csv    — all fold x metric correlations")
    print(f"    aucroc_binned_summary.csv     — AUCROC binned summary")
    print(f"    cv_fold_comparison_all_metrics.png — per-fold curves grid")
    print(f"    cv_aucroc_overlay.png         — AUCROC overlay across folds")
    print(f"    cv_unseen_epitope_scatter.png  — per-epitope scatter")
    for f in range(args.n_folds):
        print(f"    fold{f}/logdist_correlations.csv")
        print(f"    fold{f}/per_epitope_metrics.csv")


if __name__ == '__main__':
    main()
