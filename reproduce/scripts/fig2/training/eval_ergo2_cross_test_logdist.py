#!/usr/bin/env python3
"""
ERGO-II Cross-Test-Set LogDist Consistency Experiment.

Adapts eval_atm_tcr_cross_test_logdist.py for ERGO-II (standalone LSTM).
Single model trained on combined training set, predictions on 5 test sets.
Uses sigma_H chain weights for 3-chain LogDist.

Test sets (same as NetTCR/ATM_TCR cross-test):
  A: seen_test      — 50% of folds 0,1,2 (seen epitopes only)
  B: unseen_fold34  — all folds 3,4 (unseen epitopes only)
  C: v3_combined    — v3 test + val (mixed seen/unseen)
  D: v4_combined    — v4 test + val (mixed seen/unseen)
  E: mcpas          — McPAS-TCR external database

ERGO-II specifics:
  - Input: CSV with columns peptide, CDR3b, binder (with header)
  - Output: CSV with peptide, CDR3b, binder, prediction columns
  - CDR3a preserved from original data for 3-chain LogDist
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Reuse from NetTCR cross-test pipeline
from eval_cross_test_logdist import (
    build_pools,
    compute_sigma_h_weights,
    compute_logdist_per_testset,
    evaluate_and_plot,
    verify_splits,
)


# ============================================================================
# Step 2: Prepare ERGO-II-format files
# ============================================================================

def prepare_ergo_files(train_df, test_sets, splits_dir, val_frac=0.1, seed=42):
    """Convert to ERGO-II format, split validation for early stopping.

    Returns:
        ergo_train_path, ergo_val_path, ergo_mega_test_path,
        full_train_path, mega_full_df
    """
    print("\n[Step 2] Preparing ERGO-II-format files...")

    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }

    os.makedirs(splits_dir, exist_ok=True)

    # Full format train (for LogDist — uses ALL training data)
    train_full = train_df.rename(columns=col_map).copy()
    full_train_path = os.path.join(splits_dir, 'train_full.csv')
    train_full.to_csv(full_train_path, index=False)

    # Split a small validation set for ERGO-II early stopping
    from sklearn.model_selection import train_test_split
    train_main, train_val = train_test_split(
        train_full, test_size=val_frac, random_state=seed,
        stratify=train_full['binder'])
    train_main = train_main.reset_index(drop=True)
    train_val = train_val.reset_index(drop=True)
    print(f"  ERGO-II split: {len(train_main)} train + "
          f"{len(train_val)} val (for early stopping)")

    # ERGO-II format (with header)
    ergo_train = train_main[['peptide', 'CDR3b', 'binder']].copy()
    ergo_train['binder'] = ergo_train['binder'].astype(int)
    ergo_train_path = os.path.join(splits_dir, 'ergo_train.csv')
    ergo_train.to_csv(ergo_train_path, index=False)

    ergo_val = train_val[['peptide', 'CDR3b', 'binder']].copy()
    ergo_val['binder'] = ergo_val['binder'].astype(int)
    ergo_val_path = os.path.join(splits_dir, 'ergo_val.csv')
    ergo_val.to_csv(ergo_val_path, index=False)

    # Per-test-set files + mega_test
    mega_parts_ergo = []
    mega_parts_full = []
    for name, df in test_sets.items():
        test_full = df.rename(columns=col_map).copy()
        if 'CDR3a' not in test_full.columns:
            if 'cdr3_a' in df.columns:
                test_full['CDR3a'] = df['cdr3_a']
            else:
                test_full['CDR3a'] = ''
        test_full['test_set_id'] = name

        ergo_test = pd.DataFrame({
            'peptide': test_full['peptide'],
            'CDR3b': test_full['CDR3b'],
            'binder': test_full['binder'].astype(int),
            'test_set_id': name,
        })

        # Save individual test sets
        ergo_path = os.path.join(splits_dir, f'ergo_{name}.csv')
        ergo_test[['peptide', 'CDR3b', 'binder']].to_csv(
            ergo_path, index=False)
        print(f"  ergo_{name}.csv: {len(ergo_test)} rows")

        mega_parts_ergo.append(ergo_test)
        mega_parts_full.append(test_full)

    # Concatenate mega_test
    mega_ergo = pd.concat(mega_parts_ergo, ignore_index=True)
    ergo_mega_path = os.path.join(splits_dir, 'ergo_mega_test.csv')
    mega_ergo[['peptide', 'CDR3b', 'binder']].to_csv(
        ergo_mega_path, index=False)
    print(f"  ergo_mega_test.csv: {len(mega_ergo)} rows")

    # Full mega_test with CDR3a and test_set_id
    mega_full = pd.concat(mega_parts_full, ignore_index=True)
    mega_full_path = os.path.join(splits_dir, 'mega_test_full.csv')
    mega_full.to_csv(mega_full_path, index=False)

    return ergo_train_path, ergo_val_path, ergo_mega_path, full_train_path, mega_full


# ============================================================================
# Step 3: Train ERGO-II ONCE + predict on mega_test
# ============================================================================

def train_and_predict_ergo(ergo_train_path, ergo_val_path, ergo_mega_path,
                            mega_full_df, predictions_dir,
                            train_epitopes, epochs, batch_size):
    """Train ERGO-II once on train, predict on mega_test.

    Returns:
        dict of test_set_name -> DataFrame with predictions + CDR3a
    """
    print("\n[Step 3] Training ERGO-II (single run, all test sets)...")

    ergo_script = os.path.join(PROJECT_ROOT, 'Model', 'ERGO_II', 'ergo_lstm.py')
    os.makedirs(predictions_dir, exist_ok=True)

    model_path = os.path.join(predictions_dir, 'ergo_cross_test.pt')

    # Train
    cmd_train = [
        sys.executable, ergo_script,
        '--mode', 'train',
        '--train', ergo_train_path,
        '--val', ergo_val_path,
        '--model', model_path,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
    ]

    print(f"  Command: {' '.join(cmd_train)}")
    t0 = time.time()
    result = subprocess.run(cmd_train, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR training ERGO-II (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"    {line}")
    print(f"  Training done in {elapsed:.1f}s")

    # Predict on mega_test
    mega_pred_path = os.path.join(predictions_dir, 'mega_test_predictions.csv')
    cmd_pred = [
        sys.executable, ergo_script,
        '--mode', 'predict',
        '--test', ergo_mega_path,
        '--model', model_path,
        '--output', mega_pred_path,
        '--batch-size', str(batch_size),
    ]

    result = subprocess.run(cmd_pred, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR predicting ERGO-II (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    # Read predictions and merge with full data (CDR3a, test_set_id)
    pred_raw = pd.read_csv(mega_pred_path)
    n_pred = len(pred_raw)
    n_mega = len(mega_full_df)

    if n_pred != n_mega:
        print(f"  WARNING: Row count mismatch: predictions={n_pred} "
              f"vs mega_test={n_mega}")

    # Row order preserved (shuffle=False in predict), merge by position
    combined = mega_full_df.iloc[:n_pred].copy().reset_index(drop=True)
    combined['prediction'] = pred_raw['prediction'].values

    # Save mega predictions with full info
    mega_full_pred_path = os.path.join(predictions_dir,
                                        'mega_test_predictions_full.csv')
    combined.to_csv(mega_full_pred_path, index=False)

    # Split by test_set_id
    pred_sets = {}
    for name, group in combined.groupby('test_set_id'):
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
        description="ERGO-II Cross-Test-Set LogDist Consistency Experiment")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing data files')
    parser.add_argument('--output-dir', type=str,
                        default='results/ergo_ii/cross_test_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50,
                        help='ERGO-II training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ERGO-II batch size')
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
    print(f"ERGO-II Cross-Test-Set LogDist Consistency Experiment")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")

    # Step 1: Build pools (reused from NetTCR pipeline)
    train_df, test_sets, train_epitopes = build_pools(data_dir, args.seed)

    # Step 2: Prepare ERGO-II files
    ergo_train_path, ergo_val_path, ergo_mega_path, full_train_path, mega_full_df = \
        prepare_ergo_files(train_df, test_sets, splits_dir, seed=args.seed)

    # Step 3: Train ERGO-II + predict
    test_set_names = list(test_sets.keys())
    if not args.skip_training:
        pred_sets = train_and_predict_ergo(
            ergo_train_path, ergo_val_path, ergo_mega_path, mega_full_df,
            predictions_dir, train_epitopes, args.epochs, args.batch_size)
    else:
        pred_sets = load_existing_predictions(
            predictions_dir, test_set_names, train_epitopes)

    # Step 4: Compute sigma_H weights (reused)
    weights = compute_sigma_h_weights(
        full_train_path, chain_cols, args.k, args.b, args.K,
        subsample=500, seed=args.seed)

    # Step 5: Compute LogDist per test set (reused)
    result_sets = compute_logdist_per_testset(
        pred_sets, full_train_path, chain_cols, weights,
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
