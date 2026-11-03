#!/usr/bin/env python3
"""
TCR-BERT Cross-Test-Set LogDist Consistency Experiment.

Adapts eval_blosum_rf_cross_test_logdist.py for TCR-BERT (BERT embeddings + SVM).
Single model trained on combined training set, predictions on 5 test sets.
Uses sigma_H chain weights for 3-chain LogDist.

Test sets (same as NetTCR/ATM_TCR cross-test):
  A: seen_test      — 50% of folds 0,1,2 (seen epitopes only)
  B: unseen_fold34  — all folds 3,4 (unseen epitopes only)
  C: v3_combined    — v3 test + val (mixed seen/unseen)
  D: v4_combined    — v4 test + val (mixed seen/unseen)
  E: mcpas          — McPAS-TCR external database

TCR-BERT specifics:
  - Uses only CDR3b + epitope for model input
  - CDR3a preserved from original data for 3-chain LogDist
  - Requires GPU for BERT embedding extraction
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
# Step 2: Prepare standard-format files
# ============================================================================

def prepare_standard_files(train_df, test_sets, splits_dir, val_frac=0.1,
                           seed=42):
    """Convert to standard format, split validation for SVM training.

    Returns:
        train_path, val_path, mega_test_path, full_train_path, mega_full_df
    """
    print("\n[Step 2] Preparing standard-format files...")

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

    # Split a small validation set for SVM hyperparameter reporting
    from sklearn.model_selection import train_test_split
    train_main, train_val = train_test_split(
        train_full, test_size=val_frac, random_state=seed,
        stratify=train_full['binder'])
    train_main = train_main.reset_index(drop=True)
    train_val = train_val.reset_index(drop=True)
    print(f"  TCR-BERT split: {len(train_main)} train + "
          f"{len(train_val)} val")

    # Save train/val in model format (peptide, CDR3b, binder)
    train_model = train_main[['peptide', 'CDR3b', 'binder']].copy()
    train_model['binder'] = train_model['binder'].astype(int)
    train_path = os.path.join(splits_dir, 'tcrbert_train.csv')
    train_model.to_csv(train_path, index=False)

    val_model = train_val[['peptide', 'CDR3b', 'binder']].copy()
    val_model['binder'] = val_model['binder'].astype(int)
    val_path = os.path.join(splits_dir, 'tcrbert_val.csv')
    val_model.to_csv(val_path, index=False)

    # Per-test-set files + mega_test
    mega_parts = []
    for name, df in test_sets.items():
        test_full = df.rename(columns=col_map).copy()
        if 'CDR3a' not in test_full.columns:
            if 'cdr3_a' in df.columns:
                test_full['CDR3a'] = df['cdr3_a']
            else:
                test_full['CDR3a'] = ''
        test_full['test_set_id'] = name

        test_path = os.path.join(splits_dir, f'{name}.csv')
        test_full.to_csv(test_path, index=False)
        print(f"  {name}.csv: {len(test_full)} rows")

        mega_parts.append(test_full)

    mega_full = pd.concat(mega_parts, ignore_index=True)
    mega_path = os.path.join(splits_dir, 'mega_test_full.csv')
    mega_full.to_csv(mega_path, index=False)
    print(f"  mega_test_full.csv: {len(mega_full)} rows")

    # Model-format mega_test (peptide, CDR3b, binder only)
    mega_model = mega_full[['peptide', 'CDR3b', 'binder']].copy()
    mega_model['binder'] = mega_model['binder'].astype(int)
    mega_model_path = os.path.join(splits_dir, 'tcrbert_mega_test.csv')
    mega_model.to_csv(mega_model_path, index=False)

    return train_path, val_path, mega_model_path, full_train_path, mega_full


# ============================================================================
# Step 3: Train TCR-BERT+SVM ONCE + predict on mega_test
# ============================================================================

def train_and_predict_tcrbert(train_path, val_path, mega_test_path,
                               mega_full_df, predictions_dir,
                               train_epitopes, bert_model, n_pcs,
                               batch_size, device, seed):
    """Train TCR-BERT+SVM once, predict on mega_test, split by test_set_id.

    Returns:
        dict of test_set_name -> DataFrame with predictions
    """
    print("\n[Step 3] Training TCR-BERT+SVM (single run, all test sets)...")

    tcrbert_script = os.path.join(PROJECT_ROOT, 'Model', 'TCR_BERT',
                                   'tcrbert_svm.py')
    os.makedirs(predictions_dir, exist_ok=True)

    model_path = os.path.join(predictions_dir, 'tcrbert_cross_test.pkl')

    # Train
    cmd_train = [
        sys.executable, tcrbert_script,
        '--mode', 'train',
        '--train', train_path,
        '--val', val_path,
        '--model', model_path,
        '--bert-model', bert_model,
        '--n-pcs', str(n_pcs),
        '--batch-size', str(batch_size),
        '--device', str(device),
        '--seed', str(seed),
    ]

    print(f"  Command: {' '.join(cmd_train)}")
    t0 = time.time()
    result = subprocess.run(cmd_train, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR training TCR-BERT (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"    {line}")
    print(f"  Training done in {elapsed:.1f}s")

    # Predict on mega_test
    mega_pred_path = os.path.join(predictions_dir,
                                  'mega_test_predictions.csv')
    cmd_pred = [
        sys.executable, tcrbert_script,
        '--mode', 'predict',
        '--test', mega_test_path,
        '--model', model_path,
        '--output', mega_pred_path,
        '--bert-model', bert_model,
        '--batch-size', str(batch_size),
        '--device', str(device),
    ]

    result = subprocess.run(cmd_pred, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR predicting TCR-BERT (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"    {line}")

    # Read predictions and merge with full data (CDR3a, test_set_id)
    pred_raw = pd.read_csv(mega_pred_path)
    n_pred = len(pred_raw)
    n_mega = len(mega_full_df)

    assert n_pred == n_mega, \
        (f"Row count mismatch: predictions={n_pred} vs mega_test={n_mega}")

    # Row order preserved — merge prediction column into full data
    combined = mega_full_df.copy().reset_index(drop=True)
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
        description="TCR-BERT Cross-Test-Set LogDist Consistency Experiment")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing data files')
    parser.add_argument('--output-dir', type=str,
                        default='results/tcrbert/cross_test_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--bert-model', type=str, default='wukevin/tcr-bert',
                        help='TCR-BERT model name or path')
    parser.add_argument('--n-pcs', type=int, default=50,
                        help='Number of PCA components')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for BERT embedding extraction')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (-1 for CPU)')
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
    print(f"TCR-BERT Cross-Test-Set LogDist Consistency Experiment")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  BERT model: {args.bert_model}")
    print(f"  PCA components: {args.n_pcs}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")

    # Step 1: Build pools (reused)
    train_df, test_sets, train_epitopes = build_pools(data_dir, args.seed)

    # Step 2: Prepare standard files
    train_path, val_path, mega_test_path, full_train_path, mega_full_df = \
        prepare_standard_files(train_df, test_sets, splits_dir, seed=args.seed)

    # Step 3: Train TCR-BERT+SVM + predict
    test_set_names = list(test_sets.keys())
    if not args.skip_training:
        pred_sets = train_and_predict_tcrbert(
            train_path, val_path, mega_test_path, mega_full_df,
            predictions_dir, train_epitopes, args.bert_model, args.n_pcs,
            args.batch_size, args.device, args.seed)
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
