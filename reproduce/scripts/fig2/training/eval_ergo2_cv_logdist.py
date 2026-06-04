#!/usr/bin/env python3
"""
ERGO-II 5-Fold Cross-Validation LogDist Evaluation Pipeline.

Adapts eval_atm_tcr_cv_logdist.py for ERGO-II (standalone LSTM, PyTorch).
Steps 1-2 are ERGO-II-specific (data prep + subprocess training).
Steps 3-5 reuse the same LogDist evaluation, cross-fold comparison, and plots.

ERGO-II specifics:
  - Input: CSV with columns peptide, CDR3b, binder (with header).
  - Training runs as subprocess via Model/ERGO_II/ergo_lstm.py
  - Output: CSV with peptide, CDR3b, binder, prediction columns
  - CDR3a not used by ERGO-II; preserved from original fold data for 3-chain LogDist.
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Reuse evaluation and plotting from the NetTCR pipeline
from eval_cv_folds_logdist import (
    evaluate_fold_logdist,
    compare_folds,
)


# ============================================================================
# Step 1: Prepare ERGO-II-format files for a fold
# ============================================================================

def prepare_ergo_format(fold_dir, output_dir):
    """Convert fold CSVs to ERGO-II format (peptide, CDR3b, binder with header).

    Also saves full fold data with CDR3a for LogDist computation.
    """
    paths = {}
    for src_name, dst_name in [('train_data.csv', 'train.csv'),
                                ('validation_data.csv', 'val.csv'),
                                ('test_data.csv', 'test.csv')]:
        df = pd.read_csv(os.path.join(fold_dir, src_name))

        # ERGO-II format: peptide, CDR3b, binder (with header)
        ergo_df = pd.DataFrame({
            'peptide': df['epitope'],
            'CDR3b': df['cdr3_b'],
            'binder': df['binding_label'].astype(int),
        })
        out_path = os.path.join(output_dir, f'ergo_{dst_name}')
        ergo_df.to_csv(out_path, index=False)
        paths[dst_name] = out_path
        print(f"    {src_name} -> ergo_{dst_name} ({len(ergo_df)} rows)")

    # Save full data with CDR3a for LogDist (standard format)
    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }
    for src_name, dst_name in [('train_data.csv', 'train.csv'),
                                ('validation_data.csv', 'val.csv'),
                                ('test_data.csv', 'test.csv')]:
        df = pd.read_csv(os.path.join(fold_dir, src_name))
        nettcr_df = df.rename(columns=col_map)
        nettcr_path = os.path.join(output_dir, dst_name)
        nettcr_df.to_csv(nettcr_path, index=False)

    return paths


# ============================================================================
# Step 2: Train ERGO-II and get predictions
# ============================================================================

def train_ergo_fold(ergo_files, output_dir, full_fold_dir, epochs, batch_size):
    """Train ERGO-II ONCE on fold's train, predict on val and test.

    CDR3a is preserved by merging predictions back with original fold data.
    """
    ergo_script = os.path.join(PROJECT_ROOT, 'Model', 'ERGO_II', 'ergo_lstm.py')
    model_path = os.path.join(output_dir, 'ergo_model.pt')

    # Combine val+test for single prediction run
    val_df = pd.read_csv(ergo_files['val.csv'])
    test_df = pd.read_csv(ergo_files['test.csv'])
    n_val = len(val_df)
    n_test = len(test_df)
    combined = pd.concat([val_df, test_df], ignore_index=True)
    combined_path = os.path.join(output_dir, 'ergo_combined_eval.csv')
    combined.to_csv(combined_path, index=False)
    print(f"    Combined val ({n_val}) + test ({n_test}) = "
          f"{len(combined)} for prediction")

    # Build CDR3a lookup from original fold data
    cdr3a_lookup = {}
    for split_name in ['val.csv', 'test.csv']:
        full_df = pd.read_csv(os.path.join(output_dir, split_name))
        for _, row in full_df.iterrows():
            key = (row['peptide'], row['CDR3b'])
            cdr3a_lookup.setdefault(key, row.get('CDR3a', ''))

    # Train
    cmd_train = [
        sys.executable, ergo_script,
        '--mode', 'train',
        '--train', ergo_files['train.csv'],
        '--val', ergo_files['val.csv'],
        '--model', model_path,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
    ]

    print(f"    Training ERGO-II...")
    t0 = time.time()
    result = subprocess.run(cmd_train, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ERROR training ERGO-II (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    # Print training output
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"      {line}")
    print(f"      Training done in {elapsed:.1f}s")

    # Predict on combined val+test
    combined_pred_path = os.path.join(output_dir, 'ergo_combined_predictions.csv')
    cmd_pred = [
        sys.executable, ergo_script,
        '--mode', 'predict',
        '--test', combined_path,
        '--model', model_path,
        '--output', combined_pred_path,
        '--batch-size', str(batch_size),
    ]

    result = subprocess.run(cmd_pred, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR predicting ERGO-II (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    # Read predictions and add CDR3a
    pred_df = pd.read_csv(combined_pred_path)
    pred_df['CDR3a'] = [cdr3a_lookup.get((p, t), '')
                        for p, t in zip(pred_df['peptide'], pred_df['CDR3b'])]

    # Reorder columns to match expected format
    pred_df = pred_df[['peptide', 'CDR3a', 'CDR3b', 'binder', 'prediction']]

    # Verify CDR3a lookup coverage
    n_missing = (pred_df['CDR3a'] == '').sum()
    if n_missing > 0:
        print(f"    WARNING: {n_missing}/{len(pred_df)} predictions missing CDR3a")

    # Split back into val and test
    val_pred = pred_df.iloc[:n_val].copy().reset_index(drop=True)
    test_pred = pred_df.iloc[n_val:n_val + n_test].copy().reset_index(drop=True)

    for split_name, split_pred in [('val', val_pred), ('test', test_pred)]:
        out_path = os.path.join(output_dir, f'{split_name}_predictions.csv')
        split_pred.to_csv(out_path, index=False)
        print(f"      Saved {len(split_pred)} {split_name} predictions")


def add_epitope_seen_labels(output_dir, train_path):
    """Add epitope_seen column based on training epitopes."""
    train = pd.read_csv(train_path)
    pep_col = 'peptide' if 'peptide' in train.columns else 'epitope'
    train_epitopes = set(train[pep_col].unique())

    for name in ['val', 'test']:
        pred_path = os.path.join(output_dir, f'{name}_predictions.csv')
        pred_df = pd.read_csv(pred_path)
        pred_df['epitope_seen'] = pred_df['peptide'].apply(
            lambda x: 1 if x in train_epitopes else 0)
        out_path = os.path.join(output_dir, f'{name}_predictions_with_label.csv')
        pred_df.to_csv(out_path, index=False)
        n_seen = (pred_df['epitope_seen'] == 1).sum()
        n_unseen = (pred_df['epitope_seen'] == 0).sum()
        print(f"    {name}: {n_seen} seen + {n_unseen} unseen = {len(pred_df)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ERGO-II 5-Fold CV LogDist Evaluation Pipeline")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing tcr_cross_val_fold{0-4}/')
    parser.add_argument('--output-dir', type=str,
                        default='results/ergo_ii/cv_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50,
                        help='ERGO-II training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ERGO-II batch size')
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
    print(f"ERGO-II 5-Fold CV LogDist Evaluation")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
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
            # Step 1: Convert to ERGO-II format
            print(f"\n  [Step 1] Preparing ERGO-II format...")
            ergo_files = prepare_ergo_format(fold_dir, fold_output)

            # Step 2: Train and predict
            print(f"\n  [Step 2] Training ERGO-II...")
            train_ergo_fold(ergo_files, fold_output, fold_dir,
                            args.epochs, args.batch_size)

            # Step 3: Add epitope_seen labels
            print(f"\n  [Step 3] Adding epitope_seen labels...")
            full_train_path = os.path.join(fold_output, 'train.csv')
            add_epitope_seen_labels(fold_output, full_train_path)
        else:
            for name in ['val_predictions_with_label.csv',
                         'test_predictions_with_label.csv', 'train.csv']:
                path = os.path.join(fold_output, name)
                if not os.path.exists(path):
                    print(f"  ERROR: {path} not found. "
                          f"Remove --skip-training to generate.")
                    sys.exit(1)

        # Step 4: LogDist evaluation (reused from eval_cv_folds_logdist.py)
        res = evaluate_fold_logdist(
            fold_id, fold_output, chain_cols,
            args.K, args.k, args.b, args.bin_num, args.eval_metrics)
        all_fold_results.append(res)

        # Save per-fold results
        res['results_df'].to_csv(
            os.path.join(fold_output, 'logdist_correlations.csv'), index=False)
        res['ep_df'].to_csv(
            os.path.join(fold_output, 'per_epitope_metrics.csv'), index=False)

    # Step 5: Cross-fold comparison (reused from eval_cv_folds_logdist.py)
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
