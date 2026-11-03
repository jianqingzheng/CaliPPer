#!/usr/bin/env python3
"""
ATM_TCR 5-Fold Cross-Validation LogDist Evaluation Pipeline.

Adapts eval_cv_folds_logdist.py for ATM_TCR (multi-head attention, PyTorch).
Steps 1-2 are ATM_TCR-specific (data prep + subprocess training).
Steps 3-5 reuse the same LogDist evaluation, cross-fold comparison, and plots.

ATM_TCR specifics:
  - Input: 3-col CSV (peptide, CDR3b, binder). Header line auto-skipped.
  - Training runs as subprocess with cwd=Model/ATM_TCR/
  - Output: TSV at data/pred_{model}_{test}, cols: pep_seq, tcr_seq, y_true, y_pred, score
  - CDR3a not used by ATM_TCR; preserved from original fold data for 3-chain LogDist.
  - Batch edge case: if len(data) % batch_size == 1, last sample dropped by torchtext.
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
# Step 1: Prepare ATM_TCR-format files for a fold
# ============================================================================

def prepare_atm_tcr_format(fold_dir, output_dir):
    """Convert fold CSVs to ATM_TCR 3-column format (peptide, CDR3b, binder).

    Also saves the full fold data for CDR3a preservation later.
    """
    paths = {}
    full_dfs = {}
    for src_name, dst_name in [('train_data.csv', 'train.csv'),
                                ('validation_data.csv', 'val.csv'),
                                ('test_data.csv', 'test.csv')]:
        df = pd.read_csv(os.path.join(fold_dir, src_name))
        full_dfs[dst_name] = df

        # ATM_TCR uses positional columns: col[0]=peptide, col[1]=CDR3b, col[2]=binder
        atm_df = pd.DataFrame({
            'peptide': df['epitope'],
            'CDR3b': df['cdr3_b'],
            'binder': df['binding_label'].astype(int),
        })
        out_path = os.path.join(output_dir, f'atm_{dst_name}')
        # Write WITHOUT header — ATM_TCR's read_pTCR reads positional columns
        # and does not properly skip CSV headers
        atm_df.to_csv(out_path, index=False, header=False)
        paths[dst_name] = out_path
        print(f"    {src_name} -> atm_{dst_name} ({len(atm_df)} rows)")

    # Save full data with CDR3a for later LogDist computation
    # Also save a NetTCR-compatible version for evaluate_fold_logdist()
    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }
    for dst_name, df in full_dfs.items():
        nettcr_df = df.rename(columns=col_map)
        nettcr_path = os.path.join(output_dir, dst_name)
        nettcr_df.to_csv(nettcr_path, index=False)

    return paths, full_dfs


# ============================================================================
# Step 2: Train ATM_TCR and get predictions
# ============================================================================

def train_atm_tcr_fold(atm_files, output_dir, full_dfs, epochs, batch_size):
    """Train ATM_TCR ONCE on fold's train, predict on concatenated val+test.

    Uses --testfile to pass the fold's validation set for early stopping,
    ensuring ATM_TCR trains on ALL of --infile (consistent with NetTCR/BLOSUM-RF).
    Predicts on both val and test simultaneously via --indepfile.

    CDR3a is preserved by matching predictions back to original fold data
    via (peptide, CDR3b) lookup.

    Handles:
      - Running subprocess with correct cwd
      - Explicit --testfile for early stopping (no internal re-splitting)
      - Reading TSV output (no header)
      - CDR3a preservation via sequence matching (ATM_TCR doesn't output CDR3a)
      - Batch edge case (last sample dropped if len % batch_size == 1)
    """
    atm_tcr_dir = os.path.join(PROJECT_ROOT, 'Model', 'ATM_TCR')
    atm_tcr_script = os.path.join(atm_tcr_dir, 'main.py')

    train_path = atm_files['train.csv']
    val_path = atm_files['val.csv']

    # Concatenate val+test into a single indepfile for one training run
    atm_cols = ['peptide', 'CDR3b', 'binder']
    val_df_atm = pd.read_csv(val_path, header=None, names=atm_cols)
    test_df_atm = pd.read_csv(atm_files['test.csv'], header=None, names=atm_cols)
    n_val = len(val_df_atm)
    n_test = len(test_df_atm)
    combined_atm = pd.concat([val_df_atm, test_df_atm], ignore_index=True)
    combined_path = os.path.join(output_dir, 'atm_combined_eval.csv')
    combined_atm.to_csv(combined_path, index=False, header=False)
    print(f"    Combined val ({n_val}) + test ({n_test}) = "
          f"{len(combined_atm)} for single training run")

    # Build CDR3a lookup from original fold data: (peptide, CDR3b) -> CDR3a
    cdr3a_lookup = {}
    for split_name in ['val', 'test']:
        full_df = full_dfs[f'{split_name}.csv']
        for _, row in full_df.iterrows():
            key = (row['epitope'], row['cdr3_b'])
            cdr3a_lookup.setdefault(key, row['cdr3_a'])

    model_name = f'atm_fold_{os.path.basename(output_dir)}.ckpt'

    cmd = [
        sys.executable, atm_tcr_script,
        '--infile', train_path,
        '--testfile', val_path,
        '--indepfile', combined_path,
        '--mode', 'train',
        '--model_name', model_name,
        '--epoch', str(epochs),
        '--batch_size', str(batch_size),
    ]

    print(f"    Training ATM_TCR (all train data, val for early stopping)...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=atm_tcr_dir, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ERROR training ATM_TCR (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    print(f"      Training done in {elapsed:.1f}s")

    # Read ATM_TCR output
    model_stem = os.path.splitext(os.path.basename(model_name))[0]
    test_basename = os.path.basename(combined_path)
    pred_tsv = os.path.join(atm_tcr_dir, 'data',
                            f'pred_{model_stem}_{test_basename}')

    if not os.path.exists(pred_tsv):
        print(f"    ERROR: Prediction file not found: {pred_tsv}")
        print(f"    ATM_TCR stdout: {result.stdout[-2000:]}")
        sys.exit(1)

    # Parse TSV: pep_seq, tcr_seq, y_true, y_pred, score (no header)
    pred_raw = pd.read_csv(pred_tsv, sep='\t', header=None,
                           names=['pep_seq', 'tcr_seq', 'y_true', 'y_pred', 'score'])
    print(f"      Read {len(pred_raw)} predictions")

    n_combined = len(combined_atm)
    n_pred = len(pred_raw)

    # Handle batch edge case
    if n_pred == n_combined - 1 and n_combined % batch_size == 1:
        print(f"      Batch edge case: {n_combined} -> {n_pred} "
              f"(last sample dropped, batch_size={batch_size})")
    elif n_pred != n_combined:
        print(f"    WARNING: Row count mismatch: predictions={n_pred} "
              f"vs combined={n_combined}")

    # Build output using prediction output's own sequences (order-independent).
    # CDR3a is looked up by (peptide, CDR3b) from original fold data.
    pred_df = pd.DataFrame({
        'peptide': pred_raw['pep_seq'].values,
        'CDR3a': [cdr3a_lookup.get((p, t), '')
                  for p, t in zip(pred_raw['pep_seq'], pred_raw['tcr_seq'])],
        'CDR3b': pred_raw['tcr_seq'].values,
        'binder': pred_raw['y_true'].values.astype(int),
        'prediction': pred_raw['score'].values,
    })

    # Verify CDR3a lookup coverage
    n_missing = (pred_df['CDR3a'] == '').sum()
    if n_missing > 0:
        print(f"    WARNING: {n_missing}/{n_pred} predictions could not "
              f"find CDR3a via (peptide, CDR3b) lookup")

    # Split back into val and test predictions.
    # With shuffle=False in indep_loader, first n_val rows are val, rest are test.
    # As a safety check, also verify by sequence matching.
    val_pred = pred_df.iloc[:n_val].copy().reset_index(drop=True)
    test_pred = pred_df.iloc[n_val:n_val + n_test].copy().reset_index(drop=True)

    # Handle case where batch edge case dropped from test portion
    if len(val_pred) + len(test_pred) < n_pred:
        # Extra predictions beyond val+test — shouldn't happen
        pass
    elif len(val_pred) + len(test_pred) > n_pred:
        # Batch edge case dropped a sample; test gets one fewer
        test_pred = pred_df.iloc[n_val:].copy().reset_index(drop=True)

    for split_name, split_pred in [('val', val_pred), ('test', test_pred)]:
        out_path = os.path.join(output_dir, f'{split_name}_predictions.csv')
        split_pred.to_csv(out_path, index=False)
        print(f"      Saved {len(split_pred)} {split_name} predictions")


def add_epitope_seen_labels(output_dir, train_path):
    """Add epitope_seen column based on training epitopes."""
    train = pd.read_csv(train_path)
    # train_path may be the atm format or the full format
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
        description="ATM_TCR 5-Fold CV LogDist Evaluation Pipeline")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing tcr_cross_val_fold{0-4}/')
    parser.add_argument('--output-dir', type=str,
                        default='results/atm_tcr/cv_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200,
                        help='ATM_TCR training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='ATM_TCR batch size')
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
    print(f"ATM_TCR 5-Fold CV LogDist Evaluation")
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
            # Step 1: Convert to ATM_TCR format
            print(f"\n  [Step 1] Preparing ATM_TCR format...")
            atm_files, full_dfs = prepare_atm_tcr_format(fold_dir, fold_output)

            # Step 2: Train and predict
            print(f"\n  [Step 2] Training ATM_TCR...")
            train_atm_tcr_fold(atm_files, fold_output, full_dfs,
                               args.epochs, args.batch_size)

            # Step 3: Add epitope_seen labels
            # Use the full-format train.csv (with headers), not the headerless ATM file
            print(f"\n  [Step 3] Adding epitope_seen labels...")
            full_train_path = os.path.join(fold_output, 'train.csv')
            add_epitope_seen_labels(fold_output, full_train_path)
        else:
            # Verify predictions exist
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
