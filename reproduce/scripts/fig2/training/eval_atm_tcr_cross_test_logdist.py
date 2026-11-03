#!/usr/bin/env python3
"""
ATM_TCR Cross-Test-Set LogDist Consistency Experiment.

Adapts eval_cross_test_logdist.py for ATM_TCR. Holds the model constant
(single training) and varies only the test data across 5 test sets.
Uses sigma_H chain weights for 3-chain LogDist.

Test sets (same as NetTCR cross-test):
  A: seen_test      — 50% of folds 0,1,2 (seen epitopes only)
  B: unseen_fold34  — all folds 3,4 (unseen epitopes only)
  C: v3_combined    — v3 test + val (mixed seen/unseen)
  D: v4_combined    — v4 test + val (mixed seen/unseen)
  E: mcpas          — McPAS-TCR external database

ATM_TCR specifics:
  - Input: 3-col CSV (peptide, CDR3b, binder)
  - Output: TSV with pep_seq, tcr_seq, y_true, y_pred, score
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

# Reuse pool building, sigma_H weights, LogDist, evaluation, and plotting from NetTCR pipeline
from eval_cross_test_logdist import (
    build_pools,
    compute_sigma_h_weights,
    compute_logdist_per_testset,
    evaluate_and_plot,
    verify_splits,
)


# ============================================================================
# Step 2: Prepare ATM_TCR-format files (3-col: peptide, CDR3b, binder)
# ============================================================================

def prepare_atm_tcr_files(train_df, test_sets, splits_dir, val_frac=0.1, seed=42):
    """Convert to ATM_TCR column format, add test_set_id, save CSVs.

    Saves both ATM_TCR format (3-col) for training and full format for CDR3a.
    Splits a small validation set from training for ATM_TCR early stopping,
    ensuring ATM_TCR trains on the same data as other models (minus the
    small early-stopping holdout, same as NetTCR not having early stopping).

    Returns:
        atm_train_path, atm_val_path, atm_mega_test_path,
        full_train_path, mega_test_with_cdr3a
    """
    print("\n[Step 2] Preparing ATM_TCR-format files...")

    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }

    os.makedirs(splits_dir, exist_ok=True)

    # Full format train (for LogDist later — uses ALL training data)
    train_full = train_df.rename(columns=col_map).copy()
    full_train_path = os.path.join(splits_dir, 'train_full.csv')
    train_full.to_csv(full_train_path, index=False)

    # Split a small validation set for ATM_TCR early stopping
    from sklearn.model_selection import train_test_split
    train_main, train_val = train_test_split(
        train_full, test_size=val_frac, random_state=seed,
        stratify=train_full['binder'])
    train_main = train_main.reset_index(drop=True)
    train_val = train_val.reset_index(drop=True)
    print(f"  ATM_TCR split: {len(train_main)} train + "
          f"{len(train_val)} val (for early stopping)")

    # ATM_TCR 3-col train (main portion)
    atm_train = pd.DataFrame({
        'peptide': train_main['peptide'],
        'CDR3b': train_main['CDR3b'],
        'binder': train_main['binder'].astype(int),
    })
    atm_train_path = os.path.join(splits_dir, 'atm_train.csv')
    # Write WITHOUT header — ATM_TCR's read_pTCR reads positional columns
    atm_train.to_csv(atm_train_path, index=False, header=False)
    print(f"  atm_train.csv: {len(atm_train)} rows")

    # ATM_TCR 3-col val (for early stopping)
    atm_val = pd.DataFrame({
        'peptide': train_val['peptide'],
        'CDR3b': train_val['CDR3b'],
        'binder': train_val['binder'].astype(int),
    })
    atm_val_path = os.path.join(splits_dir, 'atm_val.csv')
    atm_val.to_csv(atm_val_path, index=False, header=False)
    print(f"  atm_val.csv: {len(atm_val)} rows (early stopping)")

    # Per-test-set files + mega_test
    mega_parts_atm = []
    mega_parts_full = []
    for name, df in test_sets.items():
        test_full = df.rename(columns=col_map).copy()

        # If CDR3a column doesn't exist (e.g., v4 data), fill with empty
        if 'CDR3a' not in test_full.columns:
            if 'cdr3_a' in df.columns:
                test_full['CDR3a'] = df['cdr3_a']
            else:
                test_full['CDR3a'] = ''

        test_full['test_set_id'] = name

        # ATM_TCR 3-col format
        atm_test = pd.DataFrame({
            'peptide': test_full['peptide'],
            'CDR3b': test_full['CDR3b'],
            'binder': test_full['binder'].astype(int),
            'test_set_id': name,
        })

        # Save individual test sets
        atm_path = os.path.join(splits_dir, f'atm_{name}.csv')
        atm_test[['peptide', 'CDR3b', 'binder']].to_csv(atm_path, index=False, header=False)
        print(f"  atm_{name}.csv: {len(atm_test)} rows")

        mega_parts_atm.append(atm_test)
        mega_parts_full.append(test_full)

    # Concatenate mega_test (ATM_TCR format)
    mega_atm = pd.concat(mega_parts_atm, ignore_index=True)
    atm_mega_path = os.path.join(splits_dir, 'atm_mega_test.csv')
    mega_atm[['peptide', 'CDR3b', 'binder']].to_csv(atm_mega_path, index=False, header=False)
    print(f"  atm_mega_test.csv: {len(mega_atm)} rows")

    # Full mega_test with CDR3a and test_set_id
    mega_full = pd.concat(mega_parts_full, ignore_index=True)
    mega_full_path = os.path.join(splits_dir, 'mega_test_full.csv')
    mega_full.to_csv(mega_full_path, index=False)

    return atm_train_path, atm_val_path, atm_mega_path, full_train_path, mega_full


# ============================================================================
# Step 3: Train ATM_TCR ONCE + predict on mega_test
# ============================================================================

def train_and_predict_atm_tcr(atm_train_path, atm_val_path, atm_mega_path,
                               mega_full_df, predictions_dir,
                               train_epitopes, epochs, batch_size):
    """Train ATM_TCR once on train, predict on mega_test.

    Uses --testfile for early stopping so ALL of --infile is used for training.

    Returns:
        dict of test_set_name -> DataFrame with predictions + CDR3a
    """
    print("\n[Step 3] Training ATM_TCR (single run, all test sets)...")

    atm_tcr_dir = os.path.join(PROJECT_ROOT, 'Model', 'ATM_TCR')
    atm_tcr_script = os.path.join(atm_tcr_dir, 'main.py')
    os.makedirs(predictions_dir, exist_ok=True)

    model_name = 'atm_cross_test.ckpt'

    cmd = [
        sys.executable, atm_tcr_script,
        '--infile', atm_train_path,
        '--testfile', atm_val_path,
        '--indepfile', atm_mega_path,
        '--mode', 'train',
        '--model_name', model_name,
        '--epoch', str(epochs),
        '--batch_size', str(batch_size),
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {atm_tcr_dir}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=atm_tcr_dir, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR training ATM_TCR (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    print(f"  Training + prediction done in {elapsed:.1f}s")

    # Read ATM_TCR output
    model_stem = os.path.splitext(os.path.basename(model_name))[0]
    test_basename = os.path.basename(atm_mega_path)
    pred_tsv = os.path.join(atm_tcr_dir, 'data',
                            f'pred_{model_stem}_{test_basename}')

    if not os.path.exists(pred_tsv):
        print(f"  ERROR: Prediction file not found: {pred_tsv}")
        sys.exit(1)

    pred_raw = pd.read_csv(pred_tsv, sep='\t', header=None,
                           names=['pep_seq', 'tcr_seq', 'y_true', 'y_pred', 'score'])
    print(f"  Read {len(pred_raw)} predictions")

    n_mega = len(mega_full_df)
    n_pred = len(pred_raw)

    # Handle batch edge case
    if n_pred == n_mega - 1 and n_mega % batch_size == 1:
        dropped_row = mega_full_df.iloc[-1]
        dropped_testset = dropped_row.get('test_set_id', 'unknown')
        print(f"  Batch edge case: {n_mega} -> {n_pred} "
              f"(last sample dropped from test set '{dropped_testset}')")
    elif n_pred != n_mega:
        print(f"  WARNING: Row count mismatch: predictions={n_pred} "
              f"vs mega_test={n_mega}")

    # Build CDR3a + test_set_id lookup from mega_full_df.
    # With shuffle=False in indep_loader, row order is preserved.
    # As a safety net, also build a sequence-based lookup for CDR3a.
    cdr3a_lookup = {}
    testset_lookup = {}
    for _, row in mega_full_df.iterrows():
        key = (row['peptide'], row['CDR3b'], int(row['binder']))
        cdr3a_lookup.setdefault(key, row.get('CDR3a', ''))
        # For test_set_id, we need row-order since same sequence
        # may appear in multiple test sets
        testset_lookup.setdefault(key, row.get('test_set_id', ''))

    # Build combined DataFrame using prediction output's own sequences.
    # With shuffle=False, row order matches mega_full_df, so we use
    # both row-order alignment (for test_set_id) and sequence matching
    # (for CDR3a, as a safety net).
    combined = mega_full_df.iloc[:n_pred].copy().reset_index(drop=True)
    combined['prediction'] = pred_raw['score'].values

    # Verify alignment: spot-check first 10 rows
    n_mismatched = 0
    for i in range(min(10, n_pred)):
        if combined['peptide'].iloc[i] != pred_raw['pep_seq'].iloc[i]:
            n_mismatched += 1
    if n_mismatched > 0:
        print(f"  WARNING: {n_mismatched}/10 spot-check rows mismatched, "
              f"falling back to sequence-based matching")
        # Fallback: build from prediction output directly
        combined = pd.DataFrame({
            'peptide': pred_raw['pep_seq'].values,
            'CDR3a': [cdr3a_lookup.get((p, t, int(b)), '')
                      for p, t, b in zip(pred_raw['pep_seq'],
                                         pred_raw['tcr_seq'],
                                         pred_raw['y_true'])],
            'CDR3b': pred_raw['tcr_seq'].values,
            'binder': pred_raw['y_true'].values.astype(int),
            'prediction': pred_raw['score'].values,
            'test_set_id': [testset_lookup.get((p, t, int(b)), 'unknown')
                            for p, t, b in zip(pred_raw['pep_seq'],
                                               pred_raw['tcr_seq'],
                                               pred_raw['y_true'])],
        })

    # Save mega predictions
    mega_pred_path = os.path.join(predictions_dir,
                                  'mega_test_predictions.csv')
    combined.to_csv(mega_pred_path, index=False)

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
        description="ATM_TCR Cross-Test-Set LogDist Consistency Experiment")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing data files')
    parser.add_argument('--output-dir', type=str,
                        default='results/atm_tcr/cross_test_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200,
                        help='ATM_TCR training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='ATM_TCR batch size')
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip ATM_TCR training, use existing predictions')
    args = parser.parse_args()

    t_start = time.time()
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    splits_dir = os.path.join(output_dir, 'splits')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    chain_cols = ['peptide', 'CDR3a', 'CDR3b']

    print(f"{'='*80}")
    print(f"ATM_TCR Cross-Test-Set LogDist Consistency Experiment")
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

    # Step 2: Prepare ATM_TCR files
    atm_train_path, atm_val_path, atm_mega_path, full_train_path, mega_full_df = \
        prepare_atm_tcr_files(train_df, test_sets, splits_dir, seed=args.seed)

    # Step 3: Train ATM_TCR + predict
    test_set_names = list(test_sets.keys())
    if not args.skip_training:
        pred_sets = train_and_predict_atm_tcr(
            atm_train_path, atm_val_path, atm_mega_path, mega_full_df,
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
