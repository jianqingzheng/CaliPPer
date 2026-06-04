#!/usr/bin/env python3
"""
BCR Binding — Combined SARS+Flu Antibody-Stratified CV with XBCR-net.

Extends eval_bcr_bind_ab_stratified.py by pooling SARS-CoV-2 RBD binding data
with influenza HA binding data to create a more antigen-diverse dataset.

Hypothesis: Greater antigen diversity should strengthen distance-performance
degradation patterns (S2DD correlations).

Pipeline: pool SARS + flu binding data → create antibody-stratified 5-fold CV
splits → prepare XBCR-net training data per fold → train XBCR-net → run
inference → collect predictions → evaluate LogDist degradation.

Note: Flu HA antigens (~566 AA) exceed XBCR-net's 300 AA one-hot encoding
limit. The first 290 AA (covering HA1 receptor binding domain) are kept via
truncation in prepare_xbcrnet_data().
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from General_Eval.general_evaluator import (
    safe_metric,
    binned_correlations,
)
from General_Eval.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
)

# Reuse all pipeline functions from eval_bcr_bind_ab_stratified
from eval_bcr_bind_ab_stratified import (
    _flush,
    load_binding_data,
    create_antibody_stratified_splits,
    prepare_xbcrnet_data,
    train_xbcrnet_fold,
    infer_xbcrnet_fold,
    collect_predictions,
    evaluate_fold_degradation,
    plot_cv_degradation_grid,
    plot_cv_summary_bars,
    plot_seen_unseen_heavy,
    plot_seen_unseen_heavy_overlay,
    save_seen_unseen_heavy_summary,
    print_cv_summary,
)


# ============================================================================
# Load combined SARS + flu data
# ============================================================================

def load_combined_data(data_root, include_hiv=False):
    """Load and pool SARS-CoV-2 RBD + influenza HA (+ optionally HIV) binding data.

    SARS data: from Data/bcr_seq/XBCR_net_binding/ (5 sources, ~4685 samples)
    Flu data:  from Data/bcr_seq/flu_bind/fold{0-4}/ (pool all folds, ~5545 unique)
    HIV data:  from Data/bcr_seq/hiv_bind/fold{0-4}/ (pool all folds, ~57K unique)

    Returns a DataFrame with columns:
        Heavy, Light, variant_seq, rbd, not_rbd, source, ab, data_source
    where data_source is 'sars', 'flu', or 'hiv'.
    """
    # Load SARS data using existing function
    print("  --- SARS-CoV-2 RBD binding data ---")
    sars_df = load_binding_data(data_root)
    sars_df['data_source'] = 'sars'

    # Load flu data from all fold files
    print("\n  --- Influenza HA binding data ---")
    flu_dir = os.path.join(data_root, 'Data', 'bcr_seq', 'flu_bind')
    flu_frames = []
    for fold in range(5):
        for split in ['train', 'test']:
            path = os.path.join(flu_dir, f'fold{fold}', f'{split}.csv')
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            df = pd.read_csv(path)
            flu_frames.append(df)

    if not flu_frames:
        print("  ERROR: No flu data found!")
        return sars_df

    flu_all = pd.concat(flu_frames, ignore_index=True)

    # Deduplicate flu data (same sample appears in multiple folds)
    flu_dedup = flu_all.drop_duplicates(
        subset=['Heavy', 'Light', 'variant_seq', 'rbd']
    ).reset_index(drop=True)

    # Standardize flu data to match SARS format
    flu_df = pd.DataFrame({
        'Heavy': flu_dedup['Heavy'].astype(str),
        'Light': flu_dedup['Light'].astype(str),
        'variant_seq': flu_dedup['variant_seq'].astype(str),
        'rbd': flu_dedup['rbd'].astype(int),
        'not_rbd': (1 - flu_dedup['rbd']).astype(int),
        'source': 'flu',
        'data_source': 'flu',
    })

    # Clean sequences
    for col in ['Heavy', 'Light', 'variant_seq']:
        flu_df[col] = flu_df[col].str.replace(r'[\s_|><=-]', '', regex=True)

    # Antibody identity
    flu_df['ab'] = flu_df['Heavy'] + '|' + flu_df['Light']

    n_rbd = (flu_df['rbd'] == 1).sum()
    n_notrbd = (flu_df['not_rbd'] == 1).sum()
    n_abs = flu_df['ab'].nunique()
    n_vars = flu_df['variant_seq'].nunique()
    print(f"  Flu pooled (deduplicated): {len(flu_df)} samples, "
          f"{n_vars} unique antigens, {n_abs} unique antibodies")
    print(f"  Flu positives (rbd=1): {n_rbd}, negatives: {n_notrbd}")

    frames = [sars_df, flu_df]

    # Optionally load HIV data
    if include_hiv:
        print("\n  --- HIV gp120 binding data ---")
        hiv_dir = os.path.join(data_root, 'Data', 'bcr_seq', 'hiv_bind')
        hiv_frames = []
        for fold in range(5):
            for split in ['train', 'test']:
                path = os.path.join(hiv_dir, f'fold{fold}', f'{split}.csv')
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                hiv_frames.append(df)

        if not hiv_frames:
            print("  WARNING: No HIV data found, skipping")
        else:
            hiv_all = pd.concat(hiv_frames, ignore_index=True)
            hiv_dedup = hiv_all.drop_duplicates(
                subset=['Heavy', 'Light', 'variant_seq', 'rbd']
            ).reset_index(drop=True)

            hiv_df = pd.DataFrame({
                'Heavy': hiv_dedup['Heavy'].astype(str),
                'Light': hiv_dedup['Light'].astype(str),
                'variant_seq': hiv_dedup['variant_seq'].astype(str),
                'rbd': hiv_dedup['rbd'].astype(int),
                'not_rbd': (1 - hiv_dedup['rbd']).astype(int),
                'source': 'hiv',
                'data_source': 'hiv',
            })

            # Clean sequences
            for col in ['Heavy', 'Light', 'variant_seq']:
                hiv_df[col] = hiv_df[col].str.replace(
                    r'[\s_|><=-]', '', regex=True)
            hiv_df['ab'] = hiv_df['Heavy'] + '|' + hiv_df['Light']

            n_rbd = (hiv_df['rbd'] == 1).sum()
            n_notrbd = (hiv_df['not_rbd'] == 1).sum()
            n_abs = hiv_df['ab'].nunique()
            n_vars = hiv_df['variant_seq'].nunique()
            print(f"  HIV pooled (deduplicated): {len(hiv_df)} samples, "
                  f"{n_vars} unique antigens, {n_abs} unique antibodies")
            print(f"  HIV positives (rbd=1): {n_rbd}, negatives: {n_notrbd}")
            print(f"  HIV variant_seq mean length: "
                  f"{hiv_df['variant_seq'].str.len().mean():.0f} AA")
            frames.append(hiv_df)

    # Combine all sources
    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate across sources (unlikely overlap but be safe)
    n_before = len(combined)
    combined = combined.drop_duplicates(
        subset=['Heavy', 'Light', 'variant_seq', 'rbd']
    ).reset_index(drop=True)
    n_after = len(combined)
    n_cross_dup = n_before - n_after
    if n_cross_dup > 0:
        print(f"  Removed {n_cross_dup} cross-source duplicates")

    # Recompute ab column after dedup
    combined['ab'] = combined['Heavy'] + '|' + combined['Light']

    # Summary
    source_counts = {src: (combined['data_source'] == src).sum()
                     for src in combined['data_source'].unique()}
    n_rbd_total = (combined['rbd'] == 1).sum()
    n_notrbd_total = (combined['not_rbd'] == 1).sum()
    n_abs_total = combined['ab'].nunique()
    n_vars_total = combined['variant_seq'].nunique()

    src_str = ', '.join(f'{k}={v}' for k, v in source_counts.items())
    print(f"\n  Combined dataset:")
    print(f"    Total: {len(combined)} samples ({src_str})")
    print(f"    Unique antibodies: {n_abs_total}")
    print(f"    Unique antigens: {n_vars_total}")
    print(f"    RBD binders: {n_rbd_total}, non-RBD: {n_notrbd_total}")
    print(f"    Positive rate: {n_rbd_total / len(combined):.3f}")

    return combined


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BCR Binding — Combined SARS+Flu Antibody-Stratified CV')
    parser.add_argument('--xbcrnet-dir', type=str,
                        default='Model/XBCR-net',
                        help='Path to XBCR-net directory')
    parser.add_argument('--output-dir', type=str,
                        default='results/xbcr/combined_bind_ab_cv',
                        help='Output directory')
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.03)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'])
    parser.add_argument('--chain-cols', nargs='+',
                        default=['Heavy', 'Light'],
                        help='Chain columns for LogDist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip XBCR-net training (use existing predictions)')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Do not restore pretrained binding model')
    parser.add_argument('--data-prefix', type=str, default=None,
                        help='Prefix for XBCR-net data_name (default: '
                             'combined_bind_nopt)')
    parser.add_argument('--folds-to-run', nargs='+', type=int, default=None,
                        help='Run only these fold indices')
    parser.add_argument('--include-hiv', action='store_true',
                        help='Include HIV gp120 binding data (3-pathogen)')
    args = parser.parse_args()

    t_start = time.time()

    xbcrnet_dir = os.path.join(PROJECT_ROOT, args.xbcrnet_dir) \
        if not os.path.isabs(args.xbcrnet_dir) else args.xbcrnet_dir
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    chain_cols = args.chain_cols
    restore_pretrain = 0 if args.no_pretrain else 1

    # Data prefix: always use combined prefix
    if args.data_prefix is not None:
        data_prefix = args.data_prefix
    elif args.no_pretrain:
        data_prefix = 'combined_bind_nopt'
    else:
        data_prefix = 'combined_bind'

    fold_indices = args.folds_to_run if args.folds_to_run is not None \
        else list(range(args.n_folds))

    print(f"{'='*80}")
    print(f"BCR Binding — Combined SARS+Flu Antibody-Stratified CV with XBCR-net")
    print(f"{'='*80}")
    print(f"  XBCR-net: {xbcrnet_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}, bins={args.bin_num}")
    print(f"  Folds: {args.n_folds} total, running: {fold_indices}")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Chains: {' + '.join(chain_cols)}")
    print(f"  Pretrain: {'yes' if restore_pretrain else 'no'}")
    print(f"  Data prefix: {data_prefix}")
    print(f"  Skip training: {args.skip_training}")
    _flush()

    # ====================================================================
    # Step 1: Load combined binding data
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Loading combined SARS + flu binding data")
    print(f"{'='*60}")
    _flush()
    pooled = load_combined_data(PROJECT_ROOT, include_hiv=args.include_hiv)

    # Save source breakdown
    source_counts = pooled.groupby('data_source').agg(
        n_samples=('rbd', 'size'),
        n_abs=('ab', 'nunique'),
        n_vars=('variant_seq', 'nunique'),
        pos_rate=('rbd', 'mean'),
    ).round(3)
    source_counts.to_csv(os.path.join(output_dir, 'source_breakdown.csv'))
    print(f"\n  Source breakdown saved to {output_dir}/source_breakdown.csv")

    # ====================================================================
    # Step 2: Create antibody-stratified CV splits
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Creating antibody-stratified {args.n_folds}-fold splits")
    print(f"{'='*60}")
    _flush()
    ab_assignments, folds = create_antibody_stratified_splits(
        pooled, n_folds=args.n_folds, seed=args.seed)

    # Save assignments
    ab_assignments.to_csv(
        os.path.join(output_dir, 'ab_fold_assignments.csv'), index=False)

    # Print per-fold source breakdown
    print(f"\n  Per-fold source breakdown:")
    for k_fold in fold_indices:
        tr, te = folds[k_fold]
        if 'data_source' in tr.columns:
            tr_counts = tr['data_source'].value_counts().to_dict()
            te_counts = te['data_source'].value_counts().to_dict()
            tr_str = ' '.join(f'{k}={v}' for k, v in sorted(tr_counts.items()))
            te_str = ' '.join(f'{k}={v}' for k, v in sorted(te_counts.items()))
            print(f"    Fold {k_fold}: train [{tr_str}], test [{te_str}]")

    # Save fold train/test CSVs
    for k_fold in fold_indices:
        tr, te = folds[k_fold]
        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        tr.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

    if not args.skip_training:
        # ====================================================================
        # Step 3: Prepare XBCR-net training data
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 3: Preparing XBCR-net data per fold")
        print(f"{'='*60}")
        _flush()
        for k_fold in fold_indices:
            print(f"\n  --- Fold {k_fold} ---")
            train_df, test_df = folds[k_fold]
            prepare_xbcrnet_data(k_fold, train_df, test_df, xbcrnet_dir,
                                 data_prefix=data_prefix)

        # ====================================================================
        # Step 4: Train XBCR-net per fold
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 4: Training XBCR-net per fold ({args.max_epochs} epochs)")
        print(f"{'='*60}")
        _flush()
        for k_fold in fold_indices:
            t_fold = time.time()
            success = train_xbcrnet_fold(
                k_fold, xbcrnet_dir, args.max_epochs,
                restore_pretrain=restore_pretrain,
                data_prefix=data_prefix)
            elapsed = time.time() - t_fold
            status = 'OK' if success else 'FAILED'
            print(f"  Fold {k_fold}: {status} ({elapsed:.1f}s)")
            _flush()

        # ====================================================================
        # Step 5: Run inference per fold
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 5: Running XBCR-net inference per fold")
        print(f"{'='*60}")
        _flush()

    # ====================================================================
    # Step 6: Collect predictions
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 6: Collecting predictions")
    print(f"{'='*60}")
    _flush()

    folds_with_preds = []
    for k_fold in fold_indices:
        train_df, test_df = folds[k_fold]

        # Check existing predictions
        existing_test = os.path.join(output_dir, f'fold{k_fold}', 'test.csv')
        if args.skip_training and os.path.exists(existing_test):
            test_with_pred = pd.read_csv(existing_test)
            if 'pred_prob' in test_with_pred.columns:
                if 'heavy_seen' not in test_with_pred.columns:
                    train_heavies = set(train_df['Heavy'].unique())
                    test_with_pred['heavy_seen'] = test_with_pred['Heavy'].apply(
                        lambda h: 1 if h in train_heavies else 0)
                print(f"  Fold {k_fold}: loaded existing predictions "
                      f"({len(test_with_pred)} rows)")
                folds_with_preds.append((k_fold, train_df, test_with_pred))
                continue

        # Try XBCR-net results
        data_name = f'{data_prefix}_fold{k_fold}'
        result_path = os.path.join(
            xbcrnet_dir, 'data', data_name, 'test', 'results',
            f'results_rbd_XBCR_net-{k_fold}.xlsx')
        if os.path.exists(result_path):
            pred_df = pd.read_excel(result_path)
            print(f"  Fold {k_fold}: loaded XBCR-net output "
                  f"({len(pred_df)} rows)")
        else:
            # Run inference
            model_path = os.path.join(
                xbcrnet_dir, 'models', data_name,
                f'{data_name}-XBCR_net',
                f'model_rbd_{k_fold}.tf.index')
            if not os.path.exists(model_path):
                print(f"  ERROR: no model or predictions for fold {k_fold}")
                continue
            prepare_xbcrnet_data(k_fold, train_df, test_df, xbcrnet_dir,
                                 data_prefix=data_prefix)
            pred_df = infer_xbcrnet_fold(k_fold, xbcrnet_dir,
                                          data_prefix=data_prefix)

        if pred_df is None:
            print(f"  Skipping fold {k_fold}: inference failed")
            continue

        test_with_pred = collect_predictions(k_fold, pred_df, test_df, train_df)

        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        test_with_pred.to_csv(
            os.path.join(fold_dir, 'test.csv'), index=False)

        folds_with_preds.append((k_fold, train_df, test_with_pred))

    if not folds_with_preds:
        print("\nERROR: No folds with predictions available. Exiting.")
        sys.exit(1)

    n_actual_folds = len(folds_with_preds)
    print(f"\n  Collected predictions for {n_actual_folds}/{args.n_folds} folds")
    _flush()

    # Quick AUROC check
    print(f"\n  Overall AUROC per fold:")
    for k_fold, train_df, test_df in folds_with_preds:
        auroc = safe_metric('aucroc', test_df['rbd'].values,
                            test_df['pred_prob'].values)
        n_heavy_seen = (test_df['heavy_seen'] == 1).sum()
        n_heavy_unseen = (test_df['heavy_seen'] == 0).sum()
        print(f"    Fold {k_fold}: AUROC={auroc:.4f}, N={len(test_df)}, "
              f"heavy_seen={n_heavy_seen}, heavy_unseen={n_heavy_unseen}")

    # ====================================================================
    # Step 7: LogDist degradation analysis
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 7: LogDist degradation analysis")
    print(f"{'='*60}")
    _flush()

    all_rows = []
    for k_fold, train_df, test_df in folds_with_preds:
        t_eval = time.time()
        rows, test_with_dist = evaluate_fold_degradation(
            k_fold, train_df, test_df, chain_cols,
            args.k, args.b, args.K, args.bin_num,
            args.eval_metrics, seed=args.seed)
        all_rows.extend(rows)

        # Save test.csv with distance column
        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        test_with_dist.to_csv(
            os.path.join(fold_dir, 'test.csv'), index=False)

        print(f"  Fold {k_fold} evaluated in {time.time() - t_eval:.1f}s")
        _flush()

    results_df = pd.DataFrame(all_rows)

    # Print summary
    print_cv_summary(results_df, n_actual_folds, args.eval_metrics)

    # Save CSV
    save_df = results_df.drop(columns=['bin_dists', 'bin_perfs'])
    save_df.to_csv(os.path.join(output_dir, 'cv_evaluation_summary.csv'),
                   index=False)

    # Plots (with updated titles)
    print(f"\n  Generating CV plots...")
    plot_cv_degradation_grid(results_df, n_actual_folds, output_dir)
    plot_cv_summary_bars(results_df, n_actual_folds, output_dir)
    plot_seen_unseen_heavy(results_df, n_actual_folds, output_dir)
    plot_seen_unseen_heavy_overlay(results_df, n_actual_folds, output_dir)
    save_seen_unseen_heavy_summary(results_df, output_dir)

    # ====================================================================
    # Final summary
    # ====================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'='*80}")

    aucroc_combined = results_df[
        (results_df['method'] == 'topk') &
        (results_df['metric'] == 'aucroc') &
        (results_df['scope'] == 'combined')]
    rs = aucroc_combined['pearson_r'].dropna()
    if len(rs) > 0:
        n_pos = (rs > 0).sum()
        print(f"\n  AUCROC combined: mean r={rs.mean():.4f} "
              f"(std={rs.std():.4f}), #pos={n_pos}/{len(rs)}")

    for scope in ['seen_heavy', 'unseen_heavy']:
        sub = results_df[
            (results_df['method'] == 'topk') &
            (results_df['metric'] == 'aucroc') &
            (results_df['scope'] == scope)]
        rs_s = sub['pearson_r'].dropna()
        if len(rs_s) > 0:
            n_pos_s = (rs_s > 0).sum()
            print(f"  AUCROC {scope}: mean r={rs_s.mean():.4f} "
                  f"(std={rs_s.std():.4f}), #pos={n_pos_s}/{len(rs_s)}")

    print(f"\n  Output: {output_dir}")
    print(f"  Files:")
    print(f"    source_breakdown.csv")
    print(f"    ab_fold_assignments.csv")
    print(f"    cv_evaluation_summary.csv")
    print(f"    cv_degradation_grid.png")
    print(f"    cv_summary_bars.png")
    print(f"    seen_unseen_heavy_auroc.png")
    print(f"    seen_unseen_heavy_overlay.png")
    print(f"    seen_unseen_heavy_summary.csv")
    for k_fold, _, _ in folds_with_preds:
        print(f"    fold{k_fold}/train.csv, fold{k_fold}/test.csv")


if __name__ == '__main__':
    main()
