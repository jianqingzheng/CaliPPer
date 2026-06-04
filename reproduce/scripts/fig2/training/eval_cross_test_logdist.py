#!/usr/bin/env python3
"""
Cross-Test-Set LogDist Consistency Experiment.

Verifies that the LogDist metric (sigma_H weights + topK) produces
consistent distance-performance correlations across different test sets when
evaluated against a SINGLE trained NetTCR model with a SINGLE set of weights.

Previous experiments varied both the model and test data (5-fold CV); this
experiment holds the model constant and varies only the test data.

Test sets:
  A: seen_test      — 50% of folds 0,1,2 (seen epitopes only)
  B: unseen_fold34  — all folds 3,4 (unseen epitopes only)
  C: v3_combined    — v3 test + val (mixed seen/unseen)
  D: v4_combined    — v4 test + val (mixed seen/unseen)

All test sets are concatenated into mega_test.csv and predicted by a single
NetTCR training run to guarantee identical model weights.
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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from General_Eval.general_evaluator import (
    compute_pairwise_ratios,
    logdist_from_ratios,
    compute_multichain_distances,
    safe_metric,
    binned_correlations,
)


# ============================================================================
# Step 1: Build train/test pools
# ============================================================================

def _dedup_against_train(test_df, train_df, key_cols):
    """Remove rows from test_df that also appear in train_df (by key_cols)."""
    train_keys = set(zip(*(train_df[c].astype(str) for c in key_cols)))
    mask = [
        tuple(str(row[c]) for c in key_cols) not in train_keys
        for _, row in test_df.iterrows()
    ]
    return test_df[mask].reset_index(drop=True)


def build_pools(data_dir, seed=42):
    """Build training set and up to 5 test sets from v3/v4 data + epitope folds.

    Returns:
        train_df, test_sets (dict of name -> df), train_epitopes (set)
    All DataFrames have original v3 columns (epitope, cdr3_a, cdr3_b, etc.).
    """
    print("\n[Step 1] Building train/test pools...")

    # Load epitope fold assignments
    fold_path = os.path.join(data_dir, 'epitope_fold_assignments.csv')
    fold_df = pd.read_csv(fold_path)
    ep_to_fold = dict(zip(fold_df['epitope'], fold_df['fold']))
    print(f"  Loaded {len(ep_to_fold)} epitope fold assignments")

    # Load all v3 data (train + val + test = 40,516 rows)
    v3_dir = os.path.join(data_dir, 'tcr_ml_stratified_v3')
    v3_train = pd.read_csv(os.path.join(v3_dir, 'train_data.csv'))
    v3_val = pd.read_csv(os.path.join(v3_dir, 'validation_data.csv'))
    v3_test = pd.read_csv(os.path.join(v3_dir, 'test_data.csv'))
    v3_all = pd.concat([v3_train, v3_val, v3_test], ignore_index=True)
    print(f"  v3 total: {len(v3_all)} samples, "
          f"{v3_all['epitope'].nunique()} epitopes")

    # Add fold column
    v3_all['fold'] = v3_all['epitope'].map(ep_to_fold)
    assert v3_all['fold'].notna().all(), "Some epitopes not in fold assignments!"

    # Seen pool: folds 0,1,2
    seen_pool = v3_all[v3_all['fold'].isin([0, 1, 2])].copy()
    unseen_pool = v3_all[v3_all['fold'].isin([3, 4])].copy()
    print(f"  Seen pool (folds 0,1,2): {len(seen_pool)} samples, "
          f"{seen_pool['epitope'].nunique()} epitopes")
    print(f"  Unseen pool (folds 3,4): {len(unseen_pool)} samples, "
          f"{unseen_pool['epitope'].nunique()} epitopes")

    # Split seen pool 50/50 stratified by epitope.
    # Epitopes with only 1 sample go into training to ensure all seen_test
    # epitopes are represented in training.
    from sklearn.model_selection import train_test_split

    ep_counts = seen_pool['epitope'].value_counts()
    singleton_eps = set(ep_counts[ep_counts == 1].index)
    singleton_mask = seen_pool['epitope'].isin(singleton_eps)

    main_pool = seen_pool[~singleton_mask]
    singleton_pool = seen_pool[singleton_mask]

    train_main, test_main = train_test_split(
        main_pool, test_size=0.5, random_state=seed,
        stratify=main_pool['epitope'])

    # Singletons all go to training (ensures every epitope is "seen")
    if len(singleton_pool) > 0:
        train_df = pd.concat([train_main, singleton_pool], ignore_index=True)
        seen_test_df = test_main.reset_index(drop=True)
        print(f"  ({len(singleton_pool)} singleton-epitope samples → training)")
    else:
        train_df = train_main.reset_index(drop=True)
        seen_test_df = test_main.reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    seen_test_df = seen_test_df.reset_index(drop=True)

    print(f"  Train (50% seen pool): {len(train_df)} samples, "
          f"{train_df['epitope'].nunique()} epitopes")
    print(f"  A: seen_test (50% seen pool): {len(seen_test_df)} samples, "
          f"{seen_test_df['epitope'].nunique()} epitopes")

    # Test set B: unseen_fold34
    unseen_fold34_df = unseen_pool.reset_index(drop=True)
    print(f"  B: unseen_fold34: {len(unseen_fold34_df)} samples, "
          f"{unseen_fold34_df['epitope'].nunique()} epitopes")

    # Test set C: v3 test + val (deduplicated against training)
    v3_combined_raw = pd.concat([v3_test, v3_val], ignore_index=True)
    v3_combined_df = _dedup_against_train(v3_combined_raw, train_df,
                                          key_cols=['epitope', 'cdr3_a',
                                                    'cdr3_b', 'binding_label'])
    print(f"  C: v3_combined (test+val): {len(v3_combined_raw)} → "
          f"{len(v3_combined_df)} after dedup, "
          f"{v3_combined_df['epitope'].nunique()} epitopes")

    # Test set D: v4 test + val (deduplicated against training)
    v4_dir = os.path.join(data_dir, 'tcr_ml_v4')
    v4_val = pd.read_csv(os.path.join(v4_dir, 'validation_data.csv'))
    v4_test = pd.read_csv(os.path.join(v4_dir, 'test_data.csv'))
    v4_combined_raw = pd.concat([v4_test, v4_val], ignore_index=True)
    v4_combined_df = _dedup_against_train(v4_combined_raw, train_df,
                                          key_cols=['epitope', 'cdr3_a',
                                                    'cdr3_b', 'binding_label'])
    print(f"  D: v4_combined (test+val): {len(v4_combined_raw)} → "
          f"{len(v4_combined_df)} after dedup, "
          f"{v4_combined_df['epitope'].nunique()} epitopes")

    # Test set E: McPAS-TCR (external, independent database)
    mcpas_dir = os.path.join(data_dir, 'mcpas')
    mcpas_path = os.path.join(mcpas_dir, 'test_data.csv')
    if os.path.exists(mcpas_path):
        mcpas_raw = pd.read_csv(mcpas_path)
        mcpas_df = _dedup_against_train(mcpas_raw, train_df,
                                         key_cols=['epitope', 'cdr3_a',
                                                   'cdr3_b', 'binding_label'])
        print(f"  E: mcpas (external): {len(mcpas_raw)} → "
              f"{len(mcpas_df)} after dedup, "
              f"{mcpas_df['epitope'].nunique()} epitopes")
    else:
        mcpas_df = None
        print(f"  E: mcpas -- SKIPPED (file not found)")

    # Drop the fold column (not needed downstream)
    for df in [train_df, seen_test_df, unseen_fold34_df]:
        df.drop(columns=['fold'], inplace=True, errors='ignore')

    train_epitopes = set(train_df['epitope'].unique())

    test_sets = {
        'seen_test': seen_test_df,
        'unseen_fold34': unseen_fold34_df,
        'v3_combined': v3_combined_df,
        'v4_combined': v4_combined_df,
    }
    if mcpas_df is not None:
        test_sets['mcpas'] = mcpas_df

    return train_df, test_sets, train_epitopes


# ============================================================================
# Step 2: Prepare NetTCR-format files
# ============================================================================

def prepare_nettcr_files(train_df, test_sets, splits_dir):
    """Convert to NetTCR column format, add test_set_id, save CSVs.

    Returns:
        train_path, mega_test_path
    """
    print("\n[Step 2] Preparing NetTCR-format files...")

    col_map = {
        'epitope': 'peptide',
        'cdr3_a': 'CDR3a',
        'cdr3_b': 'CDR3b',
        'binding_label': 'binder',
    }
    nettcr_cols = ['peptide', 'CDR3a', 'CDR3b', 'binder']

    os.makedirs(splits_dir, exist_ok=True)

    # Train file
    train_nettcr = train_df.rename(columns=col_map)[nettcr_cols].copy()
    train_path = os.path.join(splits_dir, 'train.csv')
    train_nettcr.to_csv(train_path, index=False)
    print(f"  train.csv: {len(train_nettcr)} rows")

    # Per-test-set files + mega_test
    mega_parts = []
    for name, df in test_sets.items():
        test_nettcr = df.rename(columns=col_map)[nettcr_cols].copy()
        test_nettcr['test_set_id'] = name

        # Save individual test set
        test_path = os.path.join(splits_dir, f'{name}.csv')
        test_nettcr.to_csv(test_path, index=False)
        print(f"  {name}.csv: {len(test_nettcr)} rows")

        mega_parts.append(test_nettcr)

    # Concatenate into mega_test
    mega_test = pd.concat(mega_parts, ignore_index=True)
    mega_path = os.path.join(splits_dir, 'mega_test.csv')
    mega_test.to_csv(mega_path, index=False)
    print(f"  mega_test.csv: {len(mega_test)} rows "
          f"({mega_test['test_set_id'].value_counts().to_dict()})")

    return train_path, mega_path


# ============================================================================
# Step 3: Train NetTCR ONCE + predict on mega_test
# ============================================================================

def train_and_predict(train_path, mega_test_path, predictions_dir,
                      train_epitopes, epochs, chain):
    """Train NetTCR once on train.csv, predict on mega_test.csv.

    Returns:
        dict of test_set_name -> DataFrame with predictions
    """
    print("\n[Step 3] Training NetTCR (single run, all test sets)...")

    nettcr_script = os.path.join(PROJECT_ROOT, 'Model', 'NetTCR', 'nettcr.py')
    nettcr_cwd = os.path.join(PROJECT_ROOT, 'Model', 'NetTCR')
    os.makedirs(predictions_dir, exist_ok=True)

    mega_pred_path = os.path.join(predictions_dir,
                                  'mega_test_predictions.csv')

    cmd = [
        sys.executable, nettcr_script,
        '--trainfile', train_path,
        '--testfile', mega_test_path,
        '--chain', chain,
        '--epochs', str(epochs),
        '--outfile', mega_pred_path,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {nettcr_cwd}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=nettcr_cwd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR training NetTCR (returncode={result.returncode}):")
        stderr = result.stderr
        print(stderr[-3000:] if len(stderr) > 3000 else stderr)
        sys.exit(1)

    print(f"  Training + prediction done in {elapsed:.1f}s")

    # Read mega predictions
    mega_pred = pd.read_csv(mega_pred_path)
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
        # Add epitope_seen column
        group['epitope_seen'] = group['peptide'].apply(
            lambda x: 1 if x in train_epitopes else 0)
        # Save
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
# Step 4: Compute sigma_H chain weights
# ============================================================================

def compute_sigma_h_weights(train_path, chain_cols, k, b, K,
                            subsample=500, seed=42):
    """Compute sigma_H chain weights (w_i = sigma_i * H_i, normalized).

    sigma_i = std of LogDist distances for chain i (train vs train subsample)
    H_i = Shannon entropy of chain i sequences in training data

    Returns:
        weights: np.array of shape (n_chains,), normalized to sum to 1
    """
    from collections import Counter

    print("\n[Step 4] Computing sigma_H chain weights...")

    train = pd.read_csv(train_path)
    n_chains = len(chain_cols)
    n_train = len(train)
    n_sub = min(subsample, n_train)
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(n_train, size=n_sub, replace=False)

    chain_stds = np.zeros(n_chains)
    chain_H = np.zeros(n_chains)

    for ch_idx, col in enumerate(chain_cols):
        ref_seqs = train[col].tolist()
        sub_seqs = [ref_seqs[i] for i in sub_idx]

        # Compute per-chain LogDist std (sigma)
        _, _, train_rmap = compute_pairwise_ratios(sub_seqs, ref_seqs)
        train_dists = np.array([
            logdist_from_ratios(train_rmap[s], k, b, K) for s in sub_seqs])
        chain_stds[ch_idx] = float(np.std(train_dists))

        # Compute Shannon entropy H
        freq = Counter(ref_seqs)
        counts = np.array(list(freq.values()), dtype=float)
        p = counts / counts.sum()
        chain_H[ch_idx] = float(-np.sum(p * np.log(p)))

        print(f"  {col}: std={chain_stds[ch_idx]:.6f}, H={chain_H[ch_idx]:.6f}")

    raw = chain_stds * chain_H
    weights = raw / raw.sum() if raw.sum() > 0 else np.ones(n_chains) / n_chains

    w_str = ', '.join(f'{c}={weights[i]:.4f}'
                      for i, c in enumerate(chain_cols))
    print(f"  Chain weights (sigma_H): {w_str}")
    return weights


# ============================================================================
# Step 5: Compute 3-chain LogDist per test set
# ============================================================================

def compute_logdist_per_testset(pred_sets, train_path, chain_cols,
                                weights, k, b, K):
    """Compute per-sample 3-chain LogDist for each test set.

    Returns:
        dict of test_set_name -> DataFrame with 'distance' column added
    """
    print("\n[Step 5] Computing 3-chain LogDist per test set...")

    train = pd.read_csv(train_path)

    # Compute pairwise ratios per chain (shared training reference)
    ratio_maps = {}
    for col in chain_cols:
        ref_seqs = train[col].tolist()
        # Collect all unique query sequences across all test sets
        all_qry_seqs = []
        for name, df in pred_sets.items():
            all_qry_seqs.extend(df[col].tolist())
        all_qry_seqs = list(dict.fromkeys(all_qry_seqs))  # unique, preserving order

        print(f"  Pairwise ratios for {col}: "
              f"{len(all_qry_seqs)} unique queries × "
              f"{len(set(ref_seqs))} unique refs")
        _, _, rmap = compute_pairwise_ratios(all_qry_seqs, ref_seqs)
        ratio_maps[col] = rmap

    # Compute distances per test set
    result_sets = {}
    for name, df in pred_sets.items():
        chain_seqs = [df[col].tolist() for col in chain_cols]
        rmaps = [ratio_maps[col] for col in chain_cols]
        distances = compute_multichain_distances(
            chain_seqs, rmaps, k, b, K, weights=weights)

        df = df.copy()
        df['distance'] = distances
        result_sets[name] = df
        print(f"  {name}: distance range [{distances.min():.4f}, "
              f"{distances.max():.4f}], mean={distances.mean():.4f}")

    return result_sets


# ============================================================================
# Step 6: Binned correlations + plots
# ============================================================================

def evaluate_and_plot(result_sets, train_epitopes, chain_cols, weights,
                      K, k, b, bin_num, eval_metrics, output_dir):
    """Compute binned correlations and generate comparison plots."""
    print("\n[Step 6] Binned correlations and plots...")

    os.makedirs(output_dir, exist_ok=True)

    test_names = list(result_sets.keys())
    all_corr_rows = []

    # Per-test-set evaluation
    test_results = {}
    for name in test_names:
        df = result_sets[name]
        base_df = pd.DataFrame({
            'peptide': df['peptide'],
            'label': df['binder'].astype(float),
            'pred': df['prediction'].astype(float),
            'epitope_seen': df['epitope_seen'],
            'distance': df['distance'],
        })

        seen_df = base_df[base_df['epitope_seen'] == 1]
        unseen_df = base_df[base_df['epitope_seen'] == 0]

        # Binned correlations for each scope
        scopes = {}
        for scope_name, scope_df in [('seen', seen_df),
                                      ('unseen', unseen_df),
                                      ('combined', base_df)]:
            dist_unique = scope_df['distance'].nunique()
            if len(scope_df) >= bin_num * 2 and dist_unique >= bin_num:
                scopes[scope_name] = binned_correlations(
                    scope_df, 'distance', eval_metrics, bin_num)
            else:
                scopes[scope_name] = {
                    m: {'pearson_r': np.nan, 'pearson_p': np.nan,
                        'spearman_r': np.nan, 'spearman_p': np.nan,
                        'bin_dists': [], 'bin_perfs': []}
                    for m in eval_metrics}

        # Per-epitope metrics
        ep_rows = []
        for ep in base_df['peptide'].unique():
            ep_data = base_df[base_df['peptide'] == ep]
            row = {
                'epitope': ep,
                'test_set': name,
                'seen_status': 'seen' if ep_data['epitope_seen'].iloc[0] == 1 else 'unseen',
                'n_samples': len(ep_data),
                'distance': ep_data['distance'].mean(),
            }
            for m in eval_metrics:
                row[m] = safe_metric(
                    m, ep_data['label'].values, ep_data['pred'].values)
            ep_rows.append(row)
        ep_df = pd.DataFrame(ep_rows)

        test_results[name] = {
            'base_df': base_df,
            'seen_binned': scopes['seen'],
            'unseen_binned': scopes['unseen'],
            'combined_binned': scopes['combined'],
            'ep_df': ep_df,
            'n_total': len(base_df),
            'n_seen': len(seen_df),
            'n_unseen': len(unseen_df),
        }

        # Build correlation summary rows
        for m in eval_metrics:
            row = {'test_set': name, 'metric': m}
            for scope_name in ['seen', 'unseen', 'combined']:
                bc = scopes[scope_name][m]
                row[f'{scope_name}_binned_r'] = bc['pearson_r']
                row[f'{scope_name}_binned_p'] = bc['pearson_p']
                row[f'{scope_name}_spearman_r'] = bc['spearman_r']
                row[f'{scope_name}_spearman_p'] = bc['spearman_p']
            all_corr_rows.append(row)

        # Print results
        print(f"\n  {name}: {len(base_df)} samples "
              f"({len(seen_df)} seen + {len(unseen_df)} unseen)")
        print(f"    {'Metric':>8}  {'S bin|r|':>8} {'p':>10}  "
              f"{'U bin|r|':>8} {'p':>10}  {'C bin|r|':>8} {'p':>10}")
        print(f"    {'-'*78}")
        p_fmt = lambda p: (f"{p:.4f}{'*' if p < 0.05 else ' '}"
                           if not np.isnan(p) else "   nan ")
        for m in eval_metrics:
            sb = scopes['seen'][m]
            ub = scopes['unseen'][m]
            cb = scopes['combined'][m]
            sb_r = abs(sb['pearson_r']) if not np.isnan(sb['pearson_r']) else np.nan
            ub_r = abs(ub['pearson_r']) if not np.isnan(ub['pearson_r']) else np.nan
            cb_r = abs(cb['pearson_r']) if not np.isnan(cb['pearson_r']) else np.nan
            print(f"    {m:>8}  {sb_r:>8.4f} {p_fmt(sb['pearson_p']):>10}  "
                  f"{ub_r:>8.4f} {p_fmt(ub['pearson_p']):>10}  "
                  f"{cb_r:>8.4f} {p_fmt(cb['pearson_p']):>10}")

    # Save summary CSV
    corr_df = pd.DataFrame(all_corr_rows)
    corr_df.to_csv(os.path.join(output_dir, 'cross_test_summary.csv'),
                   index=False)

    # Save per-test-set epitope metrics
    all_ep = pd.concat([res['ep_df'] for res in test_results.values()],
                       ignore_index=True)
    all_ep.to_csv(os.path.join(output_dir, 'per_epitope_metrics.csv'),
                  index=False)

    # ---- Summary table ----
    _print_summary(test_results, test_names, eval_metrics, K, k, b)

    # ---- Save split statistics ----
    _save_split_statistics(test_results, test_names, train_epitopes,
                           weights, chain_cols, K, k, b, bin_num, output_dir)

    # ---- Plots ----
    _plot_aucroc_overlay(test_results, test_names, K, k, b, output_dir)
    _plot_bar_comparison(test_results, test_names, eval_metrics, output_dir)
    _plot_all_metrics_grid(test_results, test_names, eval_metrics,
                           K, k, b, output_dir)

    return test_results, corr_df


def _print_summary(test_results, test_names, eval_metrics, K, k, b):
    """Print cross-test-set comparison summary."""
    print(f"\n\n{'='*120}")
    print(f"CROSS-TEST-SET COMPARISON — sigma_H + topK LogDist "
          f"(K={K}, k={k}, b={b})")
    print(f"{'='*120}")

    # AUCROC focus
    print(f"\n  AUCROC Binned |Pearson r|:")
    print(f"  {'Test Set':>20}  {'Seen':>10}  {'Unseen':>10}  "
          f"{'Combined':>10}  {'N total':>8}  {'N seen':>8}  {'N unseen':>8}")
    print(f"  {'-'*90}")

    combined_rs = []
    for name in test_names:
        res = test_results[name]
        s_r = abs(res['seen_binned']['aucroc']['pearson_r']) \
            if not np.isnan(res['seen_binned']['aucroc']['pearson_r']) else np.nan
        u_r = abs(res['unseen_binned']['aucroc']['pearson_r']) \
            if not np.isnan(res['unseen_binned']['aucroc']['pearson_r']) else np.nan
        c_r = abs(res['combined_binned']['aucroc']['pearson_r']) \
            if not np.isnan(res['combined_binned']['aucroc']['pearson_r']) else np.nan

        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "
        print(f"  {name:>20}  {fmt(s_r):>10}  {fmt(u_r):>10}  "
              f"{fmt(c_r):>10}  {res['n_total']:>8}  "
              f"{res['n_seen']:>8}  {res['n_unseen']:>8}")
        if not np.isnan(c_r):
            combined_rs.append(c_r)

    if combined_rs:
        mean_r = np.mean(combined_rs)
        std_r = np.std(combined_rs)
        cv = std_r / mean_r if mean_r > 0 else np.nan
        print(f"\n  Combined AUCROC binned |r| across test sets:")
        print(f"    Mean:  {mean_r:.4f}")
        print(f"    Std:   {std_r:.4f}")
        print(f"    CV:    {cv:.4f}")
        print(f"    Range: {min(combined_rs):.4f} – {max(combined_rs):.4f}")
        if cv < 0.15:
            print(f"    → ROBUST (CV < 0.15)")
        elif cv < 0.30:
            print(f"    → MODERATE robustness (CV < 0.30)")
        else:
            print(f"    → LOW robustness (CV >= 0.30)")


def _save_split_statistics(test_results, test_names, train_epitopes,
                           weights, chain_cols, K, k, b, bin_num, output_dir):
    """Save split statistics to text file."""
    path = os.path.join(output_dir, 'split_statistics.txt')
    with open(path, 'w') as f:
        f.write("Cross-Test LogDist Consistency Experiment\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Parameters: K={K}, k={k}, b={b}, bin_num={bin_num}\n")
        f.write(f"Chain cols: {chain_cols}\n")
        w_str = ', '.join(f'{c}={weights[i]:.4f}'
                          for i, c in enumerate(chain_cols))
        f.write(f"Chain weights (sigma_H): {w_str}\n")
        f.write(f"Training epitopes: {len(train_epitopes)}\n\n")

        f.write(f"{'Test Set':>20}  {'N total':>8}  {'N seen':>8}  "
                f"{'N unseen':>8}  {'% seen':>8}\n")
        f.write("-" * 60 + "\n")
        for name in test_names:
            res = test_results[name]
            pct = 100 * res['n_seen'] / res['n_total'] if res['n_total'] > 0 else 0
            f.write(f"{name:>20}  {res['n_total']:>8}  {res['n_seen']:>8}  "
                    f"{res['n_unseen']:>8}  {pct:>7.1f}%\n")

    print(f"\n  Saved: {path}")


def _plot_aucroc_overlay(test_results, test_names, K, k, b, output_dir):
    """Overlay AUCROC binned curves from all test sets (3 panels)."""
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(test_names)))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for col_idx, (scope_key, scope_label) in enumerate([
        ('seen_binned', 'Seen Epitopes'),
        ('unseen_binned', 'Unseen Epitopes'),
        ('combined_binned', 'Combined'),
    ]):
        ax = axes[col_idx]
        ax.set_title(scope_label, fontsize=13, fontweight='bold')
        for i, name in enumerate(test_names):
            binned = test_results[name][scope_key]
            bd = binned['aucroc'].get('bin_dists', [])
            bp = binned['aucroc'].get('bin_perfs', [])
            if bd and bp:
                r_val = binned['aucroc']['pearson_r']
                p_val = binned['aucroc']['pearson_p']
                sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
                ax.plot(bd, bp, 'o-', color=colors[i],
                        label=f'{name}  r={r_val:+.3f}{sig}',
                        markersize=5, linewidth=1.5)

        ax.set_xlabel('3-Chain LogDist (sigma_H)', fontsize=11)
        ax.set_ylabel('AUCROC', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle(f'Cross-Test AUCROC vs LogDist — Single Model '
                 f'(K={K}, k={k}, b={b})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_test_aucroc_overlay.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_bar_comparison(test_results, test_names, eval_metrics, output_dir):
    """Bar chart: test sets side by side for each metric's combined |r|."""
    n_metrics = len(eval_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / len(test_names)
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(test_names)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(test_names):
        rs = []
        for m in eval_metrics:
            binned = test_results[name]['combined_binned']
            r_val = binned[m]['pearson_r']
            rs.append(abs(r_val) if not np.isnan(r_val) else 0)
        offset = (i - len(test_names) / 2 + 0.5) * width
        ax.bar(x + offset, rs, width, label=name, color=colors[i])

    ax.set_xlabel('Performance Metric', fontsize=11)
    ax.set_ylabel('|Pearson r| (Combined)', fontsize=11)
    ax.set_title('Cross-Test Binned |Pearson r| — Combined (Single Model)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(eval_metrics)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_test_bars.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_all_metrics_grid(test_results, test_names, eval_metrics,
                           K, k, b, output_dir):
    """Grid plot: rows=test sets, cols=scopes, all metrics overlaid."""
    n_tests = len(test_names)
    fig, axes = plt.subplots(n_tests, 3, figsize=(24, 5 * n_tests))
    if n_tests == 1:
        axes = axes.reshape(1, -1)

    for row_idx, name in enumerate(test_names):
        for col_idx, (scope_key, scope_label) in enumerate([
            ('seen_binned', 'Seen'),
            ('unseen_binned', 'Unseen'),
            ('combined_binned', 'Combined'),
        ]):
            ax = axes[row_idx, col_idx]
            binned = test_results[name][scope_key]
            ax.set_title(f'{name} — {scope_label}',
                         fontsize=10, fontweight='bold')
            for m in eval_metrics:
                bd = binned[m].get('bin_dists', [])
                bp = binned[m].get('bin_perfs', [])
                if bd and bp:
                    r_val = abs(binned[m]['pearson_r']) \
                        if not np.isnan(binned[m]['pearson_r']) else 0
                    p_val = binned[m]['pearson_p']
                    sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
                    ax.plot(bd, bp, 'o-',
                            label=f'{m} |r|={r_val:.2f}{sig}',
                            markersize=3)
            ax.set_xlabel('LogDist', fontsize=9)
            ax.set_ylabel('Performance', fontsize=9)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Cross-Test: All Metrics (K={K}, k={k}, b={b})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_test_all_metrics_grid.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Verification
# ============================================================================

def verify_splits(test_results, train_epitopes):
    """Verify data integrity of the experiment."""
    print("\n[Verification]")

    # Check A: seen_test has 0% unseen epitopes
    seen_test = test_results.get('seen_test')
    if seen_test is not None:
        n_unseen = seen_test['n_unseen']
        pct_unseen = 100 * n_unseen / seen_test['n_total']
        status = "PASS" if n_unseen == 0 else "FAIL"
        print(f"  {status}: seen_test has {pct_unseen:.1f}% unseen "
              f"({n_unseen}/{seen_test['n_total']})")

    # Check B: unseen_fold34 has 0% seen epitopes
    unseen_fold34 = test_results.get('unseen_fold34')
    if unseen_fold34 is not None:
        n_seen = unseen_fold34['n_seen']
        pct_seen = 100 * n_seen / unseen_fold34['n_total']
        status = "PASS" if n_seen == 0 else "FAIL"
        print(f"  {status}: unseen_fold34 has {pct_seen:.1f}% seen "
              f"({n_seen}/{unseen_fold34['n_total']})")

    # Check C & D: mixed
    for name in ['v3_combined', 'v4_combined']:
        res = test_results.get(name)
        if res is not None:
            pct_seen = 100 * res['n_seen'] / res['n_total']
            print(f"  INFO: {name} has {pct_seen:.1f}% seen epitopes "
                  f"({res['n_seen']}/{res['n_total']})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Test-Set LogDist Consistency Experiment")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing data files')
    parser.add_argument('--output-dir', type=str,
                        default='results/nettcr/cross_test_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50,
                        help='Top-K for LogDist reduction')
    parser.add_argument('--k', type=float, default=0.1,
                        help='LogDist scaling parameter')
    parser.add_argument('--b', type=float, default=0.1,
                        help='LogDist clipping parameter')
    parser.add_argument('--bin-num', type=int, default=8,
                        help='Number of equal-sized bins')
    parser.add_argument('--epochs', type=int, default=100,
                        help='NetTCR training epochs')
    parser.add_argument('--chain', type=str, default='ab',
                        help='NetTCR chain type (a, b, ab)')
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'],
                        help='Performance metrics to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip NetTCR training, use existing predictions')
    args = parser.parse_args()

    t_start = time.time()
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    splits_dir = os.path.join(output_dir, 'splits')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    chain_cols = ['peptide', 'CDR3a', 'CDR3b']

    print(f"{'='*80}")
    print(f"Cross-Test-Set LogDist Consistency Experiment")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")

    # Step 1: Build pools
    train_df, test_sets, train_epitopes = build_pools(data_dir, args.seed)

    # Step 2: Prepare NetTCR files
    train_path, mega_test_path = prepare_nettcr_files(
        train_df, test_sets, splits_dir)

    # Step 3: Train NetTCR + predict
    test_set_names = list(test_sets.keys())
    if not args.skip_training:
        pred_sets = train_and_predict(
            train_path, mega_test_path, predictions_dir,
            train_epitopes, args.epochs, args.chain)
    else:
        pred_sets = load_existing_predictions(
            predictions_dir, test_set_names, train_epitopes)

    # Step 4: Compute sigma_H weights
    weights = compute_sigma_h_weights(
        train_path, chain_cols, args.k, args.b, args.K,
        subsample=500, seed=args.seed)

    # Step 5: Compute LogDist per test set
    result_sets = compute_logdist_per_testset(
        pred_sets, train_path, chain_cols, weights,
        args.k, args.b, args.K)

    # Step 6: Evaluate and plot
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
    print(f"    splits/                        — train.csv, mega_test.csv, "
          f"per-test-set CSVs")
    print(f"    predictions/                   — mega_test_predictions.csv, "
          f"per-test-set prediction CSVs")
    print(f"    cross_test_summary.csv         — all test set × metric "
          f"correlations")
    print(f"    per_epitope_metrics.csv        — per-epitope performance "
          f"across test sets")
    print(f"    cross_test_aucroc_overlay.png  — AUCROC curves overlay "
          f"(3 panels)")
    print(f"    cross_test_bars.png            — bar chart comparison")
    print(f"    cross_test_all_metrics_grid.png — full metrics grid")
    print(f"    split_statistics.txt           — experiment statistics")


if __name__ == '__main__':
    main()
