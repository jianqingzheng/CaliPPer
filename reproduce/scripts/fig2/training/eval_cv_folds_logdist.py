#!/usr/bin/env python3
"""
5-Fold Cross-Validation LogDist Evaluation Pipeline.

For each fold:
  1. Convert fold data to NetTCR format (peptide, CDR3a, CDR3b, binder)
  2. Train NetTCR on fold's train set → predict on val and test
  3. Add epitope_seen labels (test = unseen, val = seen)
  4. Compute 3-chain LogDist at fixed (K, k, b) = (50, 0.1, 0.1)
  5. Compute binned and per-epitope correlations

Then: compare curves and correlations across all 5 folds.
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
    compute_multichain_distances,
    safe_metric,
    binned_correlations,
    per_epitope_correlations,
)


# ============================================================================
# Step 1: Prepare NetTCR-format files for a fold
# ============================================================================

def prepare_nettcr_format(fold_dir, output_dir):
    """Convert fold CSVs (epitope/cdr3_a/cdr3_b/binding_label) to NetTCR format."""
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
        nettcr_df = df.rename(columns=col_map)
        out_path = os.path.join(output_dir, dst_name)
        nettcr_df.to_csv(out_path, index=False)
        paths[dst_name] = out_path
        print(f"    {src_name} → {dst_name} ({len(nettcr_df)} rows)")
    return paths


# ============================================================================
# Step 2: Train NetTCR and get predictions
# ============================================================================

def train_nettcr_fold(nettcr_files, output_dir, epochs, chain):
    """Train NetTCR on fold's train, predict on val and test."""
    nettcr_script = os.path.join(PROJECT_ROOT, 'Model', 'NetTCR', 'nettcr.py')
    nettcr_cwd = os.path.join(PROJECT_ROOT, 'Model', 'NetTCR')

    train_path = nettcr_files['train.csv']

    runs = [
        ('val', nettcr_files['val.csv'], os.path.join(output_dir, 'val_predictions.csv')),
        ('test', nettcr_files['test.csv'], os.path.join(output_dir, 'test_predictions.csv')),
    ]

    for run_name, test_file, out_file in runs:
        print(f"    Training NetTCR: train → {run_name} predictions")
        cmd = [
            sys.executable, nettcr_script,
            '--trainfile', train_path,
            '--testfile', test_file,
            '--chain', chain,
            '--epochs', str(epochs),
            '--outfile', out_file,
        ]
        t0 = time.time()
        result = subprocess.run(cmd, cwd=nettcr_cwd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"    ERROR training NetTCR (returncode={result.returncode}):")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            sys.exit(1)

        pred_df = pd.read_csv(out_file)
        print(f"      Done in {elapsed:.1f}s — {len(pred_df)} predictions")


def add_epitope_seen_labels(output_dir, train_path):
    """Add epitope_seen column based on training epitopes."""
    train = pd.read_csv(train_path)
    train_epitopes = set(train['peptide'].unique())

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
# Step 3: Evaluate LogDist at fixed (K, k, b)
# ============================================================================

def evaluate_fold_logdist(fold_id, output_dir, chain_cols, K, k, b,
                          bin_num, eval_metrics):
    """Compute 3-chain LogDist and correlations for one fold."""
    print(f"\n  [Fold {fold_id}] Computing 3-chain LogDist (K={K}, k={k}, b={b})...")

    train = pd.read_csv(os.path.join(output_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(output_dir, 'val_predictions_with_label.csv'))
    test = pd.read_csv(os.path.join(output_dir, 'test_predictions_with_label.csv'))
    all_qry = pd.concat([val, test], ignore_index=True)

    print(f"    Train: {len(train)} samples, {train['peptide'].nunique()} epitopes")
    print(f"    Query: {len(all_qry)} samples "
          f"({(all_qry['epitope_seen']==1).sum()} seen + "
          f"{(all_qry['epitope_seen']==0).sum()} unseen)")

    # Compute pairwise ratios per chain
    ratio_maps = {}
    for col in chain_cols:
        print(f"    Pairwise ratios: {col}")
        _, _, rmap = compute_pairwise_ratios(
            all_qry[col].tolist(), train[col].tolist())
        ratio_maps[col] = rmap

    # Compute multi-chain distances
    chain_seqs = [all_qry[col].tolist() for col in chain_cols]
    rmaps = [ratio_maps[col] for col in chain_cols]
    distances = compute_multichain_distances(chain_seqs, rmaps, k, b, K)
    print(f"    Distance range: [{distances.min():.4f}, {distances.max():.4f}], "
          f"mean={distances.mean():.4f}")

    # Build evaluation DataFrame
    base_df = pd.DataFrame({
        'peptide': all_qry['peptide'],
        'label': all_qry['binder'],
        'pred': all_qry['prediction'],
        'epitope_seen': all_qry['epitope_seen'],
        'distance': distances,
    })

    seen_df = base_df[base_df['epitope_seen'] == 1]
    unseen_df = base_df[base_df['epitope_seen'] == 0]

    # Binned correlations
    seen_dist_uniq = seen_df['distance'].nunique()
    if len(seen_df) >= bin_num * 2 and seen_dist_uniq >= bin_num:
        seen_binned = binned_correlations(seen_df, 'distance', eval_metrics, bin_num)
    else:
        seen_binned = {m: {'pearson_r': np.nan, 'pearson_p': np.nan,
                           'spearman_r': np.nan, 'spearman_p': np.nan,
                           'bin_dists': [], 'bin_perfs': []}
                       for m in eval_metrics}

    unseen_binned = binned_correlations(unseen_df, 'distance', eval_metrics, bin_num)
    combined_binned = binned_correlations(base_df, 'distance', eval_metrics, bin_num)

    # Per-epitope correlations
    ep_metrics = {}
    for ep in base_df['peptide'].unique():
        ep_data = base_df[base_df['peptide'] == ep]
        row = {
            'epitope': ep,
            'seen_status': 'seen' if ep_data['epitope_seen'].iloc[0] == 1 else 'unseen',
            'n_samples': len(ep_data),
            'distance': ep_data['distance'].mean(),
        }
        for m in eval_metrics:
            row[m] = safe_metric(m, ep_data['label'].values, ep_data['pred'].values)
        ep_metrics[ep] = row

    ep_df = pd.DataFrame(list(ep_metrics.values()))
    seen_ep = ep_df[(ep_df['seen_status'] == 'seen') & ep_df['aucroc'].notna()]
    unseen_ep = ep_df[(ep_df['seen_status'] == 'unseen') & ep_df['aucroc'].notna()]
    all_ep = ep_df[ep_df['aucroc'].notna()]

    seen_ep_corr = per_epitope_correlations(seen_ep, 'distance', eval_metrics)
    unseen_ep_corr = per_epitope_correlations(unseen_ep, 'distance', eval_metrics)
    combined_ep_corr = per_epitope_correlations(all_ep, 'distance', eval_metrics)

    # Build results rows
    result_rows = []
    for m in eval_metrics:
        result_rows.append({
            'fold': fold_id,
            'metric': m,
            'seen_binned_r': seen_binned[m]['pearson_r'],
            'seen_binned_p': seen_binned[m]['pearson_p'],
            'unseen_binned_r': unseen_binned[m]['pearson_r'],
            'unseen_binned_p': unseen_binned[m]['pearson_p'],
            'combined_binned_r': combined_binned[m]['pearson_r'],
            'combined_binned_p': combined_binned[m]['pearson_p'],
            'seen_ep_r': seen_ep_corr[m]['pearson_r'],
            'seen_ep_p': seen_ep_corr[m]['pearson_p'],
            'unseen_ep_r': unseen_ep_corr[m]['pearson_r'],
            'unseen_ep_p': unseen_ep_corr[m]['pearson_p'],
            'combined_ep_r': combined_ep_corr[m]['pearson_r'],
            'combined_ep_p': combined_ep_corr[m]['pearson_p'],
        })

    # Print fold results
    print(f"\n    {'Metric':>8}  {'S bin|r|':>8} {'p':>10}  "
          f"{'U bin|r|':>8} {'p':>10}  {'C bin|r|':>8} {'p':>10}")
    print(f"    {'-'*78}")
    p_fmt = lambda p: f"{p:.4f}{'*' if p < 0.05 else ' '}" if not np.isnan(p) else "   nan "
    for m in eval_metrics:
        sb = seen_binned[m]
        ub = unseen_binned[m]
        cb = combined_binned[m]
        sb_r = abs(sb['pearson_r']) if not np.isnan(sb['pearson_r']) else np.nan
        ub_r = abs(ub['pearson_r']) if not np.isnan(ub['pearson_r']) else np.nan
        cb_r = abs(cb['pearson_r']) if not np.isnan(cb['pearson_r']) else np.nan
        print(f"    {m:>8}  {sb_r:>8.4f} {p_fmt(sb['pearson_p']):>10}  "
              f"{ub_r:>8.4f} {p_fmt(ub['pearson_p']):>10}  "
              f"{cb_r:>8.4f} {p_fmt(cb['pearson_p']):>10}")

    return {
        'results_df': pd.DataFrame(result_rows),
        'seen_binned': seen_binned,
        'unseen_binned': unseen_binned,
        'combined_binned': combined_binned,
        'base_df': base_df,
        'ep_df': ep_df,
        'split_stats': {
            'n_train': len(train),
            'n_query': len(all_qry),
            'n_seen': int((all_qry['epitope_seen'] == 1).sum()),
            'n_unseen': int((all_qry['epitope_seen'] == 0).sum()),
            'n_epitopes_train': train['peptide'].nunique(),
            'n_epitopes_query': all_qry['peptide'].nunique(),
        },
    }


# ============================================================================
# Step 4: Cross-fold comparison
# ============================================================================

def compare_folds(all_fold_results, eval_metrics, K, k, b, output_dir):
    """Compare LogDist correlations across all folds with tables and plots."""
    n_folds = len(all_fold_results)

    print(f"\n\n{'='*120}")
    print(f"CROSS-FOLD COMPARISON — 3-chain LogDist at K={K}, k={k}, b={b}")
    print(f"{'='*120}")

    # Aggregate all results
    all_results = pd.concat(
        [res['results_df'] for res in all_fold_results], ignore_index=True)
    all_results.to_csv(os.path.join(output_dir, 'all_folds_correlations.csv'), index=False)

    # Print split statistics per fold
    print(f"\n  Split statistics:")
    print(f"  {'Fold':>6}  {'Train':>8}  {'Query':>8}  {'Seen':>8}  {'Unseen':>8}  "
          f"{'Train Ep':>9}  {'Query Ep':>9}")
    print(f"  {'-'*70}")
    for fold_id, res in enumerate(all_fold_results):
        s = res['split_stats']
        print(f"  {fold_id:>6}  {s['n_train']:>8,}  {s['n_query']:>8,}  "
              f"{s['n_seen']:>8,}  {s['n_unseen']:>8,}  "
              f"{s['n_epitopes_train']:>9}  {s['n_epitopes_query']:>9}")

    # Print per-metric comparison tables
    scopes = [
        ('seen_binned', 'Seen (binned)'),
        ('unseen_binned', 'Unseen (binned)'),
        ('combined_binned', 'Combined (binned)'),
        ('seen_ep', 'Seen (per-ep)'),
        ('unseen_ep', 'Unseen (per-ep)'),
        ('combined_ep', 'Combined (per-ep)'),
    ]

    for m in eval_metrics:
        print(f"\n  Metric: {m}")
        print(f"  {'Scope':>20}", end='')
        for f in range(n_folds):
            print(f"  {'Fold'+str(f):>10}", end='')
        print(f"  {'Mean|r|':>10}  {'Std':>8}  {'Range':>12}")
        print(f"  {'-'*110}")

        for scope_key, scope_label in scopes:
            r_col = f'{scope_key}_r'
            abs_rs = []
            print(f"  {scope_label:>20}", end='')
            for f in range(n_folds):
                fold_row = all_results[(all_results['fold'] == f) &
                                       (all_results['metric'] == m)]
                if fold_row.empty:
                    print(f"  {'nan':>10}", end='')
                    continue
                r_val = fold_row.iloc[0][r_col]
                p_col = f'{scope_key}_p'
                p_val = fold_row.iloc[0][p_col]
                abs_r = abs(r_val) if not np.isnan(r_val) else np.nan
                sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ' '
                abs_rs.append(abs_r)
                if np.isnan(abs_r):
                    print(f"  {'nan':>10}", end='')
                else:
                    print(f"  {abs_r:>8.4f}{sig}", end='')

            # Summary statistics
            valid_rs = [r for r in abs_rs if not np.isnan(r)]
            if valid_rs:
                mean_r = np.mean(valid_rs)
                std_r = np.std(valid_rs)
                range_r = f"[{min(valid_rs):.3f},{max(valid_rs):.3f}]"
                print(f"  {mean_r:>10.4f}  {std_r:>8.4f}  {range_r:>12}")
            else:
                print(f"  {'nan':>10}  {'nan':>8}  {'nan':>12}")

    # ---- Focus summary: AUCROC binned ----
    print(f"\n{'='*120}")
    print(f"SUMMARY — AUCROC Binned |Pearson r| across folds (K={K}, k={k}, b={b})")
    print(f"{'='*120}")

    summary_rows = []
    for scope_key, scope_label in [('seen_binned', 'Seen'),
                                     ('unseen_binned', 'Unseen'),
                                     ('combined_binned', 'Combined')]:
        r_col = f'{scope_key}_r'
        p_col = f'{scope_key}_p'
        fold_rs = []
        fold_ps = []
        for f in range(n_folds):
            row = all_results[(all_results['fold'] == f) &
                               (all_results['metric'] == 'aucroc')]
            if not row.empty:
                fold_rs.append(abs(row.iloc[0][r_col]) if not np.isnan(row.iloc[0][r_col]) else np.nan)
                fold_ps.append(row.iloc[0][p_col])
            else:
                fold_rs.append(np.nan)
                fold_ps.append(np.nan)

        valid_rs = [r for r in fold_rs if not np.isnan(r)]
        mean_r = np.mean(valid_rs) if valid_rs else np.nan
        std_r = np.std(valid_rs) if valid_rs else np.nan
        n_sig = sum(1 for p in fold_ps if not np.isnan(p) and p < 0.05)

        fold_str = '  '.join(
            f"{r:.4f}{'*' if (not np.isnan(p) and p < 0.05) else ' '}"
            for r, p in zip(fold_rs, fold_ps)
        )
        print(f"  {scope_label:>10}:  {fold_str}  "
              f"  mean={mean_r:.4f} ±{std_r:.4f}  sig={n_sig}/{n_folds}")

        summary_rows.append({
            'scope': scope_label,
            'mean_abs_r': mean_r,
            'std_abs_r': std_r,
            'n_significant': n_sig,
            **{f'fold{f}_abs_r': fold_rs[f] for f in range(n_folds)},
            **{f'fold{f}_p': fold_ps[f] for f in range(n_folds)},
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'aucroc_binned_summary.csv'), index=False)

    # ---- Robustness assessment ----
    print(f"\n{'='*120}")
    print("ROBUSTNESS ASSESSMENT")
    print(f"{'='*120}")

    for scope_key, scope_label in [('unseen_binned', 'Unseen'),
                                     ('combined_binned', 'Combined')]:
        r_col = f'{scope_key}_r'
        auc_rs = []
        for f in range(n_folds):
            row = all_results[(all_results['fold'] == f) &
                               (all_results['metric'] == 'aucroc')]
            if not row.empty:
                r = row.iloc[0][r_col]
                if not np.isnan(r):
                    auc_rs.append(abs(r))
        if auc_rs:
            mean_r = np.mean(auc_rs)
            std_r = np.std(auc_rs)
            cv = std_r / mean_r if mean_r > 0 else np.nan
            max_diff = max(auc_rs) - min(auc_rs)
            print(f"  {scope_label} AUCROC binned:")
            print(f"    Mean |r|:  {mean_r:.4f}")
            print(f"    Std:       {std_r:.4f}")
            print(f"    CV:        {cv:.4f}")
            print(f"    Range:     {max_diff:.4f} ({min(auc_rs):.4f} – {max(auc_rs):.4f})")
            if cv < 0.15:
                print(f"    → ROBUST (CV < 0.15)")
            elif cv < 0.30:
                print(f"    → MODERATE robustness (CV < 0.30)")
            else:
                print(f"    → LOW robustness (CV >= 0.30)")

    # ---- Plot comparison curves ----
    _plot_fold_comparison(all_fold_results, eval_metrics, K, k, b, output_dir)
    _plot_aucroc_overlay(all_fold_results, K, k, b, output_dir)

    return all_results


def _plot_fold_comparison(all_fold_results, eval_metrics, K, k, b, output_dir):
    """Plot per-fold binned curves in a grid: folds × scopes."""
    n_folds = len(all_fold_results)
    fig, axes = plt.subplots(n_folds, 3, figsize=(24, 5 * n_folds))
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    for fold_id, res in enumerate(all_fold_results):
        for col_idx, (scope_key, scope_label) in enumerate([
            ('seen_binned', 'Seen'),
            ('unseen_binned', 'Unseen'),
            ('combined_binned', 'Combined'),
        ]):
            ax = axes[fold_id, col_idx]
            binned = res[scope_key]
            ax.set_title(f'Fold {fold_id} — {scope_label}', fontsize=11, fontweight='bold')
            for m in eval_metrics:
                bd = binned[m].get('bin_dists', [])
                bp = binned[m].get('bin_perfs', [])
                if bd and bp:
                    r_val = abs(binned[m]['pearson_r']) if not np.isnan(binned[m]['pearson_r']) else 0
                    p_val = binned[m]['pearson_p']
                    sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
                    ax.plot(bd, bp, 'o-', label=f'{m} |r|={r_val:.2f}{sig}', markersize=4)
            ax.set_xlabel('LogDist', fontsize=9)
            ax.set_ylabel('Performance', fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'5-Fold CV: LogDist Generalization Curves (K={K}, k={k}, b={b})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cv_fold_comparison_all_metrics.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


def _plot_aucroc_overlay(all_fold_results, K, k, b, output_dir):
    """Overlay AUCROC curves from all folds on the same axes for direct comparison."""
    n_folds = len(all_fold_results)
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for col_idx, (scope_key, scope_label) in enumerate([
        ('seen_binned', 'Seen Epitopes'),
        ('unseen_binned', 'Unseen Epitopes'),
        ('combined_binned', 'Combined'),
    ]):
        ax = axes[col_idx]
        ax.set_title(scope_label, fontsize=13, fontweight='bold')
        for fold_id, res in enumerate(all_fold_results):
            binned = res[scope_key]
            bd = binned['aucroc'].get('bin_dists', [])
            bp = binned['aucroc'].get('bin_perfs', [])
            if bd and bp:
                r_val = binned['aucroc']['pearson_r']
                p_val = binned['aucroc']['pearson_p']
                abs_r = abs(r_val) if not np.isnan(r_val) else 0
                sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
                ax.plot(bd, bp, 'o-', color=colors[fold_id],
                        label=f'Fold {fold_id}  r={r_val:+.3f}{sig}',
                        markersize=5, linewidth=1.5)

        ax.set_xlabel('3-Chain LogDist', fontsize=11)
        ax.set_ylabel('AUCROC', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='')

    plt.suptitle(f'AUCROC vs LogDist — 5-Fold Overlay (K={K}, k={k}, b={b})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cv_aucroc_overlay.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Also plot per-epitope scatter overlay for unseen
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for fold_id, res in enumerate(all_fold_results):
        ep_df = res['ep_df']
        unseen_ep = ep_df[(ep_df['seen_status'] == 'unseen') & ep_df['aucroc'].notna()]
        if len(unseen_ep) > 0:
            ax.scatter(unseen_ep['distance'], unseen_ep['aucroc'],
                       color=colors[fold_id], alpha=0.4, s=15,
                       label=f'Fold {fold_id} ({len(unseen_ep)} ep)')

    ax.set_xlabel('Mean 3-Chain LogDist', fontsize=11)
    ax.set_ylabel('Per-Epitope AUCROC', fontsize=11)
    ax.set_title(f'Unseen Epitope AUCROC vs LogDist (K={K}, k={k}, b={b})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    path = os.path.join(output_dir, 'cv_unseen_epitope_scatter.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="5-Fold CV LogDist Evaluation Pipeline")
    parser.add_argument('--data-dir', type=str,
                        default='Data/tcr_seq/proc_files',
                        help='Parent directory containing tcr_cross_val_fold{0-4}/')
    parser.add_argument('--output-dir', type=str,
                        default='results/nettcr/cv_logdist',
                        help='Output directory for results')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='NetTCR training epochs')
    parser.add_argument('--chain', type=str, default='ab',
                        help='NetTCR chain type (a, b, ab)')
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
    print(f"5-Fold CV LogDist Evaluation")
    print(f"{'='*80}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}")
    print(f"  Bins: {args.bin_num}")
    print(f"  Chains: {chain_cols}")
    print(f"  Epochs: {args.epochs}")
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
            # Step 1: Convert to NetTCR format
            print(f"\n  [Step 1] Preparing NetTCR format...")
            nettcr_files = prepare_nettcr_format(fold_dir, fold_output)

            # Step 2: Train and predict
            print(f"\n  [Step 2] Training NetTCR...")
            train_nettcr_fold(nettcr_files, fold_output, args.epochs, args.chain)

            # Step 3: Add epitope_seen labels
            print(f"\n  [Step 3] Adding epitope_seen labels...")
            add_epitope_seen_labels(fold_output, nettcr_files['train.csv'])
        else:
            # Verify predictions exist
            for name in ['val_predictions_with_label.csv', 'test_predictions_with_label.csv',
                         'train.csv']:
                path = os.path.join(fold_output, name)
                if not os.path.exists(path):
                    print(f"  ERROR: {path} not found. Remove --skip-training to generate.")
                    sys.exit(1)

        # Step 4: LogDist evaluation
        res = evaluate_fold_logdist(
            fold_id, fold_output, chain_cols,
            args.K, args.k, args.b, args.bin_num, args.eval_metrics)
        all_fold_results.append(res)

        # Save per-fold results
        res['results_df'].to_csv(
            os.path.join(fold_output, 'logdist_correlations.csv'), index=False)
        res['ep_df'].to_csv(
            os.path.join(fold_output, 'per_epitope_metrics.csv'), index=False)

    # Step 5: Cross-fold comparison
    all_results = compare_folds(
        all_fold_results, args.eval_metrics, args.K, args.k, args.b, output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")
    print(f"  Output: {output_dir}")
    print(f"  Files:")
    print(f"    all_folds_correlations.csv    — all fold × metric correlations")
    print(f"    aucroc_binned_summary.csv     — AUCROC binned summary")
    print(f"    cv_fold_comparison_all_metrics.png — per-fold curves grid")
    print(f"    cv_aucroc_overlay.png         — AUCROC overlay across folds")
    print(f"    cv_unseen_epitope_scatter.png  — per-epitope scatter")
    for f in range(args.n_folds):
        print(f"    fold{f}/logdist_correlations.csv")
        print(f"    fold{f}/per_epitope_metrics.csv")


if __name__ == '__main__':
    main()
