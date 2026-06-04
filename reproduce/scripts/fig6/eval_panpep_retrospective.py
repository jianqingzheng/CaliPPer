#!/usr/bin/env python3
"""
PanPep Retrospective S2DD Validation.

Uses PanPep's published zero-shot test data (491 unseen peptides) and
majority-shot test data (25 peptides, balanced labels) to evaluate
S2DD v2.6 performance prediction and Bayesian recalibration.

Model: PanPep meta-learning (Gao et al., Nature Machine Intelligence 2023)
Data:  Model/PanPep/Data/ (GitHub repo)
       Model/deepAntigen/.../source_data.xlsx (Supp Fig 18, neoantigen)

Three test scenarios:
  1. Majority-shot: 5230 pairs, 25 peptides, balanced labels → AUROC/AP/F1
  2. Zero-shot: 857 positive + generated negatives, 491 unseen peptides
  3. Neoantigen cross-model: 384 pairs from deepAntigen Supp Fig 18

Key settings (per CLAUDE.md):
  - TCR params: k=0.1, b=0.1, K=50, bin_num=8
  - Chain cols: ['peptide', 'binding_TCR'] (2-chain, PanPep has no CDR3α)
  - Weight formula: sigma_C
  - Distance: Levenshtein for both chains (BLOSUM comparison run separately)
  - Label column: label (majority) or generated (zero-shot)
  - Recalibration: v2.6 vanilla PPV/NPV sigmoid, threshold=0.5, lambda=0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Project root
# CaliPPer self-contained bootstrap
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR  # also adds CaliPPer/ to sys.path
from pathlib import Path
PROJECT_ROOT = Path(INPUT_DIR)


from calipper.general_evaluator import (
    compute_pairwise_ratios, logdist_from_ratios, safe_metric
)
from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
    PERFORMANCE_PREDICTION_LAM, CALIBRATION_LAM
)
from calipper.core import predict_metric, fit_recalibration, apply_recalibration

# === Configuration ===
PANPEP_DIR = ROOT / "Model" / "PanPep"
DATA_DIR = PANPEP_DIR / "Data"
RESULTS_DIR = ROOT / "results" / "panpep_retrospective"
CACHE_DIR = RESULTS_DIR

# S2DD v2.6 settings (TCR defaults)
k, b, K = 0.1, 0.1, 50
N_BINS = 8  # for degradation analysis (NOT recalibration — uses v2.7 adaptive defaults)
CHAIN_COLS = ['peptide', 'binding_TCR']  # 2-chain: peptide + CDR3β

# ============================================================
# Data loading
# ============================================================

def load_panpep_data():
    """Load all PanPep datasets with unified column names."""
    meta = pd.read_csv(DATA_DIR / "meta_dataset.csv")
    meta.columns = ['peptide', 'binding_TCR', 'label']

    base = pd.read_csv(DATA_DIR / "base_dataset.csv")
    base.columns = ['peptide', 'binding_TCR', 'label']

    zero = pd.read_csv(DATA_DIR / "zero_dataset.csv")
    zero.columns = ['peptide', 'binding_TCR', 'label']

    maj_train = pd.read_csv(DATA_DIR / "majority_training_dataset.csv")
    maj_train.columns = ['peptide', 'binding_TCR', 'label']

    maj_test = pd.read_csv(DATA_DIR / "majority_testing_dataset.csv")
    maj_test.columns = ['peptide', 'binding_TCR', 'label']

    return {
        'meta': meta, 'base': base, 'zero': zero,
        'maj_train': maj_train, 'maj_test': maj_test
    }


def load_neoantigen_data():
    """Load PanPep predictions from deepAntigen source data (Supp Fig 18)."""
    f = (ROOT / "Model" / "deepAntigen" /
         "clinical_cancer_patients" / "Data" / "tcr_seq" /
         "proc_files" / "deepantigen_data" / "source_data.xlsx")
    df = pd.read_excel(f, sheet_name='Supplementary Fig.18', header=1)
    # PanPep columns at indices 5-8
    panpep = df.iloc[:, 5:9].copy()
    panpep.columns = ['peptide', 'binding_TCR', 'prediction', 'label']
    panpep = panpep[panpep['label'] == 1].copy()
    panpep['prediction'] = panpep['prediction'].astype(float)
    panpep['label'] = panpep['label'].astype(int)
    return panpep


# ============================================================
# PanPep inference
# ============================================================

def run_panpep_inference(test_df, mode='zero-shot', update_step=3):
    """Run PanPep inference on test data.

    Args:
        test_df: DataFrame with 'peptide', 'binding_TCR', optionally 'label'
        mode: 'zero-shot', 'few-shot', or 'majority'
        update_step: fine-tuning steps (default 3 for few-shot)

    Returns:
        DataFrame with 'peptide', 'binding_TCR', 'prediction' columns
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fin:
        if mode == 'zero-shot':
            # Zero-shot: only Peptide, CDR3
            out = test_df[['peptide', 'binding_TCR']].copy()
            out.columns = ['Peptide', 'CDR3']
            # Must be sorted by peptide
            out = out.sort_values('Peptide')
            out.to_csv(fin.name, index=False)
        else:
            # Few-shot / majority: Peptide, CDR3, Label
            out = test_df[['peptide', 'binding_TCR', 'label']].copy()
            out.columns = ['Peptide', 'CDR3', 'Label']
            out = out.sort_values('Peptide')
            out.to_csv(fin.name, index=False)
        input_path = fin.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fout:
        output_path = fout.name

    cmd = [
        sys.executable, str(PANPEP_DIR / "PanPep.py"),
        "--learning_setting", mode,
        "--input", input_path,
        "--output", output_path,
        "--update_step_test", str(update_step),
    ]

    print(f"  Running PanPep ({mode})...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(PANPEP_DIR), timeout=600)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")

    # Read predictions
    pred_df = pd.read_csv(output_path)
    pred_df.columns = ['peptide', 'binding_TCR', 'prediction']

    os.unlink(input_path)
    os.unlink(output_path)

    return pred_df


# ============================================================
# S2DD distance computation
# ============================================================

def compute_s2dd_distances(test_df, train_df, cache_name=None):
    """Compute S2DD distances with caching."""
    cache_path = CACHE_DIR / f"distance_cache_{cache_name}.npz" if cache_name else None

    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        if len(data['distances']) == len(test_df):
            print(f"  Loaded cached distances ({cache_name})")
            return data['distances']
        print(f"  Cache size mismatch, recomputing...")

    print(f"  Computing S2DD distances ({len(test_df)} test × {len(train_df)} train)...")
    weights, _ = compute_chain_weights(train_df, CHAIN_COLS, k, b, K, formula='sigma_C')
    print(f"  Chain weights: {dict(zip(CHAIN_COLS, weights))}")

    distances = compute_combine_first_distances(
        test_df, train_df, CHAIN_COLS, weights, k, b, K,
        combine_method='weighted_max_znorm'
    )

    if cache_path:
        np.savez(cache_path, distances=distances)
        print(f"  Cached to {cache_path}")

    return distances


# ============================================================
# Scenario 1: Majority-shot evaluation
# ============================================================

def run_majority_evaluation(data):
    """S2DD analysis on majority-shot test data (balanced labels, 25 peptides)."""
    print("\n" + "="*60)
    print("SCENARIO 1: Majority-shot evaluation")
    print("="*60)

    maj_train = data['maj_train']
    maj_test = data['maj_test']
    meta = data['meta']  # training reference for S2DD distances

    out_dir = RESULTS_DIR / "reproduction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run PanPep inference (zero-shot mode on majority test data)
    # Use zero-shot mode to test the pre-trained model's generalization —
    # this is what S2DD measures (distance from training → performance).
    # IMPORTANT: Deduplicate test data before merge to avoid N² inflation
    # (majority_testing_dataset has 43 duplicated positive pairs).
    pred_cache = out_dir / "majority_test_predictions.csv"
    if pred_cache.exists():
        print(f"  Using cached predictions: {pred_cache}")
        pred_df = pd.read_csv(pred_cache)
    else:
        # Deduplicate test set first (keep first occurrence for label)
        maj_test_dedup = maj_test.drop_duplicates(subset=['peptide', 'binding_TCR'], keep='first')
        print(f"  Deduped majority test: {len(maj_test)} → {len(maj_test_dedup)} rows")

        pred_df = run_panpep_inference(maj_test_dedup, mode='zero-shot')
        # Merge back labels from deduplicated test set (1:1 match)
        pred_df = pred_df.merge(
            maj_test_dedup[['peptide', 'binding_TCR', 'label']],
            on=['peptide', 'binding_TCR'],
            how='inner'
        )
        pred_df.to_csv(pred_cache, index=False)
        print(f"  Saved {len(pred_df)} predictions to {pred_cache}")

    # Step 2: Compute S2DD distances (from meta training → test)
    distances = compute_s2dd_distances(pred_df, meta, cache_name='majority')
    pred_df['distance'] = distances

    # Step 3: Overall metrics
    y = pred_df['label'].values.astype(int)
    p = pred_df['prediction'].values.astype(float)
    auroc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    f1 = f1_score(y, (p > 0.5).astype(int))
    print(f"\n  Overall: AUROC={auroc:.3f}, AP={ap:.3f}, F1={f1:.3f} (n={len(y)})")

    # Step 4: Per-peptide metrics
    pep_results = []
    for pep in sorted(pred_df['peptide'].unique()):
        sub = pred_df[pred_df['peptide'] == pep]
        yi = sub['label'].values.astype(int)
        pi = sub['prediction'].values.astype(float)
        di = sub['distance'].values
        n = len(yi)
        prev = yi.mean()
        metrics = {}
        for m_name, m_func in [('aucroc', roc_auc_score), ('ap', average_precision_score)]:
            try:
                metrics[m_name] = safe_metric(m_name, yi, pi)
            except Exception:
                metrics[m_name] = np.nan
        metrics['f1'] = f1_score(yi, (pi > 0.5).astype(int))
        pep_results.append({
            'peptide': pep, 'n_samples': n, 'prevalence': prev,
            'mean_dist': di.mean(), 'std_dist': di.std(),
            **metrics
        })

    pep_df = pd.DataFrame(pep_results)
    pep_df.to_csv(out_dir / "per_peptide_performance.csv", index=False)
    print(f"\n  Per-peptide metrics saved ({len(pep_df)} peptides)")

    # Step 5: S2DD degradation (bin by distance)
    deg_dir = RESULTS_DIR / "s2dd_degradation"
    deg_dir.mkdir(parents=True, exist_ok=True)

    sort_idx = np.argsort(distances)
    bs = len(distances) // N_BINS
    bins = []
    for i in range(N_BINS):
        s = i * bs
        e = (i + 1) * bs if i < N_BINS - 1 else len(distances)
        idx = sort_idx[s:e]
        yi = y[idx]
        pi = p[idx]
        di = distances[idx]
        row = {
            'bin': i, 'mean_dist': di.mean(), 'n_samples': len(idx),
            'prevalence': yi.mean()
        }
        for m_name in ['aucroc', 'ap', 'f1']:
            try:
                if m_name == 'f1':
                    row[m_name] = f1_score(yi, (pi > 0.5).astype(int))
                else:
                    row[m_name] = safe_metric(m_name, yi, pi)
            except Exception:
                row[m_name] = np.nan
        bins.append(row)

    bin_df = pd.DataFrame(bins)
    bin_df.to_csv(deg_dir / "majority_degradation_curves.csv", index=False)

    # Correlations
    from scipy.stats import pearsonr
    for metric in ['aucroc', 'ap', 'f1']:
        vals = bin_df[metric].dropna()
        dists = bin_df.loc[vals.index, 'mean_dist']
        if len(vals) >= 4:
            r, pval = pearsonr(dists, vals)
            print(f"  Degradation {metric}: r={r:.3f} (p={pval:.3f})")

    # Step 6: Performance prediction (LOO across peptides)
    pred_dir = RESULTS_DIR / "performance_prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)

    peptides = sorted(pred_df['peptide'].unique())
    if len(peptides) >= 3:
        loo_results = []
        for held_out in peptides:
            cal_data = {}
            for pep in peptides:
                if pep == held_out:
                    continue
                sub = pred_df[pred_df['peptide'] == pep]
                cal_data[pep] = (
                    sub['label'].values.astype(int),
                    sub['prediction'].values.astype(float),
                    sub['distance'].values
                )
            test_sub = pred_df[pred_df['peptide'] == held_out]
            test_y = test_sub['label'].values.astype(int)
            test_p = test_sub['prediction'].values.astype(float)
            test_d = test_sub['distance'].values

            for metric in ['aucroc', 'ap', 'f1']:
                try:
                    actual = safe_metric(metric, test_y, test_p) if metric != 'f1' \
                        else f1_score(test_y, (test_p > 0.5).astype(int))
                except Exception:
                    actual = np.nan

                try:
                    result = predict_metric(cal_data, test_p, test_d,
                                            metrics=[metric], n_bins=N_BINS)
                    predicted = result['estimated'].get(metric, np.nan)
                except Exception as e:
                    predicted = np.nan

                loo_results.append({
                    'peptide': held_out, 'metric': metric,
                    'actual': actual, 'predicted': predicted,
                    'abs_error': abs(actual - predicted) if not (np.isnan(actual) or np.isnan(predicted)) else np.nan,
                    'n_samples': len(test_y)
                })

        loo_df = pd.DataFrame(loo_results)
        loo_df.to_csv(pred_dir / "majority_loo_prediction.csv", index=False)

        # Summary
        for metric in ['aucroc', 'ap', 'f1']:
            sub = loo_df[loo_df['metric'] == metric].dropna(subset=['abs_error'])
            if len(sub) > 0:
                mae = sub['abs_error'].mean()
                r, _ = pearsonr(sub['actual'], sub['predicted']) if len(sub) >= 3 else (np.nan, np.nan)
                print(f"  LOO {metric}: MAE={mae:.3f}, R={r:.3f} (n={len(sub)})")

    # Step 7: Bayesian recalibration (within-test half-split)
    recal_dir = RESULTS_DIR / "neoantigen_recalibration"
    recal_dir.mkdir(parents=True, exist_ok=True)

    # Split peptides into cal/test halves
    np.random.seed(42)
    pep_shuf = np.random.permutation(peptides)
    cal_peps = set(pep_shuf[:len(pep_shuf)//2])
    test_peps = set(pep_shuf[len(pep_shuf)//2:])

    cal_data = {}
    for pep in cal_peps:
        sub = pred_df[pred_df['peptide'] == pep]
        cal_data[pep] = (
            sub['label'].values.astype(int),
            sub['prediction'].values.astype(float),
            sub['distance'].values
        )

    test_sub = pred_df[pred_df['peptide'].isin(test_peps)]
    test_y = test_sub['label'].values.astype(int)
    test_p = test_sub['prediction'].values.astype(float)
    test_d = test_sub['distance'].values

    try:
        ppv_params, npv_params, p_pos, p_neg = fit_recalibration(cal_data)
        cal_scores = apply_recalibration(
            test_y, test_p, test_d, ppv_params, npv_params, p_pos, p_neg)

        recal_results = []
        for metric in ['aucroc', 'ap', 'f1']:
            if metric == 'f1':
                before = f1_score(test_y, (test_p > 0.5).astype(int))
                after = f1_score(test_y, (cal_scores > 0.5).astype(int))
            else:
                before = safe_metric(metric, test_y, test_p)
                after = safe_metric(metric, test_y, cal_scores)
            recal_results.append({
                'metric': metric, 'n_samples': len(test_y),
                'before': before, 'after': after,
                'delta': after - before
            })
            print(f"  Recalibration {metric}: {before:.3f} → {after:.3f} (Δ={after-before:+.3f})")

        recal_df = pd.DataFrame(recal_results)
        recal_df.to_csv(recal_dir / "majority_recalibration.csv", index=False)
    except Exception as e:
        print(f"  Recalibration failed: {e}")

    return pred_df


# ============================================================
# Scenario 2: Zero-shot evaluation
# ============================================================

def run_zero_shot_evaluation(data):
    """S2DD analysis on zero-shot test data (unseen peptides)."""
    print("\n" + "="*60)
    print("SCENARIO 2: Zero-shot evaluation")
    print("="*60)

    zero = data['zero']
    meta = data['meta']

    out_dir = RESULTS_DIR / "reproduction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run PanPep inference (zero-shot mode)
    pred_cache = out_dir / "zeroshot_test_predictions.csv"
    if pred_cache.exists():
        print(f"  Using cached predictions: {pred_cache}")
        pred_df = pd.read_csv(pred_cache)
    else:
        pred_df = run_panpep_inference(zero, mode='zero-shot')
        pred_df['label'] = 1  # all positive in zero-shot dataset
        pred_df.to_csv(pred_cache, index=False)
        print(f"  Saved {len(pred_df)} zero-shot predictions")

    # Step 2: Generate negative samples for balanced evaluation
    # Standard protocol: for each positive (peptide, CDR3), create a negative
    # by pairing the CDR3 with a random different peptide
    np.random.seed(42)
    all_peps = pred_df['peptide'].unique()
    neg_rows = []
    for _, row in pred_df.iterrows():
        other_peps = [p for p in all_peps if p != row['peptide']]
        neg_pep = np.random.choice(other_peps)
        neg_rows.append({
            'peptide': neg_pep, 'binding_TCR': row['binding_TCR'],
            'label': 0
        })
    neg_df = pd.DataFrame(neg_rows)

    # Run PanPep on negatives
    neg_pred_cache = out_dir / "zeroshot_neg_predictions.csv"
    if neg_pred_cache.exists():
        neg_pred_df = pd.read_csv(neg_pred_cache)
    else:
        neg_pred_df = run_panpep_inference(neg_df, mode='zero-shot')
        neg_pred_df['label'] = 0
        neg_pred_df.to_csv(neg_pred_cache, index=False)

    # Combine positive + negative
    combined = pd.concat([pred_df, neg_pred_df], ignore_index=True)
    print(f"  Combined: {len(combined)} pairs ({combined['label'].sum()} pos, "
          f"{(combined['label']==0).sum()} neg)")

    # Step 3: Compute S2DD distances
    distances = compute_s2dd_distances(combined, meta, cache_name='zeroshot')
    combined['distance'] = distances

    # Step 4: Overall metrics
    y = combined['label'].values.astype(int)
    p = combined['prediction'].values.astype(float)
    auroc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    f1 = f1_score(y, (p > 0.5).astype(int))
    print(f"\n  Overall: AUROC={auroc:.3f}, AP={ap:.3f}, F1={f1:.3f}")

    # Step 5: S2DD degradation
    deg_dir = RESULTS_DIR / "s2dd_degradation"
    deg_dir.mkdir(parents=True, exist_ok=True)

    sort_idx = np.argsort(distances)
    bs = len(distances) // N_BINS
    bins = []
    for i in range(N_BINS):
        s = i * bs
        e = (i + 1) * bs if i < N_BINS - 1 else len(distances)
        idx = sort_idx[s:e]
        yi = y[idx]
        pi = p[idx]
        di = distances[idx]
        row = {
            'bin': i, 'mean_dist': di.mean(), 'n_samples': len(idx),
            'prevalence': yi.mean()
        }
        for m_name in ['aucroc', 'ap', 'f1']:
            try:
                if m_name == 'f1':
                    row[m_name] = f1_score(yi, (pi > 0.5).astype(int))
                else:
                    row[m_name] = safe_metric(m_name, yi, pi)
            except Exception:
                row[m_name] = np.nan
        bins.append(row)

    bin_df = pd.DataFrame(bins)
    bin_df.to_csv(deg_dir / "zeroshot_degradation_curves.csv", index=False)

    from scipy.stats import pearsonr
    for metric in ['aucroc', 'ap', 'f1']:
        vals = bin_df[metric].dropna()
        dists = bin_df.loc[vals.index, 'mean_dist']
        if len(vals) >= 4:
            r, pval = pearsonr(dists, vals)
            print(f"  Degradation {metric}: r={r:.3f} (p={pval:.3f})")

    # Save summary
    summary = pd.DataFrame([{
        'scenario': 'zero-shot',
        'n_samples': len(combined),
        'n_peptides': combined['peptide'].nunique(),
        'auroc': auroc, 'ap': ap, 'f1': f1
    }])
    summary.to_csv(out_dir / "zeroshot_summary.csv", index=False)

    return combined


# ============================================================
# Scenario 3: Neoantigen cross-model
# ============================================================

def run_neoantigen_evaluation(data):
    """Analyze PanPep predictions on deepAntigen neoantigen data."""
    print("\n" + "="*60)
    print("SCENARIO 3: Neoantigen cross-model comparison")
    print("="*60)

    meta = data['meta']
    neo_df = load_neoantigen_data()

    out_dir = RESULTS_DIR / "neoantigen_recalibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify seen vs unseen neoantigens relative to PanPep training
    train_peps = set(meta['peptide'].unique()) | set(data['base']['peptide'].unique())
    neo_df['seen_in_training'] = neo_df['peptide'].isin(train_peps)
    n_seen = neo_df['seen_in_training'].sum()
    n_unseen = (~neo_df['seen_in_training']).sum()
    print(f"  Neoantigens: {n_seen} seen, {n_unseen} unseen in PanPep training")

    # Compute S2DD distances
    distances = compute_s2dd_distances(neo_df, meta, cache_name='neoantigen')
    neo_df['distance'] = distances

    # Analyze score distributions
    for group, sub in neo_df.groupby('seen_in_training'):
        label = 'Seen' if group else 'Unseen'
        print(f"  {label}: n={len(sub)}, mean_score={sub['prediction'].mean():.3f}, "
              f"mean_dist={sub['distance'].mean():.3f}")

    # Save
    neo_df.to_csv(out_dir / "neoantigen_panpep_analysis.csv", index=False)

    # Distance distribution comparison
    from scipy.stats import mannwhitneyu
    seen = neo_df[neo_df['seen_in_training']]['distance']
    unseen = neo_df[~neo_df['seen_in_training']]['distance']
    if len(seen) > 0 and len(unseen) > 0:
        u, p = mannwhitneyu(seen, unseen, alternative='two-sided')
        print(f"  Distance MW-U: p={p:.4f} (seen closer? "
              f"seen_mean={seen.mean():.3f} vs unseen_mean={unseen.mean():.3f})")

    return neo_df


# ============================================================
# Main
# ============================================================

def main():
    print("PanPep Retrospective S2DD Validation")
    print("=" * 60)

    # Load data
    print("\nLoading PanPep data...")
    data = load_panpep_data()
    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, {df['peptide'].nunique()} peptides")

    # Run scenarios
    run_majority_evaluation(data)
    run_zero_shot_evaluation(data)
    run_neoantigen_evaluation(data)

    print("\n" + "=" * 60)
    print("All scenarios complete. Results in:", RESULTS_DIR)


if __name__ == '__main__':
    main()
