#!/usr/bin/env python3
"""
BigMHC Retrospective S2DD Validation.

Uses BigMHC's published MANAFEST clinical data (837 neoantigen candidates,
167 immunogenic) with pre-computed predictions from 10+ models to evaluate
S2DD v2.6 performance prediction and Bayesian recalibration.

Model: BigMHC IM — pan-allelic BiLSTM immunogenicity predictor
       (Albert et al., Nature Machine Intelligence 2023)
Data:  Mendeley DOI: 10.17632/dvmz6pkzvb (v4)
       All predictions pre-computed in im_test.csv

Key settings:
  - Chains: ['pep'] (peptide-only; MHC allele Levenshtein not biologically meaningful)
  - TCR params: k=0.1, b=0.1, K=50, bin_num=8
  - Weight formula: sigma_C
  - Distance: Levenshtein for both chains
  - Label column: tgt (0=non-immunogenic, 1=immunogenic)
  - Training overlap: 1/830 MANAFEST peptides in IM training (negligible)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.general_evaluator import safe_metric
from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances
)
from calipper.core import predict_metric, fit_recalibration, apply_recalibration

# === Configuration ===
MENDELEY_DIR = Path(INPUT_DIR) / "Data" / "retrospective_bigmhc" / "mendeley_data" / "extracted" / "BigMHC Training and Evaluation Data"
RESULTS_DIR = Path(OUTPUT_DIR) / "bigmhc_retrospective"

k, b, K = 0.1, 0.1, 50
N_BINS = 8  # for degradation analysis (NOT recalibration — uses v2.7 adaptive defaults)
CHAIN_COLS = ['pep']  # peptide-only (MHC allele name Levenshtein is not biologically meaningful)

# Models to analyze (score columns from im_test.csv)
MODELS = {
    'BigMHC_IM': 'BigMHC_IM',
    'BigMHC_EL': 'BigMHC_EL',
    'BigMHC_ELIM': 'BigMHC_ELIM',
    'NetMHCpan': 'NetMHCpan-4.1_Scores',
    'MHCflurry': 'MHCflurry-2.0_Scores',
    'PRIME': 'PRIME-2.0_Scores',
}


def load_data():
    """Load and merge MANAFEST + IM train/test data."""
    manafest = pd.read_csv(MENDELEY_DIR / "manafest.csv")
    im_train = pd.read_csv(MENDELEY_DIR / "im_train.csv")
    im_test = pd.read_csv(MENDELEY_DIR / "im_test.csv")

    # Merge MANAFEST with IM test to get predictions
    merged = manafest.merge(im_test, on=['mhc', 'pep'], how='inner', suffixes=('', '_te'))
    merged['label'] = merged['tgt'].astype(int)

    print(f"MANAFEST: {len(manafest)} rows, {manafest['tgt'].sum()} immunogenic")
    print(f"IM Train: {len(im_train)} rows, {im_train['pep'].nunique()} peptides")
    print(f"Merged (MANAFEST ∩ IM Test): {len(merged)} rows")

    # Check training overlap
    train_peps = set(im_train['pep'].unique())
    test_peps = set(merged['pep'].unique())
    overlap = train_peps & test_peps
    print(f"Training overlap: {len(overlap)}/{len(test_peps)} peptides")

    return merged, im_train


def compute_distances(test_df, train_df, cache_name='manafest'):
    """Compute S2DD distances with caching."""
    cache_path = RESULTS_DIR / f"distance_cache_{cache_name}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        if len(data['distances']) == len(test_df):
            print(f"  Loaded cached distances ({cache_name})")
            return data['distances']

    print(f"  Computing S2DD distances ({len(test_df)} test × {len(train_df)} train)...")
    weights, _ = compute_chain_weights(train_df, CHAIN_COLS, k, b, K, formula='sigma_C')
    print(f"  sigma_C weights: {dict(zip(CHAIN_COLS, weights))}")

    distances = compute_combine_first_distances(
        test_df, train_df, CHAIN_COLS, weights, k, b, K,
        combine_method='weighted_max_znorm'
    )

    np.savez(cache_path, distances=distances)
    print(f"  Cached to {cache_path}")
    return distances


def run_degradation(merged, distances, model_col, model_name):
    """Compute S2DD degradation curves for a given model."""
    y = merged['label'].values
    p = merged[model_col].values.astype(float)
    valid = ~np.isnan(p)
    y, p, d = y[valid], p[valid], distances[valid]

    sort_idx = np.argsort(d)
    bs = len(d) // N_BINS
    bins = []
    for i in range(N_BINS):
        s = i * bs
        e = (i + 1) * bs if i < N_BINS - 1 else len(d)
        idx = sort_idx[s:e]
        yi, pi, di = y[idx], p[idx], d[idx]
        row = {'bin': i, 'mean_dist': di.mean(), 'n_samples': len(idx),
               'prevalence': yi.mean(), 'model': model_name}
        import sys as _s_bigm
        for m in ['aucroc', 'ap']:
            try:
                row[m] = safe_metric(m, yi, pi)
            except Exception as _e_m:
                print(f"  ⚠ FALLBACK [bigmhc_bins]: model={model_name} bin={i} metric={m} failed ({type(_e_m).__name__}); NaN", file=_s_bigm.stderr, flush=True)
                row[m] = np.nan
        try:
            row['f1'] = f1_score(yi, (pi > 0.5).astype(int))
        except Exception as _e_f1:
            print(f"  ⚠ FALLBACK [bigmhc_bins]: model={model_name} bin={i} f1_score failed ({type(_e_f1).__name__}); NaN", file=_s_bigm.stderr, flush=True)
            row['f1'] = np.nan
        bins.append(row)
    return pd.DataFrame(bins)


def main():
    print("BigMHC Retrospective S2DD Validation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ['reproduction', 's2dd_degradation', 'performance_prediction', 'manafest_recalibration', 'cross_model']:
        (RESULTS_DIR / subdir).mkdir(exist_ok=True)

    # Load data
    merged, im_train = load_data()

    # Compute S2DD distances
    distances = compute_distances(merged, im_train)
    merged['distance'] = distances

    # === Reproduction ===
    print("\n" + "=" * 60)
    print("REPRODUCTION: Model performance on MANAFEST")
    print("=" * 60)
    repro = []
    y = merged['label'].values
    for name, col in MODELS.items():
        p = merged[col].values.astype(float)
        m = ~np.isnan(p)
        auroc = roc_auc_score(y[m], p[m])
        ap = average_precision_score(y[m], p[m])
        f1 = f1_score(y[m], (p[m] > 0.5).astype(int))
        repro.append({'model': name, 'auroc': auroc, 'ap': ap, 'f1': f1, 'n': m.sum()})
        print(f"  {name}: AUROC={auroc:.3f}, AP={ap:.3f}, F1={f1:.3f}")
    pd.DataFrame(repro).to_csv(RESULTS_DIR / "reproduction" / "model_performance.csv", index=False)

    # === S2DD Degradation (all models) ===
    print("\n" + "=" * 60)
    print("S2DD DEGRADATION")
    print("=" * 60)
    all_bins = []
    for name, col in MODELS.items():
        bin_df = run_degradation(merged, distances, col, name)
        all_bins.append(bin_df)
        vals = bin_df['ap'].dropna()
        dists = bin_df.loc[vals.index, 'mean_dist']
        if len(vals) >= 4:
            r, pval = pearsonr(dists, vals)
            print(f"  {name} AP degradation: r={r:.3f} (p={pval:.3f})")
    all_bin_df = pd.concat(all_bins, ignore_index=True)
    all_bin_df.to_csv(RESULTS_DIR / "s2dd_degradation" / "all_models_degradation.csv", index=False)

    # === Performance Prediction (LOO across HLA groups) ===
    print("\n" + "=" * 60)
    print("PERFORMANCE PREDICTION (LOO across HLA allele groups)")
    print("=" * 60)

    # Group by HLA allele for LOO
    model_col = 'BigMHC_IM'
    p_all = merged[model_col].values.astype(float)
    hla_groups = merged.groupby('mhc').filter(lambda x: len(x) >= 10)['mhc'].unique()
    print(f"  HLA groups with ≥10 samples: {len(hla_groups)}")

    if len(hla_groups) >= 3:
        loo_results = []
        for held_out in hla_groups:
            cal_data = {}
            for hla in hla_groups:
                if hla == held_out:
                    continue
                sub = merged[merged['mhc'] == hla]
                cal_data[hla] = (
                    sub['label'].values,
                    sub[model_col].values.astype(float),
                    sub['distance'].values
                )
            test_sub = merged[merged['mhc'] == held_out]
            test_y = test_sub['label'].values
            test_p = test_sub[model_col].values.astype(float)
            test_d = test_sub['distance'].values

            for metric in ['aucroc', 'ap', 'f1']:
                import sys as _s_loo
                try:
                    actual = safe_metric(metric, test_y, test_p) if metric != 'f1' \
                        else f1_score(test_y, (test_p > 0.5).astype(int))
                except Exception as _e_a:
                    print(f"  ⚠ FALLBACK [bigmhc_loo]: metric={metric} actual computation failed ({type(_e_a).__name__}); NaN", file=_s_loo.stderr, flush=True)
                    actual = np.nan
                try:
                    result = predict_metric(cal_data, test_p, test_d,
                                            metrics=[metric], n_bins=N_BINS)
                    predicted = result['estimated'].get(metric, np.nan)
                except Exception as _e_p:
                    print(f"  ⚠ FALLBACK [bigmhc_loo]: metric={metric} predict_metric failed ({type(_e_p).__name__}); NaN", file=_s_loo.stderr, flush=True)
                    predicted = np.nan

                loo_results.append({
                    'hla': held_out, 'metric': metric,
                    'actual': actual, 'predicted': predicted,
                    'abs_error': abs(actual - predicted) if not (np.isnan(actual) or np.isnan(predicted)) else np.nan,
                    'n_samples': len(test_y)
                })

        loo_df = pd.DataFrame(loo_results)
        loo_df.to_csv(RESULTS_DIR / "performance_prediction" / "hla_loo_prediction.csv", index=False)
        for metric in ['aucroc', 'ap', 'f1']:
            sub = loo_df[loo_df['metric'] == metric].dropna(subset=['abs_error'])
            if len(sub) >= 3:
                mae = sub['abs_error'].mean()
                r, _ = pearsonr(sub['actual'], sub['predicted'])
                print(f"  LOO {metric}: MAE={mae:.3f}, R={r:.3f} (n={len(sub)})")

    # === Bayesian Recalibration (all models) ===
    print("\n" + "=" * 60)
    print("BAYESIAN RECALIBRATION (half-split across HLA alleles)")
    print("=" * 60)

    np.random.seed(42)
    all_hlas = merged['mhc'].unique()
    hla_shuf = np.random.permutation(all_hlas)
    cal_hlas = set(hla_shuf[:len(hla_shuf) // 2])
    test_hlas = set(hla_shuf[len(hla_shuf) // 2:])

    recal_results = []
    for name, col in MODELS.items():
        cal_data = {}
        for hla in cal_hlas:
            sub = merged[merged['mhc'] == hla]
            if len(sub) < 2:
                continue
            cal_data[hla] = (
                sub['label'].values,
                sub[col].values.astype(float),
                sub['distance'].values
            )

        test_sub = merged[merged['mhc'].isin(test_hlas)]
        test_y = test_sub['label'].values
        test_p = test_sub[col].values.astype(float)
        test_d = test_sub['distance'].values

        try:
            ppv_params, npv_params, p_pos, p_neg = fit_recalibration(cal_data)
            cal_scores = apply_recalibration(
                test_y, test_p, test_d, ppv_params, npv_params, p_pos, p_neg)

            for metric in ['aucroc', 'ap', 'f1']:
                if metric == 'f1':
                    before = f1_score(test_y, (test_p > 0.5).astype(int))
                    after = f1_score(test_y, (cal_scores > 0.5).astype(int))
                else:
                    before = safe_metric(metric, test_y, test_p)
                    after = safe_metric(metric, test_y, cal_scores)
                recal_results.append({
                    'model': name, 'metric': metric,
                    'n_samples': len(test_y),
                    'before': before, 'after': after,
                    'delta': after - before
                })
            print(f"  {name}: AUROC {recal_results[-3]['before']:.3f}→{recal_results[-3]['after']:.3f} "
                  f"(Δ={recal_results[-3]['delta']:+.3f}), "
                  f"AP {recal_results[-2]['before']:.3f}→{recal_results[-2]['after']:.3f} "
                  f"(Δ={recal_results[-2]['delta']:+.3f})")
        except Exception as e:
            import sys as _s_recal
            print(f"  ⚠ FALLBACK [bigmhc-recal]: name={name} recalibration failed ({type(e).__name__}: {e}); skipping this name", file=_s_recal.stderr, flush=True)

    recal_df = pd.DataFrame(recal_results)
    recal_df.to_csv(RESULTS_DIR / "manafest_recalibration" / "all_models_recalibration.csv", index=False)

    # === Distance Analysis ===
    print("\n" + "=" * 60)
    print("DISTANCE ANALYSIS")
    print("=" * 60)

    # Immunogenic vs non-immunogenic distance
    imm = merged[merged['label'] == 1]['distance']
    non = merged[merged['label'] == 0]['distance']
    u, p = mannwhitneyu(imm, non, alternative='two-sided')
    print(f"  Immunogenic mean dist: {imm.mean():.3f}, Non-immunogenic: {non.mean():.3f}")
    print(f"  MW-U: p={p:.4f}")

    # Save merged data with distances
    merged.to_csv(RESULTS_DIR / "reproduction" / "manafest_with_distances.csv", index=False)

    print("\n" + "=" * 60)
    print(f"All results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
