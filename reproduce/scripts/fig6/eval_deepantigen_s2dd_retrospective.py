#!/usr/bin/env python3
"""
Retrospective S2DD validation on deepAntigen (Que et al., Nature Comms 2025).

Phase 2: Reproduce their published results on their own data.
Phase 3: Apply S2DD to explain, predict, and improve their results.

Datasets (from deepAntigen repo / Zenodo):
  - Training Dataset 3: 62,446 pairs (31,223 pos, 208 epitopes)
  - Dataset 8 (zero-shot): 1,714 pairs (491 unseen epitopes)
  - Dataset 9 (ImmuneCODE SARS-CoV-2): 1,129,028 pairs (518 epitopes)
  - Dataset 10 (NeoTCR): 384 tumor neoantigens (all positive)

Usage:
  python eval_deepantigen_s2dd_retrospective.py [--skip-inference]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

DATA_DIR = Path(INPUT_DIR) / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data'
MODEL_DIR = Path(INPUT_DIR) / 'Model' / 'deepAntigen'
# Read existing predictions/cached distances from INPUT_DIR; write outputs to OUTPUT_DIR
RESULTS_DIR_IN = Path(INPUT_DIR) / 'results' / 'deepantigen_retrospective'
RESULTS_DIR = Path(OUTPUT_DIR) / 'deepantigen_retrospective'

# S2DD parameters (2-chain: peptide + CDR3β)
CHAIN_COLS = ['peptide', 'CDR3b']
k, b, K = 0.1, 0.1, 50
N_BINS = 8


def safe_metric(name, y_true, y_prob):
    from calipper.general_evaluator import safe_metric as _sm
    return _sm(name, y_true, y_prob)


def run_deepantigen_inference(input_csv, output_csv):
    """Run deepAntigen inference and save results."""
    sys.path.insert(0, str(MODEL_DIR))
    from deepAntigen.antigenTCR import run_antigenTCR_seq

    result = run_antigenTCR_seq.Inference(str(input_csv), multi_process=8)
    result.to_csv(output_csv, index=False)
    print(f"  Saved {len(result)} predictions to {output_csv}")
    return result


def compute_s2dd_distances(test_df, train_df, chain_cols=CHAIN_COLS):
    """Compute S2DD distances from training to test data."""
    from calipper.combine_first_helpers import (
        compute_chain_weights, compute_combine_first_distances
    )
    # Rename columns to match our pipeline
    test_renamed = test_df.rename(columns={'binding_TCR': 'CDR3b'})
    train_renamed = train_df.rename(columns={'binding_TCR': 'CDR3b'})

    weights, _ = compute_chain_weights(
        train_renamed, chain_cols, k, b, K, formula='sigma_C'
    )
    print(f"  Chain weights (sigma_C): {dict(zip(chain_cols, weights))}")

    dists = compute_combine_first_distances(
        test_renamed, train_renamed, chain_cols, weights, k, b, K
    )
    return dists


def bin_and_eval(dists, y_true, y_prob, metrics=('aucroc', 'ap', 'f1'), n_bins=N_BINS):
    """Bin by distance and compute metrics per bin."""
    si = np.argsort(dists)
    bs = len(si) // n_bins
    results = []
    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        row = {'bin': i, 'mean_dist': dists[idx].mean(), 'n_samples': len(idx)}
        for m in metrics:
            row[m] = safe_metric(m, y_true[idx], y_prob[idx])
        results.append(row)
    return pd.DataFrame(results)


def compute_correlations(bin_df, metrics=('aucroc', 'ap', 'f1')):
    """Compute Pearson r and R² for each metric vs distance."""
    from scipy.stats import pearsonr
    results = {}
    for m in metrics:
        vals = bin_df[m].values
        dists = bin_df['mean_dist'].values
        mask = ~np.isnan(vals)
        if mask.sum() >= 3:
            r, p = pearsonr(dists[mask], vals[mask])
            results[m] = {'r': r, 'p': p, 'R2': r**2}
        else:
            results[m] = {'r': np.nan, 'p': np.nan, 'R2': np.nan}
    return results


# ── Phase 2: Reproduce published results ─────────────────────────────────

def phase2_reproduce(skip_inference=False):
    """Reproduce deepAntigen's published results."""
    print("\n" + "="*70)
    print("PHASE 2: Reproduce deepAntigen's published results")
    print("="*70)

    out_dir = RESULTS_DIR / 'reproduction'
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        'zero_shot': {
            'path': DATA_DIR / 'zero_shot_test.csv',
            'expected_auroc': 0.80,
            'description': 'Dataset 8 (491 unseen antigens)',
        },
        'immunecode': {
            'path': DATA_DIR / 'immunecode_sars.csv',
            'expected_auroc': 0.71,
            'description': 'Dataset 9 (ImmuneCODE SARS-CoV-2)',
        },
    }

    results = {}
    for name, info in datasets.items():
        print(f"\n--- {info['description']} ---")
        pred_path = out_dir / f'{name}_predictions.csv'

        if skip_inference and pred_path.exists():
            print(f"  Loading cached predictions from {pred_path}")
            pred_df = pd.read_csv(pred_path)
        else:
            print(f"  Running deepAntigen inference on {info['path']}...")
            pred_df = run_deepantigen_inference(info['path'], pred_path)

        y_true = pred_df['label'].values
        y_prob = pred_df['score'].values

        auroc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, (y_prob >= 0.5).astype(int))

        print(f"  AUROC: {auroc:.4f} (expected: {info['expected_auroc']:.2f})")
        print(f"  AP:    {ap:.4f}")
        print(f"  F1:    {f1:.4f}")

        results[name] = {'auroc': auroc, 'ap': ap, 'f1': f1,
                         'n_samples': len(pred_df),
                         'expected_auroc': info['expected_auroc']}

    # Save summary
    summary = pd.DataFrame(results).T
    summary.to_csv(out_dir / 'reproduction_summary.csv')
    print(f"\nSummary saved to {out_dir / 'reproduction_summary.csv'}")
    return results


# ── Phase 3: S2DD Retrospective Analysis ──────────────────────────────────

def phase3_s2dd_degradation(skip_inference=False):
    """Compute S2DD degradation curves on deepAntigen's own data."""
    print("\n" + "="*70)
    print("PHASE 3: S2DD Degradation on deepAntigen's own data")
    print("="*70)

    out_dir = RESULTS_DIR / 's2dd_degradation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data — use FULL training set (pos + neg) as S2DD reference,
    # matching the standard pipeline (all other models use full train data)
    print("\nLoading deepAntigen training data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Training: {len(train_df)} pairs ({(train_df['label']==1).sum()} pos), "
          f"{train_df['peptide'].nunique()} epitopes")

    datasets = {
        'zero_shot': DATA_DIR / 'zero_shot_test.csv',
        'immunecode': DATA_DIR / 'immunecode_sars.csv',
    }

    MAX_TEST_SAMPLES = 50000  # subsample large datasets to avoid OOM

    all_results = {}
    for ds_name, ds_path in datasets.items():
        print(f"\n--- {ds_name} ---")

        # Check for cached distances
        cached_dist_path = out_dir / f'{ds_name}_with_distances.csv'
        if cached_dist_path.exists():
            print(f"  Loading cached distances from {cached_dist_path}")
            cached = pd.read_csv(cached_dist_path)
            dists = cached['s2dd_distance'].values
            pred_path = RESULTS_DIR_IN / 'reproduction' / f'{ds_name}_predictions.csv'
            pred_df = pd.read_csv(pred_path)
            if len(cached) < len(pred_df):
                # Subsampled — use cached subset's predictions
                pred_df = cached
                y_true = cached['label'].values
                y_prob = cached['prediction'].values
            else:
                y_true = pred_df['label'].values
                y_prob = pred_df['score'].values

            bin_df = bin_and_eval(dists, y_true, y_prob)
            bin_df.to_csv(out_dir / f'{ds_name}_binned.csv', index=False)
            corr = compute_correlations(bin_df)
            print(f"  Degradation correlations (cached):")
            for m, vals in corr.items():
                print(f"    {m:>8s}: r={vals['r']:+.4f}, R²={vals['R2']:.4f}, p={vals['p']:.2e}")
            all_results[ds_name] = {'bin_df': bin_df, 'correlations': corr, 'dists': dists}
            continue

        test_df = pd.read_csv(ds_path)
        print(f"  Test: {len(test_df)} pairs, {test_df['peptide'].nunique()} epitopes")

        # Load predictions (from Phase 2)
        pred_path = RESULTS_DIR_IN / 'reproduction' / f'{ds_name}_predictions.csv'
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
        else:
            print(f"  No predictions found at {pred_path}. Run Phase 2 first.")
            continue

        # Subsample large datasets to avoid OOM in distance computation
        if len(test_df) > MAX_TEST_SAMPLES:
            n_orig = len(test_df)
            idx = test_df.sample(n=MAX_TEST_SAMPLES, random_state=42).index
            test_df = test_df.loc[idx].reset_index(drop=True)
            pred_df = pred_df.loc[idx].reset_index(drop=True)
            print(f"  Subsampled {n_orig} → {len(test_df)} (random, preserving label balance)")

        # Compute S2DD distances
        print("  Computing S2DD distances...")
        dists = compute_s2dd_distances(test_df, train_df)
        print(f"  Distance range: [{dists.min():.3f}, {dists.max():.3f}], "
              f"mean={dists.mean():.3f}")

        # Bin and evaluate
        y_true = pred_df['label'].values
        y_prob = pred_df['score'].values

        bin_df = bin_and_eval(dists, y_true, y_prob)
        bin_df.to_csv(out_dir / f'{ds_name}_binned.csv', index=False)

        # Compute correlations
        corr = compute_correlations(bin_df)
        print(f"\n  Degradation correlations:")
        for m, vals in corr.items():
            print(f"    {m:>8s}: r={vals['r']:+.4f}, R²={vals['R2']:.4f}, p={vals['p']:.2e}")

        all_results[ds_name] = {'bin_df': bin_df, 'correlations': corr, 'dists': dists}

        # Save per-sample distances
        test_df['s2dd_distance'] = dists
        test_df['prediction'] = y_prob
        test_df.to_csv(out_dir / f'{ds_name}_with_distances.csv', index=False)

    # Compare distance distributions
    if 'zero_shot' in all_results and 'immunecode' in all_results:
        zs_dists = all_results['zero_shot']['dists']
        ic_dists = all_results['immunecode']['dists']
        print(f"\n--- Distance comparison ---")
        print(f"  Zero-shot mean dist: {zs_dists.mean():.3f} ± {zs_dists.std():.3f}")
        print(f"  ImmuneCODE mean dist: {ic_dists.mean():.3f} ± {ic_dists.std():.3f}")
        print(f"  ImmuneCODE is {'MORE' if ic_dists.mean() > zs_dists.mean() else 'LESS'} "
              f"distant → explains AUROC drop")

    # Plot degradation curves
    _plot_degradation(all_results, out_dir)

    return all_results


def _plot_degradation(all_results, out_dir):
    """Plot S2DD degradation curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['aucroc', 'ap', 'f1']
    titles = ['AUROC', 'Average Precision', 'F1 Score']
    colors = {'zero_shot': '#e74c3c', 'immunecode': '#3498db'}
    labels = {'zero_shot': 'Dataset 8 (zero-shot)', 'immunecode': 'ImmuneCODE (SARS-CoV-2)'}

    for ax, metric, title in zip(axes, metrics, titles):
        for ds_name, data in all_results.items():
            bin_df = data['bin_df']
            corr = data['correlations'][metric]
            ax.plot(bin_df['mean_dist'], bin_df[metric],
                    'o-', color=colors.get(ds_name, 'gray'),
                    label=f"{labels.get(ds_name, ds_name)} (r={corr['r']:.3f})",
                    markersize=6, linewidth=2)
        ax.set_xlabel('S2DD distance', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} vs S2DD distance', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'degradation_curves.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'degradation_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved degradation curves to {out_dir}")


# ── Phase 4: Performance Prediction ───────────────────────────────────────

def phase4_performance_prediction():
    """Predict deepAntigen's performance on held-out subsets using S2DD (2-chain)."""
    print("\n" + "="*70)
    print("PHASE 4: Performance Prediction (2-chain: peptide + CDR3β)")
    print("="*70)

    from calipper.combine_first_helpers import fit_ridge_vbias

    out_dir = RESULTS_DIR / 'performance_prediction'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load zero-shot with cached distances
    zs_path = RESULTS_DIR / 's2dd_degradation' / 'zero_shot_with_distances.csv'
    if not zs_path.exists():
        print("  Run Phase 3 first to compute distances.")
        return
    zs = pd.read_csv(zs_path)
    print(f"  Loaded zero-shot: {len(zs)} samples with distances")

    y_true = zs['label'].values
    y_prob = zs['prediction'].values
    dists = zs['s2dd_distance'].values

    # Half-split: sort by distance, alternate into cal/val
    si = np.argsort(dists)
    cal_idx = si[0::2]
    val_idx = si[1::2]
    print(f"  Cal: {len(cal_idx)}, Val: {len(val_idx)}")

    metrics = ['aucroc', 'ap', 'f1']
    results = []

    for metric in metrics:
        # Bin cal half into 8 bins
        cal_si = np.argsort(dists[cal_idx])
        bs = len(cal_si) // N_BINS
        cal_d, cal_mp, cal_y = [], [], []
        for i in range(N_BINS):
            s = i * bs
            e = len(cal_si) if i == N_BINS - 1 else (i + 1) * bs
            idx = cal_idx[cal_si[s:e]]
            cal_d.append(dists[idx].mean())
            cal_mp.append(y_prob[idx].mean())
            cal_y.append(safe_metric(metric, y_true[idx], y_prob[idx]))

        cal_d = np.array(cal_d)
        cal_mp = np.array(cal_mp)
        cal_y = np.array(cal_y)

        # Fit ridge-bin-vbias on cal
        params = fit_ridge_vbias(cal_d, cal_mp, cal_y, lam=0.0)

        # Predict on val bins
        val_si = np.argsort(dists[val_idx])
        pred_vals, true_vals = [], []
        for i in range(N_BINS):
            s = i * bs
            e = len(val_si) if i == N_BINS - 1 else (i + 1) * bs
            idx = val_idx[val_si[s:e]]
            d_val = dists[idx].mean()
            mp_val = y_prob[idx].mean()
            # predict: a * exp(-b * d) + c + beta * mp
            a, b_param, c, beta = params
            pred_v = a * np.exp(-b_param * d_val) + c + beta * mp_val
            true_v = safe_metric(metric, y_true[idx], y_prob[idx])
            pred_vals.append(pred_v)
            true_vals.append(true_v)

        pred_vals = np.array(pred_vals)
        true_vals = np.array(true_vals)

        from scipy.stats import pearsonr
        r, p = pearsonr(pred_vals, true_vals)
        mae = np.mean(np.abs(pred_vals - true_vals))
        results.append({'metric': metric, 'R': r, 'p': p, 'MAE': mae,
                        'n_bins': N_BINS})
        print(f"  {metric:>8s}: R={r:.4f}, MAE={mae:.4f}, p={p:.2e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / 'prediction_results_2chain.csv', index=False)
    print(f"\n  Saved to {out_dir / 'prediction_results_2chain.csv'}")

    # Also try on ImmuneCODE (cross-dataset prediction)
    ic_path = RESULTS_DIR / 's2dd_degradation' / 'immunecode_with_distances.csv'
    if ic_path.exists():
        print("\n  Cross-dataset prediction: fit on zero-shot, predict on ImmuneCODE...")
        ic = pd.read_csv(ic_path)
        ic_dists = ic['s2dd_distance'].values
        ic_y = ic['label'].values
        ic_p = ic['prediction'].values

        for metric in metrics:
            # Use zero-shot cal params → predict ImmuneCODE bins
            cal_d_full = np.array(cal_d)  # from zero-shot cal above
            # Refit on full zero-shot
            full_si = np.argsort(dists)
            full_d, full_mp, full_y = [], [], []
            for i in range(N_BINS):
                s = i * (len(full_si) // N_BINS)
                e = len(full_si) if i == N_BINS - 1 else (i + 1) * (len(full_si) // N_BINS)
                idx = full_si[s:e]
                full_d.append(dists[idx].mean())
                full_mp.append(y_prob[idx].mean())
                full_y.append(safe_metric(metric, y_true[idx], y_prob[idx]))
            params_full = fit_ridge_vbias(np.array(full_d), np.array(full_mp),
                                          np.array(full_y), lam=0.0)

            # Predict ImmuneCODE bins
            ic_si = np.argsort(ic_dists)
            ic_bs = len(ic_si) // N_BINS
            ic_pred, ic_true = [], []
            for i in range(N_BINS):
                s = i * ic_bs
                e = len(ic_si) if i == N_BINS - 1 else (i + 1) * ic_bs
                idx = ic_si[s:e]
                d_v = ic_dists[idx].mean()
                mp_v = ic_p[idx].mean()
                a, b_param, c, beta = params_full
                ic_pred.append(a * np.exp(-b_param * d_v) + c + beta * mp_v)
                ic_true.append(safe_metric(metric, ic_y[idx], ic_p[idx]))

            r, p = pearsonr(ic_pred, ic_true)
            mae = np.mean(np.abs(np.array(ic_pred) - np.array(ic_true)))
            print(f"    {metric:>8s}: R={r:.4f}, MAE={mae:.4f} (cross-dataset)")


# ── Phase 5: Neoantigen Confidence ────────────────────────────────────────

def phase5_neoantigen_confidence():
    """Analyze S2DD distance vs ELISPOT results for clinical neoantigens (2-chain)."""
    print("\n" + "="*70)
    print("PHASE 5: Neoantigen Confidence Analysis (2-chain: peptide + CDR3β)")
    print("="*70)

    out_dir = RESULTS_DIR / 'neoantigen_confidence'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for clinical data
    clinical_zip = MODEL_DIR / 'clinical_cancer_patients.zip'
    clinical_dir = MODEL_DIR / 'clinical_cancer_patients'
    if not clinical_dir.exists() and clinical_zip.exists():
        import zipfile
        print(f"  Extracting {clinical_zip}...")
        with zipfile.ZipFile(clinical_zip) as zf:
            zf.extractall(MODEL_DIR)

    if not clinical_dir.exists():
        print("  Clinical cancer patient data not found. Skipping.")
        return

    # List available patient data
    print(f"  Clinical data directory: {clinical_dir}")
    import glob
    patient_files = glob.glob(str(clinical_dir / '**' / '*.csv'), recursive=True)
    print(f"  Found {len(patient_files)} CSV files:")
    for f in patient_files[:10]:
        print(f"    {os.path.relpath(f, str(clinical_dir))}")

    # Load training data for S2DD reference
    from calipper.combine_first_helpers import compute_chain_weights, compute_combine_first_distances
    train = pd.read_csv(DATA_DIR / 'train.csv').rename(columns={'binding_TCR': 'CDR3b'})
    w, _ = compute_chain_weights(train, CHAIN_COLS, k, b, K, formula='sigma_C')
    print(f"  Chain weights: {dict(zip(CHAIN_COLS, w))}")

    # Try to find neoantigen prediction files
    # The clinical data likely has patient-specific neoantigen CSVs
    neo_results = []
    for pf in patient_files:
        try:
            df = pd.read_csv(pf)
            cols = df.columns.tolist()
            # Look for files with peptide + score columns
            if 'peptide' in cols or 'Peptide' in cols or 'neoantigen' in [c.lower() for c in cols]:
                print(f"\n  Processing: {os.path.basename(pf)}")
                print(f"    Columns: {cols}")
                print(f"    Shape: {df.shape}")
                print(f"    First row: {df.iloc[0].to_dict()}")
        except Exception as e:
            import sys as _s_da
            print(f"  ⚠ FALLBACK [deepantigen-neoantigen]: failed reading {pf} ({type(e).__name__}: {e}); skipping file", file=_s_da.stderr, flush=True)

    print(f"\n  Neoantigen confidence analysis requires identifying prediction + ELISPOT columns.")
    print(f"  TODO: Parse patient-specific files once format is understood.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='deepAntigen S2DD retrospective (2-chain)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip model inference, use cached predictions')
    parser.add_argument('--phase', type=int, nargs='+', default=[2, 3, 4, 5],
                        help='Phases to run (2=reproduce, 3=S2DD, 4=prediction, 5=neoantigen)')
    args = parser.parse_args()

    if 2 in args.phase:
        os.chdir(str(MODEL_DIR))
        phase2_reproduce(skip_inference=args.skip_inference)
        pass  # os.chdir removed: PROJECT_ROOT not defined under _paths

    if 3 in args.phase:
        phase3_s2dd_degradation(skip_inference=args.skip_inference)

    if 4 in args.phase:
        phase4_performance_prediction()

    if 5 in args.phase:
        phase5_neoantigen_confidence()

    print("\n" + "="*70)
    print("Done.")


if __name__ == '__main__':
    main()
