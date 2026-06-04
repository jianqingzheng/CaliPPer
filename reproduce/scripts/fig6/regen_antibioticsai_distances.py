#!/usr/bin/env python3
"""
AntibioticsAI Retrospective S2DD Validation.

Extends S2DD beyond immune receptor binding to small-molecule drug discovery.
Uses Wong et al. Nature 2024 antibiotic activity prediction data.

Model: Ensemble of 20 Chemprop GNNs (message-passing neural network)
Data:  Zenodo DOI: 10.5281/zenodo.10095879 + Nature Supp Data 2
Task:  Binary classification — antibiotic active (growth inhibition) vs inactive

Three test scenarios:
  1. 283 experimentally tested compounds (main validation)
  2. Beta-lactam-withheld LOO (505 compounds, model trained without beta-lactams)
  3. Quinolone-withheld LOO (31 compounds, model trained without quinolones)

Key settings:
  - Distance: Morgan fingerprint (radius=2, 2048 bits) Tanimoto similarity
  - This is the molecular analog of Levenshtein for sequences
  - Single "chain" (molecule SMILES → fingerprint)
  - k=0.1, b=0.1, K=50, bin_num=8
  - Label column: ACTIVITY (1=active, 0=inactive)
  - Prediction column: ANTIBIOTIC_PS or PREDICTION_SCORE
  - Training overlap: 4/283 (negligible)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import os, sys
# CaliPPer self-contained path bootstrap (writes into INPUT_DIR so Stage 1 reads fresh distances)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR
from pathlib import Path
PROJECT_ROOT = Path(INPUT_DIR)
ROOT = PROJECT_ROOT


from calipper.general_evaluator import safe_metric
from calipper.core import predict_metric, fit_recalibration, apply_recalibration

# === Configuration ===
REPO_DIR = ROOT / "Model" / "AntibioticsAI"
SUPP_DIR = ROOT / "Data" / "retrospective_antibioticsai" / "supplementary"
RESULTS_DIR = ROOT / "results" / "antibioticsai_retrospective"

K_NEIGHBORS = 50
N_BINS = 8  # for degradation analysis and performance prediction (NOT recalibration)
# Recalibration uses v2.7 adaptive defaults (adaptive_n_bins, adaptive theta)


# ============================================================
# Morgan fingerprint Tanimoto distance (molecular S2DD)
# ============================================================

def compute_morgan_tanimoto_distances(test_smiles, train_smiles, K=50, cache_path=None):
    """Compute S2DD-style distances using Morgan fingerprint Tanimoto similarity.

    For each test molecule, compute Tanimoto similarity to all training molecules,
    then apply LogDist transform: d = log(k * (1 - Tanimoto + b)).
    Final distance = mean of top-K smallest (most similar) distances.

    This is the molecular analog of Levenshtein-based LogDist for sequences.
    """
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path)
        if len(data['distances']) == len(test_smiles):
            print(f"  Loaded cached distances")
            return data['distances']

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        raise ImportError("RDKit required: pip install rdkit")

    print(f"  Computing Morgan fingerprints...")
    k_param, b_param = 0.1, 0.1

    def smiles_to_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    train_fps = [smiles_to_fp(s) for s in train_smiles]
    train_fps_valid = [(i, fp) for i, fp in enumerate(train_fps) if fp is not None]
    print(f"  Training: {len(train_fps_valid)}/{len(train_smiles)} valid fingerprints")

    test_fps = [smiles_to_fp(s) for s in test_smiles]
    n_test = len(test_fps)

    print(f"  Computing pairwise Tanimoto: {n_test} × {len(train_fps_valid)}...")
    distances = np.zeros(n_test)
    for i, tfp in enumerate(test_fps):
        if tfp is None:
            distances[i] = 0.0  # neutral distance for unparseable molecules
            continue
        # Compute Tanimoto to all training molecules
        sims = DataStructs.BulkTanimotoSimilarity(tfp, [fp for _, fp in train_fps_valid])
        # LogDist transform
        d_all = np.log(k_param * (1 - np.array(sims) + b_param))
        # Top-K mean (most similar = most negative distances)
        topk = np.sort(d_all)[:K]
        distances[i] = topk.mean()

        if (i + 1) % 100 == 0 or i == n_test - 1:
            print(f"    [{i+1}/{n_test}] {100*(i+1)/n_test:.1f}%")

    if cache_path:
        np.savez(cache_path, distances=distances)
        print(f"  Cached to {cache_path}")

    return distances


# ============================================================
# Data loading
# ============================================================

def load_data():
    """Load training data and all test sets."""
    # Training data
    train = pd.read_csv(REPO_DIR / "working_example" / "train.csv")
    # Fix BOM
    if train.columns[0].startswith('\ufeff'):
        train.columns = [c.lstrip('\ufeff') for c in train.columns]

    # Main test set (283 compounds)
    supp_file = SUPP_DIR / "41586_2023_6887_MOESM4_ESM.xlsx"
    tested = pd.read_excel(supp_file, sheet_name='All tested compounds')

    # Beta-lactam withheld LOO
    blac_raw = pd.read_excel(supp_file, sheet_name='B-lactam-withheld train+test', header=0)
    blac_test = blac_raw.iloc[1:, 3:6].copy()
    blac_test.columns = ['SMILES', 'ACTIVITY', 'PREDICTION_SCORE']
    blac_test = blac_test.dropna(subset=['SMILES'])
    blac_test['ACTIVITY'] = pd.to_numeric(blac_test['ACTIVITY'], errors='coerce')
    blac_test['PREDICTION_SCORE'] = pd.to_numeric(blac_test['PREDICTION_SCORE'], errors='coerce')
    blac_test = blac_test.dropna(subset=['ACTIVITY', 'PREDICTION_SCORE'])

    # Beta-lactam withheld training set
    blac_train = blac_raw.iloc[1:, 0:2].copy()
    blac_train.columns = ['SMILES', 'ACTIVITY']
    blac_train = blac_train.dropna(subset=['SMILES'])
    blac_train['ACTIVITY'] = pd.to_numeric(blac_train['ACTIVITY'], errors='coerce')
    blac_train = blac_train.dropna()

    # Quinolone withheld LOO
    quin_raw = pd.read_excel(supp_file, sheet_name='Quinolone-withheld train+test', header=0)
    quin_test = quin_raw.iloc[1:, 3:6].copy()
    quin_test.columns = ['SMILES', 'ACTIVITY', 'PREDICTION_SCORE']
    quin_test = quin_test.dropna(subset=['SMILES'])
    quin_test['ACTIVITY'] = pd.to_numeric(quin_test['ACTIVITY'], errors='coerce')
    quin_test['PREDICTION_SCORE'] = pd.to_numeric(quin_test['PREDICTION_SCORE'], errors='coerce')
    quin_test = quin_test.dropna(subset=['ACTIVITY', 'PREDICTION_SCORE'])

    print(f"Training: {len(train)} molecules ({train['ACTIVITY'].sum()} active)")
    print(f"Main test: {len(tested)} ({tested['ACTIVITY'].sum()} active)")
    print(f"Beta-lactam LOO test: {len(blac_test)} ({int(blac_test['ACTIVITY'].sum())} active)")
    print(f"Quinolone LOO test: {len(quin_test)} ({int(quin_test['ACTIVITY'].sum())} active)")

    return {
        'train': train,
        'tested': tested,
        'blac_test': blac_test,
        'blac_train': blac_train,
        'quin_test': quin_test,
    }


# ============================================================
# S2DD degradation analysis
# ============================================================

def run_degradation(y, p, distances, n_bins=8):
    """Compute per-bin performance metrics."""
    sort_idx = np.argsort(distances)
    bs = len(distances) // n_bins
    bins = []
    for i in range(n_bins):
        s = i * bs
        e = (i + 1) * bs if i < n_bins - 1 else len(distances)
        idx = sort_idx[s:e]
        yi, pi, di = y[idx], p[idx], distances[idx]
        row = {'bin': i, 'mean_dist': di.mean(), 'n_samples': len(idx),
               'prevalence': yi.mean()}
        for m in ['aucroc', 'ap']:
            try:
                row[m] = safe_metric(m, yi, pi)
            except:
                row[m] = np.nan
        try:
            row['f1'] = f1_score(yi, (pi > 0.5).astype(int))
        except:
            row['f1'] = np.nan
        bins.append(row)
    return pd.DataFrame(bins)


# ============================================================
# Main
# ============================================================

def main():
    print("AntibioticsAI Retrospective S2DD Validation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ['reproduction', 's2dd_degradation', 'performance_prediction', 'recalibration']:
        (RESULTS_DIR / subdir).mkdir(exist_ok=True)

    data = load_data()
    train = data['train']
    tested = data['tested']

    # === Scenario 1: Main 283-compound test ===
    print("\n" + "=" * 60)
    print("SCENARIO 1: 283 experimentally tested compounds")
    print("=" * 60)

    y1 = tested['ACTIVITY'].values.astype(int)
    p1 = tested['ANTIBIOTIC_PS'].values.astype(float)
    auroc = roc_auc_score(y1, p1)
    ap = average_precision_score(y1, p1)
    f1 = f1_score(y1, (p1 > 0.5).astype(int))
    print(f"  Overall: AUROC={auroc:.3f}, AP={ap:.3f}, F1={f1:.3f}")

    # Compute distances
    d1 = compute_morgan_tanimoto_distances(
        tested['SMILES'].tolist(), train['SMILES'].tolist(),
        K=K_NEIGHBORS, cache_path=RESULTS_DIR / "distance_cache_main.npz"
    )
    tested_out = tested.copy()
    tested_out['distance'] = d1

    # Degradation
    bin_df = run_degradation(y1, p1, d1, N_BINS)
    bin_df.to_csv(RESULTS_DIR / "s2dd_degradation" / "main_degradation.csv", index=False)
    for m in ['aucroc', 'ap', 'f1']:
        vals = bin_df[m].dropna()
        dists = bin_df.loc[vals.index, 'mean_dist']
        if len(vals) >= 4:
            r, pval = pearsonr(dists, vals)
            rs, ps = spearmanr(dists, vals)
            print(f"  Degradation {m}: Pearson r={r:.3f} (p={pval:.3f}), Spearman ρ={rs:.3f} (p={ps:.3f})")

    # Save reproduction
    pd.DataFrame([{'auroc': auroc, 'ap': ap, 'f1': f1, 'n': len(y1)}]).to_csv(
        RESULTS_DIR / "reproduction" / "main_test_performance.csv", index=False)
    tested_out.to_csv(RESULTS_DIR / "reproduction" / "main_test_with_distances.csv", index=False)

    # === Scenario 2: Beta-lactam withheld (505 compounds) ===
    print("\n" + "=" * 60)
    print("SCENARIO 2: Beta-lactam withheld LOO (505 compounds)")
    print("=" * 60)

    blac = data['blac_test']
    blac_train = data['blac_train']
    y2 = blac['ACTIVITY'].values.astype(int)
    p2 = blac['PREDICTION_SCORE'].values.astype(float)
    auroc2 = roc_auc_score(y2, p2)
    ap2 = average_precision_score(y2, p2)
    f1_2 = f1_score(y2, (p2 > 0.5).astype(int))
    print(f"  Overall: AUROC={auroc2:.3f}, AP={ap2:.3f}, F1={f1_2:.3f}")

    # Distances from beta-lactam-withheld training to beta-lactam test
    d2 = compute_morgan_tanimoto_distances(
        blac['SMILES'].tolist(), blac_train['SMILES'].tolist(),
        K=K_NEIGHBORS, cache_path=RESULTS_DIR / "distance_cache_blactam.npz"
    )

    bin_df2 = run_degradation(y2, p2, d2, N_BINS)
    bin_df2.to_csv(RESULTS_DIR / "s2dd_degradation" / "blactam_degradation.csv", index=False)
    for m in ['aucroc', 'ap', 'f1']:
        vals = bin_df2[m].dropna()
        dists = bin_df2.loc[vals.index, 'mean_dist']
        if len(vals) >= 4:
            r, pval = pearsonr(dists, vals)
            rs, ps = spearmanr(dists, vals)
            print(f"  Degradation {m}: Pearson r={r:.3f} (p={pval:.3f}), Spearman ρ={rs:.3f} (p={ps:.3f})")

    pd.DataFrame([{'auroc': auroc2, 'ap': ap2, 'f1': f1_2, 'n': len(y2)}]).to_csv(
        RESULTS_DIR / "reproduction" / "blactam_test_performance.csv", index=False)

    # === Performance Prediction (using main test as cal, predict beta-lactam) ===
    print("\n" + "=" * 60)
    print("PERFORMANCE PREDICTION")
    print("=" * 60)

    # Dataset-level: use main test as calibration → predict beta-lactam performance
    # Both use the full training set as reference
    cal_data_main = {'main_test': (y1, p1, d1)}

    # Distances for beta-lactam from FULL training set (for cross-prediction)
    d2_full = compute_morgan_tanimoto_distances(
        blac['SMILES'].tolist(), train['SMILES'].tolist(),
        K=K_NEIGHBORS, cache_path=RESULTS_DIR / "distance_cache_blactam_full.npz"
    )

    pred_results = []
    for metric in ['aucroc', 'ap', 'f1']:
        try:
            actual = safe_metric(metric, y2, p2) if metric != 'f1' \
                else f1_score(y2, (p2 > 0.5).astype(int))
            result = predict_metric(cal_data_main, p2, d2_full, metrics=[metric], n_bins=N_BINS)
            predicted = result['estimated'].get(metric, np.nan)
            pred_results.append({
                'metric': metric, 'actual': actual, 'predicted': predicted,
                'abs_error': abs(actual - predicted) if not np.isnan(predicted) else np.nan
            })
            print(f"  Predict beta-lactam {metric}: predicted={predicted:.3f}, actual={actual:.3f}, error={abs(actual-predicted):.3f}")
        except Exception as e:
            print(f"  Predict beta-lactam {metric}: failed — {e}")

    # Within-main-test half-split prediction
    np.random.seed(42)
    n_half = len(y1) // 2
    perm = np.random.permutation(len(y1))
    cal_idx, test_idx = perm[:n_half], perm[n_half:]
    cal_data_half = {'cal_half': (y1[cal_idx], p1[cal_idx], d1[cal_idx])}
    for metric in ['aucroc', 'ap', 'f1']:
        try:
            actual = safe_metric(metric, y1[test_idx], p1[test_idx]) if metric != 'f1' \
                else f1_score(y1[test_idx], (p1[test_idx] > 0.5).astype(int))
            result = predict_metric(cal_data_half, p1[test_idx], d1[test_idx],
                                    metrics=[metric], n_bins=N_BINS)
            predicted = result['estimated'].get(metric, np.nan)
            pred_results.append({
                'metric': metric, 'actual': actual, 'predicted': predicted,
                'abs_error': abs(actual - predicted) if not np.isnan(predicted) else np.nan,
                'source': 'within_main_halfsplit'
            })
            print(f"  Within-main {metric}: predicted={predicted:.3f}, actual={actual:.3f}, error={abs(actual-predicted):.3f}")
        except Exception as e:
            print(f"  Within-main {metric}: failed — {e}")

    pd.DataFrame(pred_results).to_csv(
        RESULTS_DIR / "performance_prediction" / "prediction_results.csv", index=False)

    # === Bayesian Recalibration ===
    print("\n" + "=" * 60)
    print("BAYESIAN RECALIBRATION")
    print("=" * 60)

    # Recalibrate main test using within-test half-split
    cal_data_recal = {'cal_half': (y1[cal_idx], p1[cal_idx], d1[cal_idx])}
    try:
        ppv_params, npv_params, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data_recal)
        cal_scores = apply_recalibration(
            y1[test_idx], p1[test_idx], d1[test_idx],
            ppv_params, npv_params, p_pos, p_neg)

        recal_results = []
        for metric in ['aucroc', 'ap', 'f1']:
            if metric == 'f1':
                before = f1_score(y1[test_idx], (p1[test_idx] > 0.5).astype(int))
                after = f1_score(y1[test_idx], (cal_scores > 0.5).astype(int))
            else:
                before = safe_metric(metric, y1[test_idx], p1[test_idx])
                after = safe_metric(metric, y1[test_idx], cal_scores)
            recal_results.append({
                'test_set': 'main_halfsplit', 'metric': metric,
                'before': before, 'after': after, 'delta': after - before
            })
            print(f"  Main half-split {metric}: {before:.3f} → {after:.3f} (Δ={after-before:+.3f})")

        pd.DataFrame(recal_results).to_csv(
            RESULTS_DIR / "recalibration" / "main_recalibration.csv", index=False)
    except Exception as e:
        print(f"  Main recalibration failed: {e}")

    # Recalibrate beta-lactam using main test as calibration
    try:
        ppv_params, npv_params, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data_main)
        cal_scores_blac = apply_recalibration(
            y2, p2, d2_full, ppv_params, npv_params, p_pos, p_neg)

        recal_blac = []
        for metric in ['aucroc', 'ap', 'f1']:
            if metric == 'f1':
                before = f1_score(y2, (p2 > 0.5).astype(int))
                after = f1_score(y2, (cal_scores_blac > 0.5).astype(int))
            else:
                before = safe_metric(metric, y2, p2)
                after = safe_metric(metric, y2, cal_scores_blac)
            recal_blac.append({
                'test_set': 'blactam_loo', 'metric': metric,
                'before': before, 'after': after, 'delta': after - before
            })
            print(f"  Beta-lactam {metric}: {before:.3f} → {after:.3f} (Δ={after-before:+.3f})")

        pd.DataFrame(recal_blac).to_csv(
            RESULTS_DIR / "recalibration" / "blactam_recalibration.csv", index=False)
    except Exception as e:
        print(f"  Beta-lactam recalibration failed: {e}")

    print("\n" + "=" * 60)
    print(f"All results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
