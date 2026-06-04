#!/usr/bin/env python3
"""Neoantigen confidence analysis: S2DD distance vs ELISPOT validation.

Computes S2DD distance for each of 100 candidate neoantigens (5 patients)
from deepAntigen's training data. Analyzes whether S2DD distance separates
confirmed immunogenic (ELISPOT P<0.05) from non-immunogenic neoantigens.

Data sources:
  - Neoantigen peptides + deepAntigen scores: Source Data Excel (Fig 5b/c/d, Supp Fig 22)
  - ELISPOT validation: paper Figure 5b/c/d and Supp Fig 22 (P<0.05 marked in figure)
  - Training data: deepAntigen training set (62,446 pairs, 208 epitopes)

Confirmed immunogenic (from paper Figure 5, P<0.05 by DFR test):
  P1017343 (lung):       A1, A3, A5, A7, A17  (5/20)
  P980589 (breast):      A10, A12, A18         (3/20)
  P9280216 (pancreatic):  A3, A7               (2/20)
  P1057556 (lung):       A1, A4               (2/20)  -- from Supp Fig 22
  P1060513 (breast):     A1, A5, A17          (3/20)  -- from Supp Fig 22
  Total: 15/100 confirmed immunogenic
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

warnings.filterwarnings('ignore')

# CaliPPer self-contained bootstrap
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR  # also adds CaliPPer/ to sys.path
from pathlib import Path
PROJECT_ROOT = Path(INPUT_DIR)


from calipper.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances
)

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / 'results' / 'deepantigen_retrospective' / 'neoantigen_confidence'
XLSX_PATH = PROJECT_ROOT / 'Model' / 'deepAntigen' / 'clinical_cancer_patients' / \
    'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data' / 'source_data.xlsx'
TRAIN_PATH = PROJECT_ROOT / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data' / 'train.csv'

# ── ELISPOT confirmed immunogenic (from paper Fig 5b/c/d + Supp Fig 22) ─
# Verified from ELISPOT bar plots: purple bars = P≤0.05 by DFR test
CONFIRMED = {
    'P1017343': {'A1', 'A3', 'A5', 'A7', 'A17'},   # Fig 5b
    'P980589':  {'A10', 'A12', 'A18'},               # Fig 5c
    'P9280216': {'A3', 'A7'},                         # Fig 5d
    'P1057556': {'A19', 'A20'},                       # Supp Fig 22a
    'P1060513': {'A13', 'A15', 'A19'},                # Supp Fig 22b
}

k, b, K = 0.1, 0.1, 50


def load_neoantigens():
    """Load all 100 neoantigens from Source Data Excel."""
    patients_sheets = {
        'P1017343': ('Figure5b', 'Lung cancer'),
        'P980589': ('Figure5c', 'Breast cancer'),
        'P9280216': ('Figure5d', 'Pancreatic cancer'),
    }
    all_neo = []
    for pid, (sheet, cancer) in patients_sheets.items():
        df = pd.read_excel(str(XLSX_PATH), sheet_name=sheet, header=1)
        df.columns = ['ID', 'peptide', 'Gene', 'variant_type', 'variant_info', 'score']
        df = df.dropna(subset=['peptide']).head(20)
        df['patient'] = pid
        df['cancer'] = cancer
        all_neo.append(df)

    # Supp Fig 22: two patients side by side
    df22 = pd.read_excel(str(XLSX_PATH), sheet_name='Supplementary Fig.22', header=1)
    left = df22.iloc[:, :6].copy()
    left.columns = ['ID', 'peptide', 'Gene', 'variant_type', 'variant_info', 'score']
    left = left.dropna(subset=['peptide']).head(20)
    left['patient'] = 'P1057556'
    left['cancer'] = 'Lung cancer'
    all_neo.append(left)

    right = df22.iloc[:, 8:14].copy()
    right.columns = ['ID', 'peptide', 'Gene', 'variant_type', 'variant_info', 'score']
    right = right.dropna(subset=['peptide']).head(20)
    right['patient'] = 'P1060513'
    right['cancer'] = 'Breast cancer'
    all_neo.append(right)

    neo = pd.concat(all_neo, ignore_index=True)
    # Mark confirmed immunogenic
    neo['confirmed'] = neo.apply(
        lambda r: r['ID'] in CONFIRMED.get(r['patient'], set()), axis=1)
    return neo


CHAIN_COLS = ['peptide', 'CDR3b']


def compute_neoantigen_distances(neo_df, train_df):
    """Compute S2DD distance for neoantigen peptides using the unified pipeline.

    Uses the same method as the zero-shot/ImmuneCODE S2DD computation:
    - 2-chain (peptide + CDR3b) with sigma_C weights
    - compute_combine_first_distances with weighted_max_znorm
    - k=0.1, b=0.1, K=50

    Since neoantigens don't have paired TCRs in the prediction output,
    we create synthetic rows with a dummy CDR3b. Because sigma_C gives
    99.97% weight to peptide and 0.03% to CDR3b, the CDR3b chain has
    negligible effect on the final distance.
    """
    # Rename columns for pipeline compatibility
    train_renamed = train_df.rename(columns={'binding_TCR': 'CDR3b'})

    # Create query DataFrame with dummy CDR3b (sigma_C gives it ~0% weight)
    query_df = pd.DataFrame({
        'peptide': neo_df['peptide'].values,
        'CDR3b': 'CASSXXXXXNEQFF',  # dummy CDR3b — negligible weight
    })

    # Compute sigma_C weights (same as used for zero-shot/ImmuneCODE)
    weights, _ = compute_chain_weights(
        train_renamed, CHAIN_COLS, k, b, K, formula='sigma_C')
    print(f"  sigma_C weights: {dict(zip(CHAIN_COLS, weights))}")

    # Compute S2DD using the unified pipeline (weighted_max_znorm)
    dists = compute_combine_first_distances(
        query_df, train_renamed, CHAIN_COLS, weights, k, b, K)

    return dists


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading neoantigens...")
    neo = load_neoantigens()
    print(f"  {len(neo)} neoantigens from {neo['patient'].nunique()} patients")
    print(f"  Confirmed immunogenic: {neo['confirmed'].sum()}/100")

    print("\nLoading training data...")
    train = pd.read_csv(TRAIN_PATH)
    # Rename for consistency
    if 'binding_TCR' in train.columns:
        train = train.rename(columns={'binding_TCR': 'CDR3b'})
    print(f"  {len(train)} training pairs, {train['peptide'].nunique()} unique peptides")

    print("\nComputing S2DD distances (peptide chain only)...")
    neo['s2dd_distance'] = compute_neoantigen_distances(neo, train)

    # ── Analysis ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Neoantigen S2DD Confidence Analysis")
    print("="*60)

    confirmed = neo[neo['confirmed']]
    not_confirmed = neo[~neo['confirmed']]

    print(f"\nConfirmed immunogenic (n={len(confirmed)}):")
    print(f"  S2DD distance: {confirmed['s2dd_distance'].mean():.4f} ± {confirmed['s2dd_distance'].std():.4f}")
    print(f"  deepAntigen score: {confirmed['score'].mean():.4f} ± {confirmed['score'].std():.4f}")

    print(f"\nNot confirmed (n={len(not_confirmed)}):")
    print(f"  S2DD distance: {not_confirmed['s2dd_distance'].mean():.4f} ± {not_confirmed['s2dd_distance'].std():.4f}")
    print(f"  deepAntigen score: {not_confirmed['score'].mean():.4f} ± {not_confirmed['score'].std():.4f}")

    # Mann-Whitney U test
    u_dist, p_dist = mannwhitneyu(confirmed['s2dd_distance'].astype(float),
                                   not_confirmed['s2dd_distance'].astype(float),
                                   alternative='two-sided')
    u_score, p_score = mannwhitneyu(confirmed['score'].astype(float),
                                     not_confirmed['score'].astype(float),
                                     alternative='two-sided')

    print(f"\nMann-Whitney U test (S2DD distance): U={u_dist:.0f}, P={p_dist:.4f}")
    print(f"Mann-Whitney U test (deepAntigen score): U={u_score:.0f}, P={p_score:.4f}")

    # Correlation: S2DD distance vs deepAntigen score
    r_ds, p_ds = spearmanr(neo['s2dd_distance'].astype(float), neo['score'].astype(float))
    print(f"\nSpearman: S2DD distance vs score: rho={r_ds:.3f}, P={p_ds:.4f}")

    # Per-patient analysis
    print("\n--- Per-patient breakdown ---")
    for pid in neo['patient'].unique():
        sub = neo[neo['patient'] == pid]
        conf = sub[sub['confirmed']]
        nconf = sub[~sub['confirmed']]
        print(f"\n  {pid} ({sub['cancer'].iloc[0]}):")
        print(f"    Confirmed ({len(conf)}): dist={conf['s2dd_distance'].mean():.4f}, "
              f"score={conf['score'].mean():.4f}")
        print(f"    Not confirmed ({len(nconf)}): dist={nconf['s2dd_distance'].mean():.4f}, "
              f"score={nconf['score'].mean():.4f}")
        if len(conf) >= 2 and len(nconf) >= 2:
            u, p = mannwhitneyu(conf['s2dd_distance'], nconf['s2dd_distance'], alternative='two-sided')
            print(f"    MW-U (distance): P={p:.4f}")

    # ── S2DD-based filtering analysis ───────────────────────────────────
    print("\n--- S2DD-based filtering ---")
    # For each distance threshold, compute how many confirmed/not-confirmed are retained
    thresholds = np.percentile(neo['s2dd_distance'], [25, 50, 75])
    for thr in thresholds:
        close = neo[neo['s2dd_distance'] <= thr]
        n_conf = close['confirmed'].sum()
        n_total = len(close)
        tdr = n_conf / n_total if n_total > 0 else 0
        print(f"  Threshold ≤ {thr:.3f}: retain {n_total} neoantigens, "
              f"{n_conf} confirmed, TDR={tdr:.1%} (baseline 15%)")

    # ── Save results ────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / 'neoantigen_s2dd_confidence.csv'
    neo.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Summary CSV
    summary = pd.DataFrame({
        'group': ['confirmed', 'not_confirmed'],
        'n': [len(confirmed), len(not_confirmed)],
        'mean_s2dd': [confirmed['s2dd_distance'].mean(), not_confirmed['s2dd_distance'].mean()],
        'std_s2dd': [confirmed['s2dd_distance'].std(), not_confirmed['s2dd_distance'].std()],
        'mean_score': [confirmed['score'].mean(), not_confirmed['score'].mean()],
        'mwu_p_distance': [p_dist, p_dist],
        'mwu_p_score': [p_score, p_score],
    })
    summary.to_csv(RESULTS_DIR / 'neoantigen_summary.csv', index=False)
    print(f"Saved summary to {RESULTS_DIR / 'neoantigen_summary.csv'}")


if __name__ == '__main__':
    main()
