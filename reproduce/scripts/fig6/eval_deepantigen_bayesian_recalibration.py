#!/usr/bin/env python3
"""Compute S2DD-SW (BLOSUM + top-K) distances for ImmuneCODE (Fig 6, Stage 0f).

Regenerates the ImmuneCODE SW-BLOSUM top-K distance file
(``immunecode_sw_topk_distances.csv``) that Fig 6 Panel C/D consumes. The
deepAntigen recalibration and prediction themselves are computed label-free by
the shared library in ``compute_fig6_recal_data.py`` /
``compute_fig6_panel_c_d.py`` (``fit_recalibration`` / ``apply_recalibration``
anchored on the calibration-set prevalence); this stage only produces the
distance input. (An earlier standalone recalibration prototype lived here; it
was superseded by the library path and removed.)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Self-contained path anchors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

# Read existing distance CSVs from OUTPUT_DIR (where phase 3 wrote them)
# Fall back to INPUT_DIR (canonical cached) if not in OUTPUT_DIR.
_OUTPUT_S2DD = Path(OUTPUT_DIR) / 'deepantigen_retrospective' / 's2dd_degradation'
_INPUT_S2DD = Path(INPUT_DIR) / 'results' / 'deepantigen_retrospective' / 's2dd_degradation'
S2DD_DIR = _OUTPUT_S2DD if (_OUTPUT_S2DD / 'immunecode_with_distances.csv').exists() else _INPUT_S2DD


def compute_immunecode_sw_topk(k=0.1, b=0.1, K=50):
    """Compute S2DD-SW (BLOSUM + topK) distances for ImmuneCODE.

    Mirrors the zero-shot computation in fig_s2dd_vs_sw_comparison.py.
    Uses BLOSUM62 SW similarity, topK=50 mean, expanded via peptide frequency.
    """
    import parasail
    import time

    cache_path = S2DD_DIR / 'immunecode_sw_topk_distances.csv'
    if cache_path.exists():
        print(f"  Loading cached ImmuneCODE S2DD-SW distances")
        return pd.read_csv(cache_path)['distance'].values

    print(f"  Computing S2DD-SW distances for ImmuneCODE (this takes ~2-3 min)...")
    train_path = Path(INPUT_DIR) / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data' / 'train.csv'
    train = pd.read_csv(train_path)
    ic = pd.read_csv(S2DD_DIR / 'immunecode_with_distances.csv')

    train_peps_rows = train['peptide'].values
    unique_train_peps = np.unique(train_peps_rows)
    test_peps = ic['peptide'].values

    # Self-scores
    train_self = {s: parasail.sw_stats(s, s, 10, 1, parasail.blosum62).score
                  for s in unique_train_peps}

    # Unique test peptides → similarity against all unique train peptides
    unique_test_peps = np.unique(test_peps)
    print(f"    Unique test peptides: {len(unique_test_peps)}, "
          f"unique train peptides: {len(unique_train_peps)}")
    t0 = time.time()
    sim_by_test_pep = {}
    for i, q in enumerate(unique_test_peps):
        q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
        if q_self <= 0:
            sim_by_test_pep[q] = np.zeros(len(unique_train_peps))
            continue
        sims = np.array([
            parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score /
            np.sqrt(q_self * train_self[r]) if train_self[r] > 0 else 0.0
            for r in unique_train_peps
        ])
        sim_by_test_pep[q] = sims
        if (i + 1) % 50 == 0 or i == len(unique_test_peps) - 1:
            print(f"    [{i+1:>3}/{len(unique_test_peps)}] {(i+1)/len(unique_test_peps)*100:.0f}% "
                  f"({time.time()-t0:.0f}s)")

    # Expand: each training row has the sim corresponding to its peptide
    pep_to_rows = {p: np.where(train_peps_rows == p)[0] for p in unique_train_peps}
    distances = np.zeros(len(test_peps))
    for i, q in enumerate(test_peps):
        pep_sims = sim_by_test_pep[q]
        row_sims = np.zeros(len(train_peps_rows))
        for pep_idx, p in enumerate(unique_train_peps):
            row_sims[pep_to_rows[p]] = pep_sims[pep_idx]
        topk_sims = np.sort(row_sims)[::-1][:K]
        topk_dists = np.log(k * (1.0 - topk_sims + b))
        distances[i] = topk_dists.mean()

    pd.DataFrame({'distance': distances}).to_csv(cache_path, index=False)
    print(f"    Cached to {cache_path}")
    return distances


def main():
    print("Computing ImmuneCODE S2DD-SW (BLOSUM top-K) distances for Fig 6 Panel C/D...")
    distances = compute_immunecode_sw_topk()
    print(f"  Done: {len(distances)} ImmuneCODE distances "
          f"-> {S2DD_DIR / 'immunecode_sw_topk_distances.csv'}")
    return distances


if __name__ == '__main__':
    main()
