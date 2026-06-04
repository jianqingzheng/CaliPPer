#!/usr/bin/env python3
"""Stage Levenshtein per-model TCR distance arrays for Fig 5 panels l, n, o.

Fig 5's scripts (`gen_fig5_lev_vs_blosum_recal.py`, `generate_fig5_new_panels.py`,
`gen_subset_recal_scatter.py`) expect Levenshtein per-model TCR distance
arrays at:
    INPUT_DIR/results/fig2_cache/{model}_ct_{ts}_dist.npy

for 5 TCR models × 6 test sets = 30 files. These files don't exist anywhere
on disk (deposit nor research repo), but the underlying Levenshtein
distances are SEQUENCE-derived (independent of model predictions), so they
can be regenerated WITHOUT model retraining.

This script:
1. Runs `compute_fig2_levlog_distances.py` to produce shared
   `{ts}_dist.npy` arrays at OUTPUT_DIR/fig2_cache_lev/
   (1 file per test set, distances against the seen-pool training pool —
   same for all 5 models since the test set is shared).
2. Copies each shared `{ts}_dist.npy` to per-model
   INPUT_DIR/results/fig2_cache/{model}_ct_{ts}_dist.npy for 5 TCR models.

After staging, re-run `reproduce_fig5.sh` and the previously-blocked panels
(l, n, o) will produce real content.

Note: this is NOT a model retraining step (no GPU, no per-model variation).
The distances are computed by Levenshtein on raw test set sequences, which
are identical regardless of which model is being evaluated.

Usage (called automatically by `reproduce_fig5.sh`):
    python reproduce/scripts/fig5/stage_lev_distances_for_fig5.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TEST_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined',
              'mcpas', 'iedb_sars']

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEV_DIST_SOURCE = os.path.join(OUTPUT_DIR, 'fig2_cache_lev')
TARGET_DIR = os.path.join(INPUT_DIR, 'results', 'fig2_cache')


def ensure_lev_distances_computed() -> bool:
    """Run compute_fig2_levlog_distances.py if its output isn't present."""
    expected_files = [os.path.join(LEV_DIST_SOURCE, f'{ts}_dist.npy')
                       for ts in TEST_SETS]
    if all(os.path.exists(f) for f in expected_files):
        print(f"[stage_lev] All 6 shared Lev arrays already cached at {LEV_DIST_SOURCE}")
        return True
    print(f"[stage_lev] Running compute_fig2_levlog_distances.py...")
    compute_script = os.path.join(SCRIPTS_DIR, 'compute_fig2_levlog_distances.py')
    proc = subprocess.run([sys.executable, compute_script], capture_output=False)
    if proc.returncode != 0:
        print(f"[stage_lev] ERROR: compute_fig2_levlog_distances.py failed (rc={proc.returncode})")
        return False
    return all(os.path.exists(f) for f in expected_files)


def stage_per_model_copies() -> int:
    """Copy each shared {ts}_dist.npy as per-model {model}_ct_{ts}_dist.npy.

    Returns count of files staged.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)
    n_staged = 0
    for ts in TEST_SETS:
        src = os.path.join(LEV_DIST_SOURCE, f'{ts}_dist.npy')
        if not os.path.exists(src):
            print(f"[stage_lev] WARNING: source missing: {src}")
            continue
        for model in TCR_MODELS:
            dst = os.path.join(TARGET_DIR, f'{model}_ct_{ts}_dist.npy')
            if os.path.exists(dst):
                continue  # don't overwrite (per user 'don't overwrite' rule)
            shutil.copy2(src, dst)
            n_staged += 1
    return n_staged


def main() -> int:
    print(f"[stage_lev] Staging Levenshtein per-model TCR distance arrays for Fig 5")
    print(f"  Source: {LEV_DIST_SOURCE} (shared {{ts}}_dist.npy from fig2 pipeline)")
    print(f"  Target: {TARGET_DIR} (per-model {{model}}_ct_{{ts}}_dist.npy)")
    print()
    if not ensure_lev_distances_computed():
        return 1
    n = stage_per_model_copies()
    print(f"\n[stage_lev] Staged {n} per-model copies "
          f"({len(TCR_MODELS)} models × {len(TEST_SETS)} test sets, "
          f"skipped existing).")
    if n == 0 and len(os.listdir(TARGET_DIR)) > 0:
        print(f"[stage_lev] All target files already existed (no overwrites).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
