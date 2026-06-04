#!/usr/bin/env python3
"""Regenerate panel1_test_with_fresh_predictions.csv (1003 rows) via the
authors' canonical xbcr_original_repo pipeline.

This is the canonical "fresh inference" path. It reproduces what the
2026-05-22 commit b5776e3c did (panel-exact reproduction of XBCR-net's
Fig 6 Panel E ΔAUROC=+0.163 / ΔAP=+0.112), using the unmodified XBCR-net
authors' code (Lou et al., Cell Research 2023 supplementary code).

Reproducibility status:
    - PANEL-EXACT: downstream compute_fig6_recal_data.py + this pipeline
      reproduces Fig 6 Panel E XBCR-net (ΔAUROC=+0.163 / ΔAP=+0.112) bit-exact
    - DISTANCE: this regenerated panel1_test_with_fresh_predictions.csv,
      when fed into compute_xbcr_panel1_distances.py, produces a
      distance array that differs from the cached
      `distance_cache_panel1.npz` (May 22 inline build, NEVER committed,
      labelled "unrecoverable" in reproducing_progress.md) by max |diff|
      ≈ 0.086. Sorted values agree, ranges identical. The discrepancy
      is from the May 22 inline pipeline being lost; the panel-level
      metrics are unaffected because they are rank-based.
    - PRACTICAL: for canonical Fig 6 Panel E reproduction, use the
      staged cached `distance_cache_panel1.npz` (which is what
      `compute_fig6_recal_data.py` reads). This script's output is
      for end-to-end audit only.

Pipeline (matches 2026-05-22 commit b5776e3c "Path A"):
    1. Stage `Model/xbcr_original_repo/` (already staged at
       `INPUT_DIR/Model/xbcr_original_repo/`)
    2. Pre-filter panel1_test antigens to length <= 280 chars
       (xbcr_original_repo/utils.py uses seq_shift=20 → effective max
       seq_length - shift = 300 - 20 = 280)
    3. Pre-filter panel1_test antibodies to Heavy/Light length in (10, 280]
    4. Write filtered antigens to data/binding/test/ag_to_pred/panel1_antigens.xlsx
    5. Write filtered antibodies to data/binding/test/ab_to_pred/panel1_antibodies.xlsx
    6. Run authors' main_infer.py in tf env (~5 min GPU) → cross-product
       (e.g., 5 antig × 1223 antib = 6115 predictions)
    7. Merge cross-product back to panel1_test (Heavy, Light, variant_seq)
       to get the 1003 panel1 pairings

Inputs (under INPUT_DIR):
    Model/xbcr_original_repo/ (full authors' XBCR-net code)
        - infer_rbd.py, main_infer.py, utils.py
        - models/binding/binding-XBCR_net/model_rbd_0.tf.* (trained weights)
    Data/retrospective_xbcr/extracted_panels/panel1_test.csv (rbd labels)

Output (gitignored):
    INPUT_DIR/results/xbcr_retrospective/fresh_inference/
        panel1_test_with_fresh_predictions.csv (1003 rows)

Requires:
    conda env `tf` (TF 2.5.0 + GPU)
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR

XBCR_DIR = os.path.join(INPUT_DIR, 'Model', 'xbcr_original_repo')
AG_PATH = os.path.join(XBCR_DIR, 'data', 'binding', 'test', 'ag_to_pred',
                        'panel1_antigens.xlsx')
AB_PATH = os.path.join(XBCR_DIR, 'data', 'binding', 'test', 'ab_to_pred',
                        'panel1_antibodies.xlsx')
RESULT_PATH = os.path.join(XBCR_DIR, 'data', 'binding', 'test', 'results',
                            'results_rbd_XBCR_net-0.xlsx')
PANEL1 = os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                       'extracted_panels', 'panel1_test.csv')
OUT_DIR = os.path.join(INPUT_DIR, 'results', 'xbcr_retrospective', 'fresh_inference')
OUT_PATH = os.path.join(OUT_DIR, 'panel1_test_with_fresh_predictions.csv')

# Effective max length: seq_length - seq_shift = 300 - 20 = 280
MAX_LEN = 280
MIN_LEN = 10


def prepare_inputs() -> None:
    panel1 = pd.read_csv(PANEL1)
    for c in ['Heavy', 'Light', 'variant_seq']:
        panel1[c] = panel1[c].fillna('').astype(str)
    print(f"  panel1_test: {len(panel1)} rows")

    # Filter to valid lengths for seq_shift=20
    valid = panel1[
        (panel1['Heavy'].str.len() > MIN_LEN) & (panel1['Heavy'].str.len() <= MAX_LEN) &
        (panel1['Light'].str.len() > MIN_LEN) & (panel1['Light'].str.len() <= MAX_LEN) &
        (panel1['variant_seq'].str.len() > MIN_LEN) & (panel1['variant_seq'].str.len() <= MAX_LEN)
    ]
    print(f"  panel1_valid (lengths in ({MIN_LEN}, {MAX_LEN}]): {len(valid)}")

    # Build unique antigen file
    antig = valid[['variant_name', 'variant_seq']].drop_duplicates().reset_index(drop=True)
    antig['Antig_full_seq'] = antig['variant_seq']
    antig['rbd'] = 1
    os.makedirs(os.path.dirname(AG_PATH), exist_ok=True)
    for f in os.listdir(os.path.dirname(AG_PATH)):
        if f != os.path.basename(AG_PATH):
            os.remove(os.path.join(os.path.dirname(AG_PATH), f))
    antig.to_excel(AG_PATH, index=False)
    print(f"  Wrote {len(antig)} antigens to {AG_PATH}")

    # Build unique antibody file with placeholder cols expected by authors' code
    antib = valid[['Name', 'Heavy', 'Light']].drop_duplicates().reset_index(drop=True)
    placeholder_cols = [
        'Ab or Nb', 'Binds to', "Doesn't Bind to", 'Neutralising Vs',
        'Not Neutralising Vs', 'Protein + Epitope', 'Origin', 'VH or VHH',
        'VL', 'Heavy V Gene', 'Heavy J Gene', 'Light V Gene', 'Light J Gene',
        'CDRH3', 'CDRL3', 'Structures', 'ABB Homology Model (if no structure)',
        'Sources', 'Date Added', 'Last Updated', 'Update Description',
        'Notes/Following Up?'
    ]
    for col in placeholder_cols:
        antib[col] = ''
    os.makedirs(os.path.dirname(AB_PATH), exist_ok=True)
    for f in os.listdir(os.path.dirname(AB_PATH)):
        if f != os.path.basename(AB_PATH):
            os.remove(os.path.join(os.path.dirname(AB_PATH), f))
    antib.to_excel(AB_PATH, index=False)
    print(f"  Wrote {len(antib)} antibodies to {AB_PATH}")


def run_inference() -> None:
    """Run authors' main_infer.py via conda env `tf` (TF 2.5.0)."""
    print(f"  Running xbcr_original_repo/main_infer.py in conda env 'tf' (~5 min GPU)...")
    # Clear stale pycache and results
    pycache = os.path.join(XBCR_DIR, '__pycache__')
    if os.path.isdir(pycache):
        import shutil
        shutil.rmtree(pycache)
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)
    proc = subprocess.run(
        ['conda', 'run', '-n', 'tf', 'python', 'main_infer.py'],
        cwd=XBCR_DIR, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(RESULT_PATH):
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"main_infer.py failed (rc={proc.returncode})")
    print(f"  Inference complete: {RESULT_PATH}")


def merge_with_panel1() -> int:
    res = pd.read_excel(RESULT_PATH)
    panel1 = pd.read_csv(PANEL1)
    for df in (res, panel1):
        for c in ['Heavy', 'Light', 'variant_seq']:
            df[c] = df[c].fillna('').astype(str)
    valid = panel1[
        (panel1['Heavy'].str.len() > MIN_LEN) & (panel1['Heavy'].str.len() <= MAX_LEN) &
        (panel1['Light'].str.len() > MIN_LEN) & (panel1['Light'].str.len() <= MAX_LEN) &
        (panel1['variant_seq'].str.len() > MIN_LEN) & (panel1['variant_seq'].str.len() <= MAX_LEN)
    ]
    res_uniq = res.drop_duplicates(subset=['Heavy', 'Light', 'variant_seq'])
    merged = valid.merge(
        res_uniq[['Heavy', 'Light', 'variant_seq', 'pred_prob']],
        on=['Heavy', 'Light', 'variant_seq'], how='left')
    n_unmatched = int(merged['pred_prob'].isna().sum())
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} unmatched")
    merged = merged.dropna(subset=['pred_prob'])
    merged['rbd'] = merged['rbd'].astype(int)
    os.makedirs(OUT_DIR, exist_ok=True)
    merged[['Heavy', 'Light', 'variant_seq', 'pred_prob', 'rbd',
            'variant_name', 'Name']].to_csv(OUT_PATH, index=False)
    return len(merged)


def main() -> int:
    print("[panel1] Step 1: prepare antigen + antibody inputs")
    prepare_inputs()
    print("[panel1] Step 2: run xbcr_original_repo authors' inference (cross-product)")
    run_inference()
    print("[panel1] Step 3: merge cross-product with panel1_test pairings")
    n = merge_with_panel1()
    print(f"\nSaved: {OUT_PATH} ({n} rows)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
