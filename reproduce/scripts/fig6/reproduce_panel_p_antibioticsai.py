#!/usr/bin/env python3
"""Reproduce a FULL-DATASET variant of Fig 6 Panel P (AntibioticsAI TDR
@ k = 1, 10, 20, 50, 100) -- now SUPERSEDED.

⚠ HISTORICAL: as of 2026-05-21, Panel P uses the FLIPPED halfsplit
(cal=odd, test=even) from ``recal_data/AntibioticsAI_samples.csv``,
matching Panels E and O. The full-dataset variant this script produces
(TDR@20 = 10/20 → 14/20, n=283, self-calibration) NO LONGER matches the
committed Panel P. For the canonical Panel P reproduction, see
``reproduce_panel_p_antibioticsai_halfsplit.py`` (TDR@20 = 7/20 → 12/20,
n=142). This script is retained as documentation of the prior full-
dataset arrangement and for users who want to compare the two designs.

The override branch at ``generate_fig6_redesign.py:256-275`` referenced
in the comments below was REMOVED on 2026-05-21 when Panel P switched
to the halfsplit; the ``main_test_with_distances.csv`` file this
script regenerates is no longer consumed by ``generate_fig6_redesign.py``
but is still produced for distance-computation reproducibility.

Original purpose: this script regenerates the upstream file
``results/antibioticsai_retrospective/reproduction/main_test_with_distances.csv``
that ``generate_fig6_redesign.py`` Panel P override branch (lines 256-275)
requires.  The override file was lost in the 2026-05-20 filter-repo
session (see REPRODUCIBILITY.md, ``Lost'' section); this script
recreates it from the still-present source data:

  * Wong et al. (Nature 2024) supplementary xlsx
    ``Data/retrospective_antibioticsai/supplementary/41586_2023_6887_MOESM4_ESM.xlsx``
    sheet ``All tested compounds`` (283 compounds with SMILES + ACTIVITY
    + ANTIBIOTIC_PS)
  * Wong et al. training set ``Model/AntibioticsAI/working_example/train.csv``
    (39,312 compounds, used as the S2DD reference)

Pipeline (mirrors ``eval_antibioticsai_retrospective.py:compute_morgan_tanimoto_distances``
and ``generate_fig6_redesign.py:256-275`` exactly):

  1. Morgan fingerprint (radius=2, 2048 bits) for both pools.
  2. Tanimoto similarity 283 x 39312; LogDist transform
     ``d = log(k * (1 - Tanimoto + b))`` with k=0.1, b=0.1.
  3. S2DD per test compound = mean of top-K=50 most-similar training
     distances.
  4. Full-dataset v2.7 self-calibration: ``fit_recalibration({'full':
     (y, raw_p, d)})`` then ``apply_recalibration(y, raw_p, d, ...,
     prev=cal_prev)``.
  5. TDR @ k = (# actives in top-k by score) / k.

Expected output under the canonical adaptive_n_bins formula
``max(4, min(8, n_minority // 8))`` (matches main.tex Methods
L376/L453/L498/L511/L531 + supplementary.tex L406, set by commit
40203223 on 2026-05-20):

    TDR@1  : 1/1   -> 1/1
    TDR@10 : 9/10  -> 9/10
    TDR@20 : 10/20 -> 14/20   <-- matches Fig 6 Panel P annotation
    TDR@50 : 14/50 -> 19/50
    TDR@100: 18/100 -> 25/100

After running this script, ``generate_fig6_redesign.py`` will pick up
the reconstituted ``main_test_with_distances.csv`` automatically and
regenerate Panel P from it.

Dependencies: rdkit, numpy, pandas, openpyxl, and the repo's
``calipper.core`` module.  Run from the repo root.

Usage:
    cd <published_repo>/CaliPPer
    python3 Manuscript/designed_figures/panels/fig6/scripts/reproduce_panel_p_antibioticsai.py

**Sibling script — halfsplit / adapted-retrospective variant:**
``reproduce_panel_p_antibioticsai_halfsplit.py`` implements the
*halfsplit* protocol that matches the AntibioticsAI design described
elsewhere in the manuscript (supplementary ``stab:models`` row 58
"Adapted | Compound halfsplit: 142 -> 141 (distance-sorted)"; main.tex
L232 Deployment paragraph "pilot-calibration half and a subsequent-
campaign half").  The halfsplit produces TDR@20 = 4/20 -> 6/20 on the
test half (test half has 15 actives), which is noisier than the
full-dataset variant but methodologically cleaner (genuine cal/test
separation, no test-leakage in PPV/NPV estimation).  The committed
panel in the manuscript shows 10/20 -> 14/20 (full-dataset); use the
halfsplit sibling if you prefer to regenerate Panel P under the
adapted-retrospective design instead.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

INPUT = Path(INPUT_DIR)
SUPP_XLSX = INPUT / "Data" / "retrospective_antibioticsai" / "supplementary" / "41586_2023_6887_MOESM4_ESM.xlsx"
TRAIN_CSV = INPUT / "Model" / "AntibioticsAI" / "working_example" / "train.csv"
OUT_DIR = Path(OUTPUT_DIR) / "antibioticsai_retrospective" / "reproduction"
OUT_CSV = OUT_DIR / "main_test_with_distances.csv"

K_FP, B_FP, K_TOPK = 0.1, 0.1, 50  # Methods L627: TCR/Antibiotics params

# --- Step 1: load source data ---------------------------------------
print("[1] Load Wong et al. source data")
tested = pd.read_excel(SUPP_XLSX, sheet_name="All tested compounds")
assert len(tested) == 283 and {"SMILES", "ACTIVITY", "ANTIBIOTIC_PS"}.issubset(tested.columns), \
    f"unexpected tested format: {len(tested)} rows, cols={list(tested.columns)}"
n_pos = int(tested["ACTIVITY"].sum())
print(f"    tested:  {len(tested):>5d} rows, {n_pos:>3d} actives (prev {n_pos/len(tested):.4f})")

train = pd.read_csv(TRAIN_CSV)
print(f"    train:   {len(train):>5d} rows")

# --- Step 2: Morgan fingerprint + Tanimoto + LogDist top-K ---------
print("[2] Morgan fingerprint + Tanimoto top-K LogDist (k=%.2f, b=%.2f, K=%d)"
      % (K_FP, B_FP, K_TOPK))
t0 = time.time()

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048)

train_fps_valid = [fp for fp in (smiles_to_fp(s) for s in train["SMILES"]) if fp is not None]
test_fps = [smiles_to_fp(s) for s in tested["SMILES"]]
print(f"    train_fps_valid: {len(train_fps_valid)}/{len(train)} "
      f"({time.time()-t0:.1f}s)")

distances = np.zeros(len(test_fps))
for i, tfp in enumerate(test_fps):
    if tfp is None:
        distances[i] = 0.0  # neutral; unparseable
        continue
    sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps_valid)
    d_all = np.log(K_FP * (1 - np.array(sims) + B_FP))
    distances[i] = np.sort(d_all)[:K_TOPK].mean()
    if (i + 1) % 100 == 0 or i == len(test_fps) - 1:
        print(f"    [{i+1:>3d}/{len(test_fps)}] {time.time()-t0:.1f}s")

# --- Step 3: write main_test_with_distances.csv --------------------
print(f"[3] Write {OUT_CSV}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_df = tested.copy()
out_df["distance"] = distances
out_df.to_csv(OUT_CSV, index=False)
print(f"    wrote {len(out_df)} rows; cols: {list(out_df.columns)}")
print(f"    distance range [{distances.min():.3f}, {distances.max():.3f}]")

# --- Step 4: full-dataset v2.7 self-calibration (Panel P branch) ---
print("[4] full-dataset v2.7 self-calibration (mirrors generate_fig6_redesign.py:256-275)")
from calipper.core import fit_recalibration, apply_recalibration

y = tested["ACTIVITY"].values.astype(int)
raw_p = tested["ANTIBIOTIC_PS"].values.astype(float)
d_aa = distances

ppv, npv, p_pos, p_neg, cal_prev = fit_recalibration({"full": (y, raw_p, d_aa)})
cal_p = apply_recalibration(y, raw_p, d_aa, ppv, npv, p_pos, p_neg, prev=cal_prev)

# --- Step 5: TDR @ k ----------------------------------------------
print("[5] TDR @ k = 1, 10, 20, 50, 100")
ks = [1, 10, 20, 50, 100]
raw_order = np.argsort(raw_p)[::-1]
cal_order = np.argsort(cal_p)[::-1]
print(f"\n  {'k':>5s}  {'raw TDR':>14s}  {'cal TDR':>14s}  {'delta':>8s}")
print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*8}")
for k in ks:
    raw_n = int(y[raw_order[:k]].sum())
    cal_n = int(y[cal_order[:k]].sum())
    print(f"  {k:>5d}  {raw_n}/{k:<3d} = {raw_n/k:.3f}  "
          f"{cal_n}/{k:<3d} = {cal_n/k:.3f}  {(cal_n-raw_n)/k:>+8.3f}")
print(f"\n  Expected (Fig 6 Panel P annotation): TDR@20 = 10/20 -> 14/20")
print(f"  This run:                              TDR@20 = "
      f"{int(y[raw_order[:20]].sum())}/20 -> {int(y[cal_order[:20]].sum())}/20")
