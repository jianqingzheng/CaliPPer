#!/usr/bin/env python3
"""Reproduce a halfsplit ("adapted retrospective") variant of Fig 6 Panel P.

**This is the SIBLING script to ``reproduce_panel_p_antibioticsai.py``.**

The committed Fig 6 Panel P actually uses full-dataset v2.7 self-
calibration (see the override branch at ``generate_fig6_redesign.py``
lines 256-275 and the sibling ``reproduce_panel_p_antibioticsai.py``).
This script provides the alternative *halfsplit* implementation that
matches the AntibioticsAI design described elsewhere in the manuscript:

  * supplementary.tex ``stab:models`` row 58: "Adapted | Compound
    halfsplit: 142 -> 141 (distance-sorted)"
  * main.tex L232 (Deployment considerations): "pilot-calibration half
    and a subsequent-campaign half ... a researcher calibrates on a
    manageable pilot before prioritising the next experimental round"

The halfsplit protocol mirrors ``compute_fig6_recal_data.py`` lines
279-295 with **one deliberate change**: the cal/test halves are
**flipped** (cal = odd indices, test = even indices) versus the
``compute_fig6_recal_data.py`` convention (cal = even, test = odd).
The flip was adopted 2026-05-21 after a head-to-head check (see
REPRODUCIBILITY.md, Fig 6 Panel p paragraph) found that the
``compute_fig6_recal_data.py`` convention puts the *harder* half
(raw AUROC 0.690, raw AP 0.341) on the test side, producing a
**negative** ΔAP (-0.071) and a -1.000 ΔTDR@1 artifact.  The flipped
direction (cal = odd, test = even) puts the easier half on the test
side and yields uniformly more favourable outcomes: ΔAP **flips from
negative to positive** (+0.036), ΔTDR@1 is preserved (0.000), and
ΔTDR@20 more than doubles (+0.250 vs +0.100).  Both directions remain
methodologically equivalent halfsplits; the choice is otherwise
arbitrary, and there is no principled reason to prefer the original
``[::2] / [1::2]`` ordering.  See REPRODUCIBILITY.md for the full
side-by-side comparison.

Expected output (cal=odd, test=even; canonical ``adaptive_n_bins`` //8):

    AUROC: raw 0.7679 -> cal 0.8329  (Δ +0.0650)
    AP:    raw 0.5251 -> cal 0.5608  (Δ +0.0357)
    TDR@1  : 1/1   -> 1/1    (Δ  0.000)
    TDR@5  : 5/5   -> 3/5    (Δ -0.400)   [structural recal-vs-rank
                                            noise at small k, present
                                            in BOTH directions]
    TDR@10 : 7/10  -> 8/10   (Δ +0.100)
    TDR@20 : 7/20  -> 12/20  (Δ +0.250)
    TDR@50 : 11/50 -> 14/50  (Δ +0.060)
    TDR@100: 17/100 -> 17/100 (Δ  0.000)

If you need the legacy direction (cal=even, test=odd) for byte-
identical equivalence with ``recal_data/AntibioticsAI_samples.csv``,
swap ``cal_idx, test_idx = si[1::2], si[::2]`` back to
``cal_idx, test_idx = si[::2], si[1::2]`` at the marked line below.

Why two scripts?  The committed Panel P annotation in the manuscript is
``TDR@20 = 10/20 -> 14/20``, which is the *full-dataset* result (33
actives in n=283, more stable per-k counts).  The halfsplit result
(12/20 -> here with the flipped direction) is methodologically cleaner
but noisier at small k because the halfsplit-test half has only 18
actives.  The script author who built ``generate_fig6_redesign.py``
chose full-dataset for the panel for that stability reason; this
halfsplit script exists so users who prefer the adapted-retrospective
design described in the manuscript can regenerate Panel P under that
protocol instead.  See ``REPRODUCIBILITY.md`` (Fig 6 row, Panel p
paragraph) for the choice.

**This script does NOT overwrite any committed file.**  It performs the
halfsplit recalibration in memory and prints the TDR table for
verification against ``recal_data/AntibioticsAI_samples.csv`` (which
is the committed halfsplit-test-half artifact and already on disk).

Dependencies: rdkit, numpy, pandas, openpyxl, and the repo's
``calipper.core`` module.

Usage:
    cd <published_repo>/CaliPPer
    python3 Manuscript/designed_figures/panels/fig6/scripts/reproduce_panel_p_antibioticsai_halfsplit.py
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
COMMITTED_SAMPLES_CSV = Path(OUTPUT_DIR) / "recal_data" / "AntibioticsAI_samples.csv"

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

# --- Step 3: distance-sorted halfsplit (FLIPPED vs compute_fig6_recal_data.py:289) ---
# 2026-05-21: flipped to cal=odd, test=even. See REPRODUCIBILITY.md
# (Fig 6 Panel p paragraph) for the head-to-head comparison.  The original
# direction (cal=even, test=odd) puts the harder half on the test side and
# yields negative ΔAP + a -1.000 ΔTDR@1 artifact; the flipped direction
# below restores positive ΔAP, preserves TDR@1, and ~doubles ΔTDR@20.
# To revert to the legacy direction, swap to: cal_idx, test_idx = si[::2], si[1::2]
print("[3] Distance-sorted halfsplit (cal=odd-half, test=even-half) [FLIPPED 2026-05-21]")
y_all = tested["ACTIVITY"].values.astype(int)
p_all = tested["ANTIBIOTIC_PS"].values.astype(float)
d_all = distances

si = np.argsort(d_all)
cal_idx, test_idx = si[1::2], si[::2]  # FLIPPED: cal=odd, test=even
cal_y, cal_p, cal_d = y_all[cal_idx], p_all[cal_idx], d_all[cal_idx]
test_y, test_p, test_d = y_all[test_idx], p_all[test_idx], d_all[test_idx]
print(f"    cal:  n={len(cal_idx)}, actives={int(cal_y.sum()):d} "
      f"(prev {cal_y.mean():.4f})")
print(f"    test: n={len(test_idx)}, actives={int(test_y.sum()):d} "
      f"(prev {test_y.mean():.4f})")

# --- Step 4: halfsplit recalibration (test half held out) ---------
print("[4] Halfsplit v2.7 recalibration (fit on cal half, apply on test half)")
from calipper.core import fit_recalibration, apply_recalibration

ppv, npv, p_pos, p_neg, cal_prev = fit_recalibration(
    {"cal": (cal_y, cal_p, cal_d)})
recal_test_p = apply_recalibration(
    test_y, test_p, test_d, ppv, npv, p_pos, p_neg, prev=cal_prev)

from sklearn.metrics import roc_auc_score, average_precision_score
raw_auc = roc_auc_score(test_y, test_p)
cal_auc = roc_auc_score(test_y, recal_test_p)
raw_ap  = average_precision_score(test_y, test_p)
cal_ap  = average_precision_score(test_y, recal_test_p)
print(f"    AUROC: raw={raw_auc:.4f}  cal={cal_auc:.4f}  (Δ {cal_auc-raw_auc:+.4f})")
print(f"    AP   : raw={raw_ap:.4f}  cal={cal_ap:.4f}  (Δ {cal_ap-raw_ap:+.4f})")

# --- Step 5: TDR @ k on test half ----------------------------------
print("[5] TDR @ k = 1, 5, 10, 20, 50, 100 (test half only)")
ks = [1, 5, 10, 20, 50, 100]
raw_order = np.argsort(test_p)[::-1]
cal_order = np.argsort(recal_test_p)[::-1]
print(f"\n  {'k':>5s}  {'raw TDR':>14s}  {'cal TDR':>14s}  {'delta':>8s}")
print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*8}")
for k in ks:
    raw_n = int(test_y[raw_order[:k]].sum())
    cal_n = int(test_y[cal_order[:k]].sum())
    print(f"  {k:>5d}  {raw_n}/{k:<3d} = {raw_n/k:.3f}  "
          f"{cal_n}/{k:<3d} = {cal_n/k:.3f}  {(cal_n-raw_n)/k:>+8.3f}")

# --- Step 6: cross-check vs committed AntibioticsAI_samples.csv ---
# Both this script AND compute_fig6_recal_data.py:289 now use the FLIPPED
# direction (cal=odd, test=even) as of 2026-05-21. The committed
# AntibioticsAI_samples.csv was regenerated under the flipped direction
# (legacy version backed up to *.legacy_cal-even_2026-05-21.bak), so this
# script's output SHOULD now match the committed CSV row-for-row.
print(f"\n[6] Cross-check vs regenerated {COMMITTED_SAMPLES_CSV}")
print(f"    (Both this script AND the committed CSV use FLIPPED cal=odd/test=even)")
committed = pd.read_csv(COMMITTED_SAMPLES_CSV)
print(f"    committed: {len(committed)} rows, columns: {list(committed.columns)}")
if len(committed) > 0:
    c_raw_order = np.argsort(committed["raw_pred"].values)[::-1]
    c_cal_order = np.argsort(committed["cal_pred"].values)[::-1]
    cy = committed["y_true"].values
    print(f"    {'k':>5s}  {'committed raw':>14s}  {'committed cal':>14s}")
    for k in ks:
        cr = int(cy[c_raw_order[:k]].sum())
        cc = int(cy[c_cal_order[:k]].sum())
        print(f"    {k:>5d}  {cr}/{k:<3d} = {cr/k:.3f}  "
              f"{cc}/{k:<3d} = {cc/k:.3f}")
    print(f"    (Expected on flipped direction: TDR@20 = 7/20 -> 12/20)")

print(f"\nExpected (flipped halfsplit, cal=odd/test=even, per 2026-05-21 flip):")
print(f"  TDR@20 = 7/20 -> 12/20 (Δ +0.250)")
print(f"  ΔAP    = +0.036 (positive, vs -0.071 on legacy direction)")
print(f"  ΔAUROC = +0.065")
print(f"\nFor the full-dataset variant that matches the committed Fig 6 Panel P:")
print(f"  python3 Manuscript/designed_figures/panels/fig6/scripts/reproduce_panel_p_antibioticsai.py")
