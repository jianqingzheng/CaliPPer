"""PanPep anchor ablation + bit-exact reproduction check.

Steps:
1. Load reconstructed PanPep predictions + reconstructed BLOSUM-sqrt distances
2. Apply the canonical halfsplit (peptide-identity, seed=42)
3. Run WITH-anchor recalibration → compare cal_pred against committed PanPep_samples.csv
4. Run WITHOUT-anchor recalibration → record as the missing ablation row
5. Produce the comparison table that should have been recorded on 2026-05-11
"""
import os
import sys
from pathlib import Path

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from calipper.core import fit_recalibration, apply_recalibration

R = Path(INPUT_DIR) / 'results'

# === Load reconstructed predictions + distances ===
pp = pd.concat([
    pd.read_csv(R / 'panpep_retrospective/reproduction/zeroshot_test_predictions.csv'),
    pd.read_csv(R / 'panpep_retrospective/reproduction/zeroshot_neg_predictions.csv'),
], ignore_index=True)
pp_bls = np.load(R / 'panpep_retrospective/blosum_sqrt/zeroshot_test_blosumsqrt_dist.npy')
pp['blosum_dist'] = pp_bls[:len(pp)]

print(f"Loaded reconstructed pool: n={len(pp)} ({int(pp['label'].sum())} pos / {len(pp) - int(pp['label'].sum())} neg)")
print(f"  Predictions mean: {pp['prediction'].mean():.6f}")
print(f"  Distance range:   [{pp['blosum_dist'].min():.3f}, {pp['blosum_dist'].max():.3f}]")

# === Apply canonical halfsplit (seed=42, peptide identity) ===
np.random.seed(42)
peptides = sorted(pp['peptide'].unique())
pep_shuf = np.random.permutation(peptides)
cal_peps = set(pep_shuf[:len(pep_shuf) // 2])
test_peps = set(pep_shuf[len(pep_shuf) // 2:])
cal_pp = pp[pp['peptide'].isin(cal_peps)].reset_index(drop=True)
test_pp = pp[pp['peptide'].isin(test_peps)].reset_index(drop=True)

cal_y = cal_pp['label'].values.astype(int)
cal_p = cal_pp['prediction'].values.astype(float)
cal_d = cal_pp['blosum_dist'].values
test_y = test_pp['label'].values.astype(int)
test_p = test_pp['prediction'].values.astype(float)
test_d = test_pp['blosum_dist'].values

print(f"\nCal: n={len(cal_y)}, pos={int(cal_y.sum())}, prev={cal_y.mean():.3f}")
print(f"Test: n={len(test_y)}, pos={int(test_y.sum())}, prev={test_y.mean():.3f}")

cal_data = {'zeroshot_cal': (cal_y, cal_p, cal_d)}

# === Build train_anchor from majority test ===
pp_maj = pd.read_csv(R / 'panpep_retrospective/reproduction/majority_test_predictions.csv')
pp_maj_y = pp_maj['label'].values.astype(int)
pp_maj_p = pp_maj['prediction'].values.astype(float)
_theta = 0.5
_pos_m = pp_maj_p >= _theta
_neg_m = pp_maj_p < _theta
pp_train_anchor = {
    'distance': 0.0, 'mp': float(pp_maj_p.mean()),
    'ppv': float(pp_maj_y[_pos_m].mean()), 'npv': float((1 - pp_maj_y[_neg_m]).mean()),
}
print(f"\nAnchor: d=0.0, mp={pp_train_anchor['mp']:.6f}, PPV={pp_train_anchor['ppv']:.6f}, NPV={pp_train_anchor['npv']:.6f}")

# === Raw test metrics ===
raw_auc = roc_auc_score(test_y, test_p)
raw_ap = average_precision_score(test_y, test_p)
raw_f1 = f1_score(test_y, (test_p >= 0.5).astype(int))
print(f"\nRaw test metrics: AUROC={raw_auc:.4f}, AP={raw_ap:.4f}, F1={raw_f1:.4f}")

# === Reference: committed PanPep_samples.csv ===
committed = pd.read_csv(Path(OUTPUT_DIR) / 'recal_data' / 'PanPep_samples.csv')
com_auc_raw = roc_auc_score(committed['y_true'], committed['raw_pred'])
com_auc_cal = roc_auc_score(committed['y_true'], committed['cal_pred'])
print(f"\nCommitted PanPep_samples.csv:")
print(f"  raw AUROC: {com_auc_raw:.4f}")
print(f"  cal AUROC: {com_auc_cal:.4f}")
print(f"  ΔAUROC:    {com_auc_cal - com_auc_raw:+.4f}")

# === Helper ===
def tdr_at(y, scores, k):
    order = np.argsort(scores)[::-1]
    return int(y[order[:k]].sum()), k

def run_config(anchor, label):
    ppv, npv, p_pos, p_neg, cal_prev = fit_recalibration(cal_data, train_anchor=anchor)
    cs = apply_recalibration(test_y, test_p, test_d, ppv, npv, p_pos, p_neg, prev=cal_prev)
    auc = roc_auc_score(test_y, cs)
    ap = average_precision_score(test_y, cs)
    f1 = f1_score(test_y, (cs >= 0.5).astype(int))
    print(f"\n=== {label} ===")
    print(f"  AUROC: raw {raw_auc:.4f} -> cal {auc:.4f}  Δ {auc-raw_auc:+.4f}")
    print(f"  AP   : raw {raw_ap:.4f} -> cal {ap:.4f}  Δ {ap-raw_ap:+.4f}")
    print(f"  F1   : raw {raw_f1:.4f} -> cal {f1:.4f}  Δ {f1-raw_f1:+.4f}")
    for k in [5, 10, 20, 50, 100]:
        rn, _ = tdr_at(test_y, test_p, k)
        cn, _ = tdr_at(test_y, cs, k)
        print(f"  TDR@{k:>3d}: raw {rn}/{k} -> cal {cn}/{k}  Δ {(cn-rn)/k:+.3f}")
    return auc, ap, f1, cs

# === Run WITH-anchor (canonical, expect to match committed) ===
with_auc, with_ap, with_f1, with_cs = run_config(pp_train_anchor, "WITH anchor (canonical config)")

# === Reproduction check ===
print("\n" + "=" * 70)
print("REPRODUCTION CHECK (WITH-anchor vs committed PanPep_samples.csv)")
print("=" * 70)
# Sort both by (peptide, binding_TCR) to align
test_pp_sorted = test_pp.copy()
test_pp_sorted['cal_pred_new'] = with_cs
test_pp_sorted = test_pp_sorted.sort_values(['peptide', 'binding_TCR']).reset_index(drop=True)
# Committed PanPep_samples.csv has no peptide column; compare sorted distributions
sorted_new = np.sort(with_cs)
sorted_com = np.sort(committed['cal_pred'].values)
diff = sorted_new - sorted_com
print(f"  Sorted cal_pred diff stats: min={diff.min():.2e}, max={diff.max():.2e}, mean_abs={np.abs(diff).mean():.2e}")
match_6dp = np.allclose(sorted_new, sorted_com, atol=1e-6)
match_4dp = np.allclose(sorted_new, sorted_com, atol=1e-4)
print(f"  Match to 6 decimal places: {match_6dp}")
print(f"  Match to 4 decimal places: {match_4dp}")
print(f"  ΔAUROC new: {with_auc - raw_auc:+.6f}")
print(f"  ΔAUROC committed (from CSV): {com_auc_cal - com_auc_raw:+.6f}")
print(f"  Match? {abs((with_auc - raw_auc) - (com_auc_cal - com_auc_raw)) < 1e-6}")

# === Run WITHOUT-anchor ===
without_auc, without_ap, without_f1, without_cs = run_config(None, "WITHOUT anchor (ablation)")

# === Summary table ===
print("\n" + "=" * 70)
print("ABLATION SUMMARY: with vs without train_anchor")
print("=" * 70)
print(f"{'Metric':<10}{'WITHOUT anchor':>18}{'WITH anchor':>16}{'Anchor effect':>18}")
print(f"{'AUROC Δ':<10}{f'{without_auc-raw_auc:+.4f}':>18}{f'{with_auc-raw_auc:+.4f}':>16}{f'{with_auc-without_auc:+.4f}':>18}")
print(f"{'AP Δ':<10}{f'{without_ap-raw_ap:+.4f}':>18}{f'{with_ap-raw_ap:+.4f}':>16}{f'{with_ap-without_ap:+.4f}':>18}")
print(f"{'F1 Δ':<10}{f'{without_f1-raw_f1:+.4f}':>18}{f'{with_f1-raw_f1:+.4f}':>16}{f'{with_f1-without_f1:+.4f}':>18}")
