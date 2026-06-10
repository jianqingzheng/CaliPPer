"""End-to-end fig6 XBCR-net pipeline on FRESH inference predictions.

Self-contained: reads inputs from reproduce/data/input/ (gitignored,
committed to reproduce/data/input/) and writes outputs to
reproduce/data/output/recal_data/ (also gitignored). Does not modify
any committed file.

Required input (must be staged by [retired]):
  reproduce/data/input/results/xbcr_retrospective/fresh_inference/panel1_test_with_fresh_predictions.csv

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/fig6/reproduce_fig6_xbcr.py
"""
import os, sys
import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.combine_first_helpers import compute_chain_weights, compute_combine_first_distances
from calipper.core import fit_recalibration, apply_recalibration
from calipper.general_evaluator import safe_metric

# ── Settings (copied from eval_xbcr_panel2_omicron.py + compute_fig6_recal_data.py) ──
CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
k, b, K = 0.1, 0.03, 30
PREVALENCE_PRIOR = 0.5  # corrected per fig6 comment "simulated negatives inflate cal prevalence"

OMICRON_RBD = {
    'omicron': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
    'BA2': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLQSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
    'BA4': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
}
VARIANT_MAP = {'omicron': 'omicron', 'omicron ': 'omicron', 'BA2': 'BA2', 'BA4': 'BA4'}

# ── 1. Load all data ──
print("=== Loading data ===")
panel1 = pd.read_csv(os.path.join(INPUT_DIR, 'results', 'xbcr_retrospective',
                                    'fresh_inference', 'panel1_test_with_fresh_predictions.csv'))
panel1['Light'] = panel1['Light'].fillna('')
panel1['variant_seq'] = panel1['variant_seq'].fillna('')
print(f"  panel1 fresh: {len(panel1)} (rbd=1: {(panel1['rbd']==1).sum()}, rbd=0: {(panel1['rbd']==0).sum()})")

train = pd.read_csv(os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                                  'extracted_panels', 'panel1_training.csv'))
train['Light'] = train['Light'].fillna('')
train['variant_seq'] = train['variant_seq'].fillna('')
print(f"  training: {len(train)}")

panel2 = pd.read_csv(os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                                   'extracted_panels', 'panel2_therapeutic_mab.csv'))
panel2_valid = panel2[panel2['Heavy'].notna() & panel2['Light'].notna()].copy()
panel2_valid['variant_clean'] = panel2_valid['variant'].map(VARIANT_MAP)
panel2_valid['variant_seq'] = panel2_valid['variant_clean'].map(OMICRON_RBD)
panel2_with_seq = panel2_valid[panel2_valid['variant_seq'].notna()].copy()
print(f"  panel2 (Omicron): {len(panel2_with_seq)}")

# ── 2. Compute 3-chain sigma_C weights from training ──
weights, _ = compute_chain_weights(train, CHAIN_COLS, k, b, K, formula='sigma_C')
print(f"\n=== sigma_C weights: Heavy={weights[0]:.4f}, Light={weights[1]:.4f}, variant_seq={weights[2]:.4f}")

# ── 3. Compute distances for panel1 + panel2 ──
print("\n=== Computing distances ===")
p1_dist = compute_combine_first_distances(panel1, train, CHAIN_COLS, weights, k, b, K)
print(f"  panel1: shape={p1_dist.shape}, range=[{p1_dist.min():.3f}, {p1_dist.max():.3f}]")
p2_dist = compute_combine_first_distances(panel2_with_seq, train, CHAIN_COLS, weights, k, b, K)
print(f"  panel2: shape={p2_dist.shape}, range=[{p2_dist.min():.3f}, {p2_dist.max():.3f}]")

# ── 4. Build cal_data ──
cal_data = {
    'panel1': (
        panel1['rbd'].values.astype(int),
        panel1['pred_prob'].values.astype(float),
        p1_dist,
    )
}

# ── 5. fit_recalibration + apply_recalibration ──
print("\n=== Recalibration ===")
ppv_p, npv_p, p_pos, p_neg, cal_prev = fit_recalibration(cal_data, threshold=PREVALENCE_PRIOR)
print(f"  fit OK: cal_prev={cal_prev:.4f}, p_pos={p_pos:.4f}, p_neg={p_neg:.4f}")

cs = apply_recalibration(
    panel2_with_seq['gt_binds'].values.astype(int),
    panel2_with_seq['prediction_score'].values.astype(float),
    p2_dist,
    ppv_p, npv_p, p_pos, p_neg,
    prev=PREVALENCE_PRIOR
)

# ── 6. Save fresh per-sample CSV ──
out = pd.DataFrame({
    'y_true':   panel2_with_seq['gt_binds'].astype(int).values,
    'raw_pred': panel2_with_seq['prediction_score'].astype(float).values,
    'cal_pred': cs,
    'distance': p2_dist,
})
os.makedirs(os.path.join(OUTPUT_DIR, 'recal_data'), exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'recal_data', 'XBCR-net_fresh_samples.csv')
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── 7. Metrics ──
print("\n=== Metric comparison (fresh 1003-row cal vs committed 1293-row cal) ===")
raw_auc = safe_metric('aucroc', out['y_true'].values, out['raw_pred'].values)
cal_auc = safe_metric('aucroc', out['y_true'].values, out['cal_pred'].values)
raw_ap  = safe_metric('ap',     out['y_true'].values, out['raw_pred'].values)
cal_ap  = safe_metric('ap',     out['y_true'].values, out['cal_pred'].values)
print(f"FRESH:     raw_AUROC={raw_auc:.4f}  cal_AUROC={cal_auc:.4f}  ΔAUROC={cal_auc-raw_auc:+.4f}")
print(f"FRESH:     raw_AP   ={raw_ap:.4f}  cal_AP   ={cal_ap:.4f}  ΔAP   ={cal_ap-raw_ap:+.4f}")

ref = pd.read_csv(os.path.join(OUTPUT_DIR, 'recal_data', 'XBCR-net_samples.csv'))
raw_auc_c = safe_metric('aucroc', ref['y_true'].values, ref['raw_pred'].values)
cal_auc_c = safe_metric('aucroc', ref['y_true'].values, ref['cal_pred'].values)
raw_ap_c  = safe_metric('ap',     ref['y_true'].values, ref['raw_pred'].values)
cal_ap_c  = safe_metric('ap',     ref['y_true'].values, ref['cal_pred'].values)
print(f"COMMITTED: raw_AUROC={raw_auc_c:.4f}  cal_AUROC={cal_auc_c:.4f}  ΔAUROC={cal_auc_c-raw_auc_c:+.4f}")
print(f"COMMITTED: raw_AP   ={raw_ap_c:.4f}  cal_AP   ={cal_ap_c:.4f}  ΔAP   ={cal_ap_c-raw_ap_c:+.4f}")

# Same panel2 inputs → same y_true and raw_pred
if len(out) == len(ref):
    print(f"\n=== Per-sample diff vs committed ===")
    print(f"  y_true exact: {np.array_equal(out.y_true.values, ref.y_true.values)}")
    print(f"  raw_pred max diff: {np.abs(out.raw_pred.values - ref.raw_pred.values).max():.3e}")
    print(f"  distance max diff: {np.abs(out.distance.values - ref.distance.values).max():.3e}")
    print(f"  cal_pred max diff: {np.abs(out.cal_pred.values - ref.cal_pred.values).max():.3e}")
