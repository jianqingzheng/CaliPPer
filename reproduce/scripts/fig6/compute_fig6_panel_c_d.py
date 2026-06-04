#!/usr/bin/env python3
"""Compute Panel C (prediction scatter) and Panel D (3-method comparison) data for fig6.

For each of 5 retrospective studies:
  - S2DD: predict_metric() from v2.7 (uses distance + prediction DRE + vbias)
  - PAPE: prediction-only DRE (no distance input) — Eq.4 calibrator
  - M-CBPE: prediction-only DRE (no distance input) — logistic calibrator

Outputs:
  recal_data/fig6_panel_c_predictions.csv   — 10 rows: 5 studies x 2 metrics (actual, predicted)
  recal_data/fig6_prediction_3method.csv    — 10 rows: 5 studies x 2 metrics (S2DD, PAPE, MCBPE errors)

Usage:
    cd <published_repo>/CaliPPer
    python Manuscript/designed_figures/panels/fig6/scripts/compute_fig6_panel_c_d.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.general_evaluator import safe_metric
from calipper.core import predict_metric  # v2.7 default

# PAPE and M-CBPE — same modules as fig3/scripts/compute_dataset_level_3method.py
from PAPE.pape_core import (estimate_importance_weights, fit_weighted_calibration,
                             apply_calibration, estimate_metric as pape_eq4)
from MCBPE.mcbpe_core import (estimate_density_ratios, fit_weighted_calibrator,
                               calibrate_predictions, estimate_metric_from_calibrated)

RESULTS = os.path.join(INPUT_DIR, 'results')
OUT_DIR = os.path.join(OUTPUT_DIR, 'recal_data')
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ['aucroc', 'ap']


def pape_predict_no_distance(cal_y, cal_p, test_p, metric):
    """PAPE prediction using prediction-only DRE (no distance features)."""
    w, _, _ = estimate_importance_weights(
        cal_p.reshape(-1, 1), test_p.reshape(-1, 1))
    cm = fit_weighted_calibration(cal_p, cal_y, w)
    c = apply_calibration(cm, test_p)
    return pape_eq4(c, test_p, metric, threshold=0.5)


def pape_predict_2d(cal_y, cal_p, cal_d, test_p, test_d, metric):
    """PAPE prediction using 2D DRE (distance + prediction features).

    Same DRE as CaliPPer uses internally — fair comparison since both
    methods have access to distance information.
    """
    w, _, _ = estimate_importance_weights(
        np.stack([cal_d, cal_p], axis=1),
        np.stack([test_d, test_p], axis=1))
    cm = fit_weighted_calibration(cal_p, cal_y, w)
    c = apply_calibration(cm, test_p)
    return pape_eq4(c, test_p, metric, threshold=0.5)


def mcbpe_predict_no_distance(cal_y, cal_p, test_p, metric):
    """M-CBPE prediction using prediction-only DRE (no distance features).

    Uses the proper M-CBPE module (same as fig3 audit):
      1. estimate_density_ratios on [prediction] features only
      2. fit_weighted_calibrator on cal
      3. calibrate_predictions for test
      4. estimate_metric_from_calibrated
    """
    w, _ = estimate_density_ratios(
        cal_p.reshape(-1, 1), test_p.reshape(-1, 1))
    cal = fit_weighted_calibrator(cal_p, cal_y, w)
    c = calibrate_predictions(cal, test_p)
    return estimate_metric_from_calibrated(c, metric)


# ═══════════════════════════════════════════════════════════════════
# Study-specific data loading and prediction
# ═══════════════════════════════════════════════════════════════════

panel_c_rows = []
panel_d_rows = []


def add_result(study, metric, actual, s2dd_pred, pape_pred, mcbpe_pred):
    """Record results for both Panel C and Panel D."""
    panel_c_rows.append({
        'study': study, 'metric': metric,
        'actual': actual, 'predicted': s2dd_pred,
    })
    panel_d_rows.append({
        'study': study, 'metric': metric,
        'S2DD': abs(s2dd_pred - actual),
        'PAPE': abs(pape_pred - actual),
        'MCBPE': abs(mcbpe_pred - actual),
    })
    print(f'  {metric}: actual={actual:.4f}  S2DD={s2dd_pred:.4f}(err={abs(s2dd_pred-actual):.4f})  '
          f'PAPE={pape_pred:.4f}(err={abs(pape_pred-actual):.4f})  '
          f'M-CBPE={mcbpe_pred:.4f}(err={abs(mcbpe_pred-actual):.4f})')


# ── 1. deepAntigen ────────────────────────────────────────────────
print("=" * 70)
print("1. deepAntigen — zero-shot(cal) → ImmuneCODE(test), BLOSUM-SW distances")
print("=" * 70)

zs_pred = pd.read_csv(os.path.join(RESULTS, 'deepantigen_retrospective', 'reproduction',
                                     'zero_shot_predictions.csv'))
zs_dist_df = pd.read_csv(os.path.join(RESULTS, 'deepantigen_retrospective', 's2dd_degradation',
                                        'zero_shot_sw_topk_distances.csv'))
ic_pred_path = os.path.join(RESULTS, 'deepantigen_retrospective', 's2dd_degradation',
                              'immunecode_with_distances.csv')
ic_dist_df = pd.read_csv(os.path.join(RESULTS, 'deepantigen_retrospective', 's2dd_degradation',
                                        'immunecode_sw_topk_distances.csv'))
ic_df = pd.read_csv(ic_pred_path)

# Cal: zero-shot (1714 samples)
cal_y_da = zs_pred['label'].values.astype(int)
cal_p_da = zs_pred['score'].values.astype(float)
cal_d_da = zs_dist_df['distance'].values.astype(float)
n_cal = min(len(cal_y_da), len(cal_d_da))
cal_y_da, cal_p_da, cal_d_da = cal_y_da[:n_cal], cal_p_da[:n_cal], cal_d_da[:n_cal]

# Test: ImmuneCODE (50K subsample)
test_y_da = ic_df['label'].values.astype(int)
test_p_da = ic_df['prediction'].values.astype(float)
test_d_da = ic_dist_df['distance'].values.astype(float)
n_test = min(len(test_y_da), len(test_d_da))
test_y_da, test_p_da, test_d_da = test_y_da[:n_test], test_p_da[:n_test], test_d_da[:n_test]

print(f'  Cal: n={n_cal}, pos={cal_y_da.sum()}')
print(f'  Test: n={n_test}, pos={test_y_da.sum()}')

cal_data_da = {'zero_shot': (cal_y_da, cal_p_da, cal_d_da)}

# Test 1: ImmuneCODE (50K, computational labels — meaningful for prediction)
for metric in METRICS:
    actual = safe_metric(metric, test_y_da, test_p_da)
    result = predict_metric(cal_data_da, test_p_da, test_d_da, metrics=[metric])
    s2dd_pred = result['estimated'][metric]
    pape_pred = pape_predict_2d(cal_y_da, cal_p_da, cal_d_da, test_p_da, test_d_da, metric)
    mcbpe_pred = mcbpe_predict_no_distance(cal_y_da, cal_p_da, test_p_da, metric)
    add_result('deepAntigen', metric, actual, s2dd_pred, pape_pred, mcbpe_pred)

# NOTE: Neoantigen (100 samples) REMOVED from Panel C/D.
# The 100 neoantigens were pre-selected by the model (top 20 per patient),
# so they are not an independent test set for performance PREDICTION.
# ImmuneCODE above is the correct prediction test set.
# Neoantigen recalibration results are still shown in panels i/j (ROC/reranking).


# ── 2. AntibioticsAI ─────────────────────────────────────────────
# TRUE CROSS-DATASET: main_test(283) cal → beta-lactam(505) test for ALL 3 methods
# Beta-lactam compounds are structurally distinct (0 SMILES overlap with main_test)
# Per-sample predictions from paper's supplementary data (Chemprop ensemble scores)
print("\n" + "=" * 70)
print("2. AntibioticsAI — main_test(283) cal → beta-lactam(505) test, ALL methods")
print("=" * 70)

aa_df = pd.read_csv(os.path.join(RESULTS, 'antibioticsai_retrospective', 'reproduction',
                                   'main_test_with_distances.csv'))

# Extract beta-lactam test data from paper supplementary
_xls = pd.ExcelFile(os.path.join(INPUT_DIR, 'Data', 'retrospective_antibioticsai',
                                   'supplementary', '41586_2023_6887_MOESM4_ESM.xlsx'))
_bl_raw = pd.read_excel(_xls, 'B-lactam-withheld train+test', header=1)
_bl_test = _bl_raw.iloc[:, 3:6].copy()
_bl_test.columns = ['SMILES', 'ACTIVITY', 'PREDICTION_SCORE']
_bl_test = _bl_test.dropna(subset=['SMILES'])
_bl_test['ACTIVITY'] = _bl_test['ACTIVITY'].astype(int)
_bl_test['PREDICTION_SCORE'] = _bl_test['PREDICTION_SCORE'].astype(float)

# Cal: main_test (283 compounds, all with labels + predictions + distances)
cal_y_aa = aa_df['ACTIVITY'].values.astype(int)
cal_p_aa = aa_df['ANTIBIOTIC_PS'].values.astype(float)
cal_d_aa = aa_df['distance'].values.astype(float)

# Test: beta-lactam (505 compounds, labels + Chemprop predictions from paper)
test_y_aa = _bl_test['ACTIVITY'].values.astype(int)
test_p_aa = _bl_test['PREDICTION_SCORE'].values.astype(float)

# Beta-lactam distances (pre-computed Morgan FP)
# Use _full version: distances from beta-lactam to FULL training set (same reference as cal)
# NOT distance_cache_blactam.npz which uses beta-lactam-withheld training (different reference)
bl_dist_path = os.path.join(RESULTS, 'antibioticsai_retrospective', 'distance_cache_blactam_full.npz')
test_d_aa = np.load(bl_dist_path)['distances'][:len(test_y_aa)]

print(f'  Cal (main_test): n={len(cal_y_aa)}, pos={cal_y_aa.sum()} ({cal_y_aa.mean():.1%})')
print(f'  Test (beta-lactam): n={len(test_y_aa)}, pos={test_y_aa.sum()} ({test_y_aa.mean():.1%})')

cal_data_aa = {'main_test': (cal_y_aa, cal_p_aa, cal_d_aa)}

for metric in METRICS:
    actual = safe_metric(metric, test_y_aa, test_p_aa)
    result = predict_metric(cal_data_aa, test_p_aa, test_d_aa, metrics=[metric])
    s2dd_pred = result['estimated'][metric]
    pape_pred = pape_predict_2d(cal_y_aa, cal_p_aa, cal_d_aa, test_p_aa, test_d_aa, metric)
    mcbpe_pred = mcbpe_predict_no_distance(cal_y_aa, cal_p_aa, test_p_aa, metric)
    add_result('AntibioticsAI', metric, actual, s2dd_pred, pape_pred, mcbpe_pred)


# ── 3. BigMHC ────────────────────────────────────────────────────
# HALFSPLIT within MANAFEST for PERFORMANCE PREDICTION (Panel C/D)
# Using 2-chain sigma_C BLOSUM-sqrt (HLA pseudoseq + peptide)
#
# Reason: im_val→MANAFEST is concept drift (benchmark peptides vs clinical
# neoantigens), not distribution shift. P(y|p) differs between the two domains,
# violating PAPE's covariate shift assumption. All 3 methods fail equally
# (AUROC err ~0.28) with im_val as cal. Halfsplit within MANAFEST keeps
# cal and test in the same domain → prediction works (AUROC err ~0.015).
#
# Note: RECALIBRATION (panels e, m, n) uses im_val→MANAFEST (true
# retrospective) because recalibration adjusts per-sample scores via
# PPV/NPV curves, which does not require P(y|p) stability.
print("\n" + "=" * 70)
print("3. BigMHC — HLA halfsplit within MANAFEST, 2-chain sigma_C BLOSUM-sqrt")
print("=" * 70)

BM_DATA = os.path.join(INPUT_DIR, 'Data', 'retrospective_bigmhc', 'mendeley_data',
                        'extracted', 'BigMHC Training and Evaluation Data')

bm_df = pd.read_csv(os.path.join(RESULTS, 'bigmhc_retrospective', 'reproduction',
                                   'manafest_with_distances.csv'))
bm_train = pd.read_csv(os.path.join(BM_DATA, 'im_train.csv'))

# Build HLA pseudosequences for 2-chain distance
_ps = pd.read_csv(os.path.join(INPUT_DIR, 'Model', 'BigMHC', 'data', 'pseudoseqs.csv'))
_pos_cols = [c for c in _ps.columns if c != 'mhc']
_positions = sorted(set(int(c.split('_')[0]) for c in _pos_cols))
def _get_pseq(row):
    seq = []
    for pos in _positions:
        for c in [cc for cc in _pos_cols if cc.startswith(f'{pos}_')]:
            if row[c] == 1:
                aa = c.split('_')[1]
                if aa != 'X': seq.append(aa)
                break
    return ''.join(seq)
_ps_map = {row['mhc']: _get_pseq(row) for _, row in _ps.iterrows()}
bm_df['hla_pseq'] = bm_df['mhc'].map(_ps_map)
bm_train['hla_pseq'] = bm_train['mhc'].map(_ps_map)

# Compute 2-chain sigma_C BLOSUM-sqrt distances for ALL MANAFEST samples
from calipper.pluggable_distance import compute_s2dd_pluggable, make_sw_blosum62_similarity
_sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)
_chain_cols_bm = ['hla_pseq', 'pep']
bm_d_all, _ = compute_s2dd_pluggable(
    bm_df, bm_train, _chain_cols_bm, 'auto',
    similarity_fn=_sw_sim, K=50, transform='sqrt', return_stats=True, verbose=False)

bm_y = bm_df['label'].values[:len(bm_d_all)].astype(int)
bm_p = bm_df['BigMHC_IM'].values[:len(bm_d_all)].astype(float)

# HLA halfsplit (seed=42)
np.random.seed(42)
all_hlas_bm = bm_df['mhc'].unique()
hla_shuf_bm = np.random.permutation(all_hlas_bm)
cal_hla_bm = set(hla_shuf_bm[:len(hla_shuf_bm) // 2])
cal_mask_bm = bm_df['mhc'].isin(cal_hla_bm).values[:len(bm_d_all)]
test_mask_bm = ~cal_mask_bm

cal_y_bm = bm_y[cal_mask_bm]
cal_p_bm = bm_p[cal_mask_bm]
test_y_bm = bm_y[test_mask_bm]
test_p_bm = bm_p[test_mask_bm]
test_d_bm = bm_d_all[test_mask_bm]

print(f'  Cal: n={cal_mask_bm.sum()}, pos={cal_y_bm.sum()}, HLA={len(cal_hla_bm)}')
print(f'  Test: n={test_mask_bm.sum()}, pos={test_y_bm.sum()}')

# Per-HLA cal_data: most entries have < 30 samples, so distance binning
# is skipped → predict_metric returns DRE base estimate without vbias.
# This is CORRECT for BigMHC: sigma_C saturates on HLA (99.85%), causing
# 5/8 distance bins to collapse to the same value. Pooled vbias adds
# +0.135 wrong correction (R²=0.215). Per-HLA naturally avoids this.
cal_data_bm = {}
for hla in cal_hla_bm:
    sub_mask = (bm_df['mhc'] == hla).values[:len(bm_d_all)]
    if sub_mask.sum() < 2:
        continue
    cal_data_bm[hla] = (bm_y[sub_mask], bm_p[sub_mask], bm_d_all[sub_mask])

for metric in METRICS:
    actual = safe_metric(metric, test_y_bm, test_p_bm)
    result = predict_metric(cal_data_bm, test_p_bm, test_d_bm, metrics=[metric])
    s2dd_pred = result['estimated'][metric]
    pape_pred = pape_predict_2d(cal_y_bm, cal_p_bm, bm_d_all[cal_mask_bm], test_p_bm, test_d_bm, metric)
    mcbpe_pred = mcbpe_predict_no_distance(cal_y_bm, cal_p_bm, test_p_bm, metric)
    add_result('BigMHC', metric, actual, s2dd_pred, pape_pred, mcbpe_pred)


# ── 4. XBCR-net ──────────────────────────────────────────────────
# TRUE CROSS-DATASET: Panel1 WT(cal) → Panel2 Omicron(test) for ALL 3 methods
# Aligned with compute_fig6_recal_data.py: same cal/test as recalibration panels
# Panel 1 = published XBCR-net WT test set (1293 samples)
# Panel 2 = Omicron-era mAbs (21 samples, independent experimental validation)
print("\n" + "=" * 70)
print("4. XBCR-net — Panel1 WT(cal) → Panel2 Omicron(test), ALL methods")
print("=" * 70)

p1_df = pd.read_csv(os.path.join(RESULTS, 'xbcr_retrospective', 'reproduction',
                                   'test_predictions_original.csv'))
p2_df = pd.read_csv(os.path.join(RESULTS, 'xbcr_retrospective', 'mab_recalibration',
                                   'panel2_omicron_results_3chain_clean.csv'))
p2_df = p2_df.dropna(subset=['antibody_name'])
p2_df = p2_df[p2_df['antibody_name'].str.len() > 0]

# Cal: Panel 1 WT (labels + predictions + distances)
cal_y_xb = p1_df['rbd'].values.astype(int)
cal_p_xb = p1_df['pred_prob'].values.astype(float)
cal_d_xb = np.load(os.path.join(RESULTS, 'xbcr_retrospective',
                                  'distance_cache_panel1.npz'))['distances'][:len(cal_y_xb)]

# Test: Panel 2 Omicron (labels + predictions + distances)
test_y_xb = p2_df['gt_binds'].values.astype(int)
test_p_xb = p2_df['prediction_score'].values.astype(float)
test_d_xb = p2_df['s2dd_distance'].values.astype(float)

print(f'  Panel 1 cal: n={len(cal_y_xb)}, pos={cal_y_xb.sum()}')
print(f'  Panel 2 test: n={len(test_y_xb)}, pos={test_y_xb.sum()} (AUROC={safe_metric("aucroc", test_y_xb, test_p_xb):.3f})')

cal_data_xb = {'panel1': (cal_y_xb, cal_p_xb, cal_d_xb)}

for metric in METRICS:
    actual = safe_metric(metric, test_y_xb, test_p_xb)
    result = predict_metric(cal_data_xb, test_p_xb, test_d_xb, metrics=[metric])
    s2dd_pred = result['estimated'][metric]
    pape_pred = pape_predict_2d(cal_y_xb, cal_p_xb, cal_d_xb, test_p_xb, test_d_xb, metric)
    mcbpe_pred = mcbpe_predict_no_distance(cal_y_xb, cal_p_xb, test_p_xb, metric)
    add_result('XBCR-net', metric, actual, s2dd_pred, pape_pred, mcbpe_pred)


# ── 5. PanPep ────────────────────────────────────────────────────
# ZERO-SHOT peptide halfsplit: 491 unseen peptides (0 overlap with training).
# Majority test (25 seen peptides) has 100% training overlap → not a true retrospective.
# Cal pooled into one entry (per-peptide has ~3.4 samples → all skipped by min_samples=30).
# Train anchor from majority test (seen peptides, training performance proxy).
print("\n" + "=" * 70)
print("5. PanPep — zero-shot peptide halfsplit(cal→test), BLOSUM-sqrt, pooled+anchor")
print("=" * 70)

pp_zs_pos = pd.read_csv(os.path.join(RESULTS, 'panpep_retrospective', 'reproduction',
                                       'zeroshot_test_predictions.csv'))
pp_zs_neg = pd.read_csv(os.path.join(RESULTS, 'panpep_retrospective', 'reproduction',
                                       'zeroshot_neg_predictions.csv'))
pp_pred = pd.concat([pp_zs_pos, pp_zs_neg], ignore_index=True)
pp_dist_blosum = np.load(os.path.join(RESULTS, 'panpep_retrospective', 'blosum_sqrt',
                                        'zeroshot_test_blosumsqrt_dist.npy'))

pp_y = pp_pred['label'].values.astype(int)
pp_p = pp_pred['prediction'].values.astype(float)
n_pp = min(len(pp_y), len(pp_dist_blosum))
pp_y, pp_p = pp_y[:n_pp], pp_p[:n_pp]
pp_d = pp_dist_blosum[:n_pp]
pp_peptides = pp_pred['peptide'].values[:n_pp]

# Peptide halfsplit — MUST match compute_fig6_recal_data.py exactly
np.random.seed(42)
peptides_sorted = sorted(pd.unique(pp_pred['peptide']))
pep_shuf = np.random.permutation(peptides_sorted)
cal_pep = set(pep_shuf[:len(pep_shuf) // 2])
cal_mask = np.array([p in cal_pep for p in pp_peptides])
test_mask = ~cal_mask

cal_y_pp, cal_p_pp, cal_d_pp = pp_y[cal_mask], pp_p[cal_mask], pp_d[cal_mask]
test_y_pp, test_p_pp, test_d_pp = pp_y[test_mask], pp_p[test_mask], pp_d[test_mask]

print(f'  Cal: n={cal_mask.sum()}, pos={cal_y_pp.sum()}, peptides={len(cal_pep)}')
print(f'  Test: n={test_mask.sum()}, pos={test_y_pp.sum()}')

# Pooled cal (NOT per-peptide) + train anchor from majority test
cal_data_pp = {'zeroshot_cal': (cal_y_pp, cal_p_pp, cal_d_pp)}

pp_maj = pd.read_csv(os.path.join(RESULTS, 'panpep_retrospective', 'reproduction',
                                    'majority_test_predictions.csv'))
pp_maj_y = pp_maj['label'].values.astype(int)
pp_maj_p = pp_maj['prediction'].values.astype(float)
pp_train_anchor = {
    'metrics': {'aucroc': safe_metric('aucroc', pp_maj_y, pp_maj_p),
                'ap': safe_metric('ap', pp_maj_y, pp_maj_p)},
    'mp': float(pp_maj_p.mean()), 'distance': 0.0,
}

for metric in METRICS:
    actual = safe_metric(metric, test_y_pp, test_p_pp)
    result = predict_metric(cal_data_pp, test_p_pp, test_d_pp, metrics=[metric],
                             train_anchor=pp_train_anchor)
    s2dd_pred = result['estimated'][metric]
    pape_pred = pape_predict_2d(cal_y_pp, cal_p_pp, cal_d_pp, test_p_pp, test_d_pp, metric)
    mcbpe_pred = mcbpe_predict_no_distance(cal_y_pp, cal_p_pp, test_p_pp, metric)
    add_result('PanPep', metric, actual, s2dd_pred, pape_pred, mcbpe_pred)


# ═══════════════════════════════════════════════════════════════════
# Save outputs
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Saving outputs")
print("=" * 70)

panel_c_df = pd.DataFrame(panel_c_rows)
panel_c_path = os.path.join(OUT_DIR, 'fig6_panel_c_predictions.csv')
panel_c_df.to_csv(panel_c_path, index=False)
print(f'  Panel C: {panel_c_path} ({len(panel_c_df)} rows)')
print(panel_c_df.to_string(index=False))

print()

panel_d_df = pd.DataFrame(panel_d_rows)
panel_d_path = os.path.join(OUT_DIR, 'fig6_prediction_3method.csv')
panel_d_df.to_csv(panel_d_path, index=False)
print(f'  Panel D: {panel_d_path} ({len(panel_d_df)} rows)')
print(panel_d_df.to_string(index=False))

print("\nDone.")
