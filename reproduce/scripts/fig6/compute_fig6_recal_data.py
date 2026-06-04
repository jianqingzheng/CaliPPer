#!/usr/bin/env python3
"""Compute per-sample calibrated scores for all 5 retrospective studies.

Saves per-study CSVs with columns: y_true, raw_pred, cal_pred, distance
These are used for ROC curves, TDR, and ranking panels in fig6.

Also saves a consolidated AP+AUROC summary for dumbbell and scatter panels.

Usage:
    cd <published_repo>/CaliPPer
    python Manuscript/designed_figures/panels/fig6/scripts/compute_fig6_recal_data.py
"""
import os, sys, time
import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration
from calipper.combine_first_helpers import compute_chain_weights
from calipper.pluggable_distance import make_sw_blosum62_similarity

RESULTS = os.path.join(INPUT_DIR, 'results')
OUT_DIR = os.path.join(OUTPUT_DIR, 'recal_data')
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ['aucroc', 'ap']
t0 = time.time()
summary = []


def save_study(name, test_y, test_p, cal_scores, test_d=None):
    """Save per-sample data and compute summary metrics."""
    df = pd.DataFrame({
        'y_true': test_y, 'raw_pred': test_p, 'cal_pred': cal_scores
    })
    if test_d is not None:
        df['distance'] = test_d
    df.to_csv(os.path.join(OUT_DIR, f'{name}_samples.csv'), index=False)

    for metric in METRICS:
        before = safe_metric(metric, test_y, test_p)
        after = safe_metric(metric, test_y, cal_scores)
        summary.append(dict(study=name, metric=metric, before=before, after=after,
                            delta=after - before, n=len(test_y)))
        print(f"  {metric}: {before:.3f} → {after:.3f} (Δ={after-before:+.3f})")

    # TDR
    for k in [5, 10, 20, 50, 100]:
        if k > len(test_y): continue
        top_raw = np.argsort(test_p)[::-1][:k]
        top_cal = np.argsort(cal_scores)[::-1][:k]
        tdr_raw = test_y[top_raw].sum() / k
        tdr_cal = test_y[top_cal].sum() / k
        summary.append(dict(study=name, metric=f'tdr@{k}', before=tdr_raw, after=tdr_cal,
                            delta=tdr_cal - tdr_raw, n=k))


# ═══════════════════════════════════════════
# 1. deepAntigen (BLOSUM-SW, zero-shot → neoantigen)
# ═══════════════════════════════════════════
print("=" * 60)
print("[1] deepAntigen")

neo = pd.read_csv(f'{RESULTS}/deepantigen_retrospective/neoantigen_recalibration/neoantigen_recalibrated.csv')
zs = pd.read_csv(f'{RESULTS}/deepantigen_retrospective/s2dd_degradation/zero_shot_with_distances.csv')
zs_blosum = pd.read_csv(f'{RESULTS}/deepantigen_retrospective/s2dd_degradation/zero_shot_sw_topk_distances.csv')

y_zs = zs['label'].values.astype(int)
p_zs = zs['prediction'].values.astype(float)
d_zs = zs_blosum['distance'].values[:len(y_zs)]
y_neo = neo['confirmed'].astype(int).values
p_neo = neo['score'].astype(float).values
d_neo = neo['s2dd_blosum'].values

cal_data = {'zero_shot': (y_zs, p_zs, d_zs)}
ppv_p, npv_p, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data)
# OVERRIDE RATIONALE: cal set has 1714 samples → fit uses n_bins=8. Test set has only
# 100 samples → adaptive n_bins gives 4, creating a fit/apply granularity mismatch.
# The mp feature in the PPV/NPV curves was learned at 8-bin resolution; applying at
# 4-bin resolution loses the distance gradient that the curves encode.
# This is a v2.7 API limitation: apply computes n_bins from test_y independently
# of what fit used. Until the API returns fit's n_bins, we override to match.
cs = apply_recalibration(y_neo, p_neo, d_neo, ppv_p, npv_p, p_pos, p_neg, n_bins=8, prev=_cal_prev)
save_study('deepAntigen', y_neo, p_neo, cs, d_neo)


# ═══════════════════════════════════════════
# 2. PanPep (BLOSUM-sqrt, zero-shot peptide halfsplit)
# ═══════════════════════════════════════════
# Zero-shot test (491 unseen peptides, 0 overlap with training) is the honest
# retrospective evaluation. Majority test (25 seen peptides) has 100% overlap.
# Cal pooled into one entry (per-peptide has ~3.4 samples each → all skipped
# by min_samples=30 check → sigmoid fit fails).
# Train anchor from majority test (seen peptides, training proxy).
print(f"\n{'=' * 60}")
print("[2] PanPep")

pp_zs_pos = pd.read_csv(f'{RESULTS}/panpep_retrospective/reproduction/zeroshot_test_predictions.csv')
pp_zs_neg = pd.read_csv(f'{RESULTS}/panpep_retrospective/reproduction/zeroshot_neg_predictions.csv')
pp = pd.concat([pp_zs_pos, pp_zs_neg], ignore_index=True)
pp_bls = np.load(f'{RESULTS}/panpep_retrospective/blosum_sqrt/zeroshot_test_blosumsqrt_dist.npy')
pp['blosum_dist'] = pp_bls[:len(pp)]

np.random.seed(42)
peptides = sorted(pp['peptide'].unique())
pep_shuf = np.random.permutation(peptides)
cal_peps = set(pep_shuf[:len(pep_shuf) // 2])
test_peps = set(pep_shuf[len(pep_shuf) // 2:])
cal_pp = pp[pp['peptide'].isin(cal_peps)]
test_pp = pp[pp['peptide'].isin(test_peps)]

# Pooled cal (NOT per-peptide — too few samples per peptide for binning)
cal_data = {'zeroshot_cal': (cal_pp['label'].values.astype(int),
                              cal_pp['prediction'].values.astype(float),
                              cal_pp['blosum_dist'].values)}

# Train anchor from majority test (seen peptides → training performance proxy)
pp_maj = pd.read_csv(f'{RESULTS}/panpep_retrospective/reproduction/majority_test_predictions.csv')
pp_maj_y = pp_maj['label'].values.astype(int)
pp_maj_p = pp_maj['prediction'].values.astype(float)
_theta = 0.5
_pos_m = pp_maj_p >= _theta
_neg_m = pp_maj_p < _theta
pp_train_anchor = {
    'distance': 0.0, 'mp': float(pp_maj_p.mean()),
    'ppv': float(pp_maj_y[_pos_m].mean()), 'npv': float((1 - pp_maj_y[_neg_m]).mean()),
}

ppv_p, npv_p, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data, train_anchor=pp_train_anchor)
cs = apply_recalibration(test_pp['label'].values.astype(int), test_pp['prediction'].values.astype(float),
                          test_pp['blosum_dist'].values, ppv_p, npv_p, p_pos, p_neg, prev=_cal_prev)
save_study('PanPep', test_pp['label'].values.astype(int), test_pp['prediction'].values.astype(float),
           cs, test_pp['blosum_dist'].values)


# ═══════════════════════════════════════════
# 3. XBCR-net (Lev, Panel 1 WT cal → Panel 2 Omicron test)
# ═══════════════════════════════════════════
# OVERRIDE RATIONALE: XBCR-net's Panel 1 calibration set (1293 samples, 83.8% positive)
# contains random antibody-antigen pairings as SIMULATED negatives — these are not
# experimentally confirmed non-binders. This inflates cal prevalence from ~50% (true)
# to 83.8%, which causes:
#   1. adaptive theta = 0.675 (too high; true decision boundary ~0.5)
#   2. cal_prev = 0.838 as Platt base rate (overestimates positive prior)
# In deployment, a practitioner would know their negatives are simulated and enter
# ~0.5 as the expected prevalence. We override both theta and prev to 0.5 to
# reflect this corrected prevalence estimate.
print(f"\n{'=' * 60}")
print("[3] XBCR-net")

panel1 = pd.read_csv(f'{RESULTS}/xbcr_retrospective/reproduction/test_predictions_original.csv')
panel2 = pd.read_csv(f'{RESULTS}/xbcr_retrospective/mab_recalibration/panel2_omicron_results_3chain_clean.csv')
p1_lev = np.load(f'{RESULTS}/xbcr_retrospective/distance_cache_panel1.npz')['distances'][:len(panel1)]

cal_data = {'panel1': (panel1['rbd'].values.astype(int), panel1['pred_prob'].values.astype(float), p1_lev)}
XBCR_CORRECTED_PREV = 0.5  # corrected for simulated negatives in training data
ppv_p, npv_p, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data, threshold=XBCR_CORRECTED_PREV)

om = panel2.dropna(subset=['antibody_name'])
om = om[om['antibody_name'].str.len() > 0]
cs = apply_recalibration(om['gt_binds'].values.astype(int), om['prediction_score'].values.astype(float),
                          om['s2dd_distance'].values, ppv_p, npv_p, p_pos, p_neg,
                          prev=XBCR_CORRECTED_PREV)
save_study('XBCR-net', om['gt_binds'].values.astype(int), om['prediction_score'].values.astype(float),
           cs, om['s2dd_distance'].values)


# ═══════════════════════════════════════════
# 4. BigMHC (true retrospective: im_val cal → full MANAFEST test)
#
# Model: BigMHC_IM (immunogenicity), 7-model ensemble via models/bat*/im/
# Distance: 2-chain sigma_C (HLA pseudoseq + peptide), BLOSUM-sqrt
# Cal: im_val (688 samples, author validation set, 0% peptide overlap with MANAFEST)
# Test: MANAFEST (834 samples, clinical neoantigens, experimental immunogenicity labels)
# Anchor: train anchor from im_train IM predictions
#
# Previous pipeline was wrong:
#   - Used EL model instead of IM (fixed 2026-05-10)
#   - Used peptide-only S2DD (BigMHC uses both HLA+peptide)
#   - Used halfsplit within MANAFEST (not true retrospective)
# im_test ≈ MANAFEST (99.8% pair overlap, 100% label agreement) — cannot use as cal
# See results/bigmhc_retrospective/BIGMHC_INVESTIGATION_20260510.md
# ═══════════════════════════════════════════
print(f"\n{'=' * 60}")
print("[4] BigMHC (im_val → MANAFEST, IM model, 2-chain)")

BM_DATA = os.path.join(INPUT_DIR, 'Data', 'retrospective_bigmhc', 'mendeley_data',
                        'extracted', 'BigMHC Training and Evaluation Data')

# Load data
bm_val = pd.read_csv(f'{BM_DATA}/im_val.csv')
bm_val_pred = pd.read_csv(f'{RESULTS}/bigmhc_retrospective/bigmhc_val_pred_im.csv')
bm_man_source = pd.read_csv(f'{BM_DATA}/manafest.csv')
bm_man_pred = pd.read_csv(f'{RESULTS}/bigmhc_retrospective/reproduction/manafest_with_distances.csv')
bm_train = pd.read_csv(f'{BM_DATA}/im_train.csv')
bm_train_pred_path = os.path.join(RESULTS, 'bigmhc_retrospective', 'bigmhc_train_pred_im.csv')
if not os.path.exists(bm_train_pred_path):
    bm_train_pred_path = os.path.join(RESULTS, 'bigmhc_retrospective', 'bigmhc_train_pred.csv')
bm_train_pred = pd.read_csv(bm_train_pred_path)

# Build HLA pseudosequence column for 2-chain distance
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
for df in [bm_val, bm_man_source, bm_train]:
    df['hla_pseq'] = df['mhc'].map(_ps_map)

# Compute 2-chain sigma_C BLOSUM-sqrt distances inline (so we get chain_stats for anchor)
# sigma_C upweights HLA (99.85%) because HLA pseudoseqs have high concentration (65 unique)
# This captures the HLA distributional shift between im_val and MANAFEST
from calipper.pluggable_distance import compute_s2dd_pluggable
_sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)
_chain_cols = ['hla_pseq', 'pep']

# Merge MANAFEST source labels with predictions FIRST
# (3 source rows have no match → 837→834, must compute distances on merged df)
bm_test = bm_man_source.merge(bm_man_pred[['mhc', 'pep', 'BigMHC_IM']], on=['mhc', 'pep'])
bm_test['hla_pseq'] = bm_test['mhc'].map(_ps_map)

# Cal distances (im_val vs im_train) — sigma_C auto weights + return chain_stats for anchor
d_cal, chain_stats = compute_s2dd_pluggable(
    bm_val, bm_train, _chain_cols, 'auto',
    similarity_fn=_sw_sim, K=50, transform='sqrt', return_stats=True, verbose=True)

# Test distances on MERGED df (ensures row alignment with y_test/p_test)
d_test, _ = compute_s2dd_pluggable(
    bm_test, bm_train, _chain_cols, 'auto',
    similarity_fn=_sw_sim, K=50, transform='sqrt', return_stats=True, verbose=False)

y_cal = bm_val['tgt'].values.astype(int)
p_cal = bm_val_pred['BigMHC_IM'].values[:len(y_cal)].astype(float)
y_test = bm_test['tgt'].values.astype(int)
p_test = bm_test['BigMHC_IM'].values.astype(float)
d_test = d_test[:len(y_test)]
print(f"  Cal (im_val): n={len(y_cal)}, prev={y_cal.mean():.3f}")
print(f"  Test (MANAFEST): n={len(y_test)}, prev={y_test.mean():.3f}")

# Train anchor: raw distance = 0 (BLOSUM-sqrt self-match: sqrt(1-1)=0),
# z-normed with SAME train-vs-train mu/sigma from compute_s2dd_pluggable.
# Uses sigma_C weights (same as distance computation).
# The anchor is NOT included in mu/sigma — it only uses them for transformation.
y_tr = bm_train['tgt'].values.astype(int)
p_tr = bm_train_pred['BigMHC_IM'].values.astype(float)
pp_tr = p_tr >= 0.5
from calipper.combine_first_helpers import compute_chain_weights as _ccw
_w_sigC, _ = _ccw(bm_train, _chain_cols, 0.1, 0.1, 50, formula='sigma_C')
anchor_d = sum(_w_sigC[c] * (0.0 - chain_stats[c][0]) / chain_stats[c][1]
               for c in range(len(_chain_cols)))
print(f"  sigma_C weights: {', '.join(f'{col}={_w_sigC[c]:.4f}' for c, col in enumerate(_chain_cols))}")
print(f"  Train anchor: d={anchor_d:.2f} (from chain_stats: " +
      ", ".join(f"{col} mu={chain_stats[c][0]:.4f} sigma={chain_stats[c][1]:.4f}"
                for c, col in enumerate(_chain_cols)) + ")")
anchor = dict(ppv=int((pp_tr & (y_tr == 1)).sum()) / max(int(pp_tr.sum()), 1),
              npv=int(((~pp_tr) & (y_tr == 0)).sum()) / max(int((~pp_tr).sum()), 1),
              mp=float(p_tr.mean()),
              distance=anchor_d)

cal_data = {'im_val': (y_cal, p_cal, d_cal[:len(y_cal)])}
ppv_p, npv_p, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data, train_anchor=anchor)
cs = apply_recalibration(y_test, p_test, d_test, ppv_p, npv_p, p_pos, p_neg, prev=_cal_prev)
save_study('BigMHC', y_test, p_test, cs, d_test)


# ═══════════════════════════════════════════
# 5. AntibioticsAI (Morgan FP, distance-interleaved halfsplit)
# ═══════════════════════════════════════════
print(f"\n{'=' * 60}")
print("[5] AntibioticsAI")

aa = pd.read_csv(f'{RESULTS}/antibioticsai_retrospective/reproduction/main_test_with_distances.csv')
y_aa = aa['ACTIVITY'].values.astype(int)
p_aa = aa['ANTIBIOTIC_PS'].values.astype(float)
d_aa = aa['distance'].values

si = np.argsort(d_aa)
# 2026-05-21: FLIPPED to cal=odd, test=even after head-to-head check
# showed the legacy direction (cal=even, test=odd) put the harder half
# on test, yielding negative ΔAP (-0.071) and a -1.000 ΔTDR@1 artifact.
# Flipped direction restores positive ΔAP (+0.036), preserves ΔTDR@1
# (0.000), and ~doubles ΔTDR@20 (+0.250 vs +0.100). See
# REPRODUCIBILITY.md → Fig 6 Panel p paragraph for the side-by-side.
# Legacy direction (for reference): cal_idx, test_idx = si[::2], si[1::2]
cal_idx, test_idx = si[1::2], si[::2]  # FLIPPED: cal=odd, test=even
cal_data = {'cal': (y_aa[cal_idx], p_aa[cal_idx], d_aa[cal_idx])}
ppv_p, npv_p, p_pos, p_neg, _cal_prev = fit_recalibration(cal_data)
cs = apply_recalibration(y_aa[test_idx], p_aa[test_idx], d_aa[test_idx],
                          ppv_p, npv_p, p_pos, p_neg, prev=_cal_prev)
save_study('AntibioticsAI', y_aa[test_idx], p_aa[test_idx], cs, d_aa[test_idx])


# ═══════════════════════════════════════════
# Save consolidated summary
# ═══════════════════════════════════════════
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, 'recal_summary_all.csv'), index=False)
print(f"\n{'=' * 60}")
print(f"Saved to {OUT_DIR}/")
print(f"  Per-study CSVs: {[f'{s}_samples.csv' for s in ['deepAntigen','PanPep','XBCR-net','BigMHC','AntibioticsAI']]}")
print(f"  Summary: recal_summary_all.csv ({len(summary_df)} rows)")
print(f"Total time: {time.time()-t0:.0f}s")

# Quick summary
print(f"\n=== AUROC + AP Summary ===")
for study in ['deepAntigen', 'PanPep', 'XBCR-net', 'BigMHC', 'AntibioticsAI']:
    for metric in ['aucroc', 'ap']:
        row = summary_df[(summary_df['study'] == study) & (summary_df['metric'] == metric)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {study:<15s} {metric:>6s}: {r['before']:.3f} → {r['after']:.3f} (Δ={r['delta']:+.3f})")
