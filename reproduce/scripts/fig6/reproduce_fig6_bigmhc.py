# AUTO-GENERATED WRAPPER — BigMHC only
# (Removed __file__ override hack; original wrapper for compute_fig6_recal_data.py with BigMHC config)

import os, sys, time
import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR  # also adds CaliPPer/ to sys.path

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


OUT_DIR = os.path.join(CACHE_DIR, 'recal_data_reproduce')
os.makedirs(OUT_DIR, exist_ok=True)

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


print('=== BigMHC done; wrapper exit ===')
