#!/usr/bin/env python3
"""Generate ECE comparison panel: model-native vs S2DD-derived, 10 models.

Format: combined dumbbell (TCR top, BCR bottom), matching fig5_combined_recal_dumbbell.

ECE = Expected Calibration Error: mean |predicted probability - actual positive rate| per bin.
Model-native ECE: bin by raw model probability, measure calibration.
S2DD-derived ECE: bin by S2DD distance, use PPV(d) as predicted probability for positives
  and (1-NPV(d)) for negatives, measure calibration.

Uses v2.7 fit_recalibration/apply_recalibration with adaptive defaults.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
_FIG_DIR = os.path.join(FIG_DIR, 'fig5')
os.makedirs(_FIG_DIR, exist_ok=True)
from style_config import (apply_publication_style, MODEL_COLORS, MODEL_DISPLAY,
                           BCR_MODEL_COLORS, BCR_MODEL_DISPLAY)
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE, get_bcr_ct_distance
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

# Per-domain distance: BLOSUM for TCR (≤20 AA), Lev for BCR (≥100 AA)
TCR_DIST_SUFFIX = DIST_SUFFIX['blosum-sqrt']
BCR_DIST_MODE_ACTUAL = BCR_DIST_MODE['lev-log']

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_CAL = ['v3_combined', 'v4_combined']
TCR_TEST = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
MIN_SAMPLES = 30
ECE_BINS = 10


def compute_ece(y, p, n_bins=ECE_BINS):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = p[mask].mean()
        avg_true = y[mask].mean()
        ece += mask.sum() / len(y) * abs(avg_pred - avg_true)
    return ece


def save(fig, name):
    for subdir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(_FIG_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ═══════════════════════════════════════════
# TCR: compute ECE for raw and recalibrated
# ═══════════════════════════════════════════
print("=== Computing ECE for TCR models ===")
tcr_ece = {}  # model -> (ece_raw, ece_recal)

for model in TCR_MODELS:
    display = MODEL_DISPLAY.get(model, model)
    ct = {}
    for ts in TCR_CAL + TCR_TEST:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{TCR_DIST_SUFFIX}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ct[ts] = {'y': df[lc].values[:n].astype(int),
                  'p': df[pc].values[:n].astype(float),
                  'd': d[:n].astype(float)}

    if 'v3_combined' not in ct or 'v4_combined' not in ct:
        continue

    cal_data = {s: (ct[s]['y'], ct[s]['p'], ct[s]['d']) for s in TCR_CAL}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

    # Pool all test sets for ECE
    all_y, all_raw, all_cal = [], [], []
    for ts in TCR_TEST:
        if ts not in ct:
            continue
        test = ct[ts]
        cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                    ppv_p, npv_p, pp, pn)
        all_y.extend(test['y'].tolist())
        all_raw.extend(test['p'].tolist())
        all_cal.extend(cal_s.tolist())

    if all_y:
        ya, ra, ca = np.array(all_y), np.array(all_raw), np.array(all_cal)
        ece_raw = compute_ece(ya, ra)
        ece_cal = compute_ece(ya, ca)
        tcr_ece[model] = (ece_raw, ece_cal)
        pct = (ece_raw - ece_cal) / ece_raw * 100 if ece_raw > 0 else 0
        print(f'  {display}: ECE {ece_raw:.4f} -> {ece_cal:.4f} (-{pct:.0f}%)')


# ═══════════════════════════════════════════
# BCR: compute ECE for raw and recalibrated
# ═══════════════════════════════════════════
print("\n=== Computing ECE for BCR models ===")
bcr_ece = {}

for model in BCR_MODELS:
    sk = BCR_STYLE_KEY[model]
    display = BCR_MODEL_DISPLAY.get(sk, model)
    model_dir = os.path.join(BCR_FOLD4CAL, model)
    if not os.path.isdir(model_dir):
        continue

    cal_path = os.path.join(model_dir, "cal_predictions.csv")


    if not os.path.exists(cal_path):


        print(f"  [BCR] {model}: cal_predictions.csv missing; skipping")


        continue


    cal = pd.read_csv(cal_path)
    if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
        cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
    cal['source'] = 'fold4_test'
    parts = [cal]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
            df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
        df['source'] = ts
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)

    all_y, all_raw, all_cal = [], [], []
    for domain in ['sars', 'flu']:
        domain_df = pooled[pooled['data_source'] == domain]
        variants = domain_df.groupby('variant_seq').size()
        valid = variants[variants >= MIN_SAMPLES].index.tolist()
        for held_v in valid:
            test_mask = domain_df['variant_seq'] == held_v
            cal_mask = ~test_mask
            cal_sub = domain_df[cal_mask]
            test_sub = domain_df[test_mask]
            if len(test_sub) < 10:
                continue
            test_y = test_sub['rbd'].values.astype(int)
            if test_y.sum() == 0 or test_y.sum() == len(test_y):
                continue
            cal_y = cal_sub['rbd'].values.astype(int)
            if cal_y.sum() < 3 or (len(cal_y) - cal_y.sum()) < 3:
                continue
            cal_data_v = {'cal': (cal_y,
                                  cal_sub['pred_prob'].values.astype(float),
                                  cal_sub['distance'].values.astype(float))}
            test_p = test_sub['pred_prob'].values.astype(float)
            test_d = test_sub['distance'].values.astype(float)
            ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_v)
            cal_s = apply_recalibration(test_y, test_p, test_d,
                                        ppv_p, npv_p, pp, pn)
            all_y.extend(test_y.tolist())
            all_raw.extend(test_p.tolist())
            all_cal.extend(cal_s.tolist())

    if all_y:
        ya, ra, ca = np.array(all_y), np.array(all_raw), np.array(all_cal)
        ece_raw = compute_ece(ya, ra)
        ece_cal = compute_ece(ya, ca)
        bcr_ece[model] = (ece_raw, ece_cal)
        pct = (ece_raw - ece_cal) / ece_raw * 100 if ece_raw > 0 else 0
        print(f'  {display}: ECE {ece_raw:.4f} -> {ece_cal:.4f} (-{pct:.0f}%)')


# ═══════════════════════════════════════════
# Plot: combined dumbbell (matching recal dumbbell format)
# ═══════════════════════════════════════════
fig, (ax_tcr, ax_bcr) = plt.subplots(2, 1, figsize=(4.5, 4.5),
                                      gridspec_kw={'hspace': 0.35})


def plot_ece_dumbbell(ax, results, style_key_map, color_map, display_map, title):
    """ECE dumbbell: model-native (open) → S2DD-recalibrated (filled)."""
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1][0] - x[1][1],  # sort by improvement
                          reverse=True)
    yp = np.arange(len(sorted_items))[::-1]
    all_vals = []

    for i, (model, (ece_raw, ece_cal)) in enumerate(sorted_items):
        sk = style_key_map.get(model, model)
        color = color_map.get(sk, '#888')
        pct = (ece_raw - ece_cal) / ece_raw * 100 if ece_raw > 0 else 0

        ax.plot([ece_raw, ece_cal], [yp[i], yp[i]], color=color, linewidth=2.5,
                solid_capstyle='round', alpha=0.6)
        ax.scatter(ece_raw, yp[i], color='white', edgecolor=color, s=40,
                   zorder=5, linewidth=1.0)
        ax.scatter(ece_cal, yp[i], color=color, s=45, zorder=5,
                   edgecolor='white', linewidth=0.5)
        label = f'-{pct:.0f}%' if pct > 0 else f'+{abs(pct):.0f}%'
        ax.text(max(ece_raw, ece_cal) + 0.005, yp[i],
                label, va='center', fontsize=7,
                color=color, fontweight='bold')
        all_vals.extend([ece_raw, ece_cal])

    ax.set_yticks(yp)
    ax.set_yticklabels([display_map.get(style_key_map.get(m, m), m)
                        for m, _ in sorted_items], fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=10, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return all_vals


tcr_sk = {m: m for m in TCR_MODELS}
tcr_vals = plot_ece_dumbbell(ax_tcr, tcr_ece, tcr_sk, MODEL_COLORS,
                              MODEL_DISPLAY, 'TCR-epitope (5 models)')
ax_tcr.set_xticklabels([])

bcr_vals = plot_ece_dumbbell(ax_bcr, bcr_ece, BCR_STYLE_KEY, BCR_MODEL_COLORS,
                              BCR_MODEL_DISPLAY, 'BCR-antigen (5 models)')
ax_bcr.set_xlabel('Expected Calibration Error (ECE)', fontsize=10)

# Shared x limits
all_vals = tcr_vals + bcr_vals
x_max = max(all_vals) + 0.05
ax_tcr.set_xlim(-0.005, x_max)
ax_bcr.set_xlim(-0.005, x_max)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
           markeredgecolor='gray', markersize=5, linewidth=0, label='Model-native'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
           markersize=5, linewidth=0, label='After recalib.'),
]
ax_tcr.legend(handles=legend_elements, loc='lower right', ncol=1,
              fontsize=7, frameon=True, framealpha=0.9)

save(fig, 'fig5_ece_dumbbell_10models')

print(f"\n=== ECE dumbbell generated (TCR: {len(tcr_ece)}, BCR: {len(bcr_ece)} models) ===")
