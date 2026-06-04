#!/usr/bin/env python3
"""Generate PPV/NPV vs distance curves per model.

Outputs: fig5_ppv_npv_{tcr_ct|bcr_ct}_{model}.pdf
Uses v2.7 adaptive threshold for PPV/NPV computation.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from calipper.core import adaptive_n_bins

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
N_BINS = 8
PW, PH = 3.2, 2.8


def save(fig, name):
    fig.savefig(os.path.join(PANEL_DIR, name + '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PANEL_DIR, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


def compute_ppv_npv(y, p, d, n_bins, theta):
    si = np.argsort(d); bs = len(si) // n_bins
    centers, ppvs, npvs = [], [], []
    for i in range(n_bins):
        s = i * bs; e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]; yi = y[idx]; pi = p[idx]
        pp = pi >= theta
        tp = int((pp & (yi == 1)).sum()); fp = int((pp & (yi == 0)).sum())
        tn = int(((~pp) & (yi == 0)).sum()); fn = int(((~pp) & (yi == 1)).sum())
        centers.append(d[idx].mean())
        ppvs.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
        npvs.append(tn / (tn + fn) if tn + fn > 0 else np.nan)
    return np.array(centers), np.array(ppvs), np.array(npvs)


def plot_ppv_npv(test_sets, title, out_name):
    """Plot PPV/NPV curves for multiple test sets on one panel."""
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    colors_ts = {'seen_test': '#2ecc71', 'unseen_fold34': '#e74c3c',
                 'v3_combined': '#3498db', 'v4_combined': '#9b59b6',
                 'sars': '#e74c3c', 'flu': '#3498db'}
    labels_ts = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen',
                 'v3_combined': 'v3', 'v4_combined': 'v4',
                 'sars': 'SARS', 'flu': 'Flu'}

    for ts_name, (centers, ppvs, npvs) in test_sets.items():
        c = colors_ts.get(ts_name, '#888')
        lb = labels_ts.get(ts_name, ts_name)
        valid_ppv = ~np.isnan(ppvs)
        valid_npv = ~np.isnan(npvs)
        if valid_ppv.any():
            ax.plot(centers[valid_ppv], ppvs[valid_ppv], 'o-', color=c, markersize=3,
                    linewidth=1.2, label=f'PPV ({lb})')
        if valid_npv.any():
            ax.plot(centers[valid_npv], npvs[valid_npv], 's--', color=c, markersize=3,
                    linewidth=1.0, alpha=0.7, label=f'NPV ({lb})')

    ax.set_xlabel('S2DD distance', fontsize=9)
    ax.set_ylabel('PPV / NPV', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5, loc='best', ncol=2)
    save(fig, out_name)


# ═══════════════════════════════════════════
# TCR CT
# ═══════════════════════════════════════════
print("=== TCR CT PPV/NPV curves ===")
for model in TCR_MODELS:
    display = MODEL_DISPLAY.get(model, model)
    test_sets = {}
    for ts in ['seen_test', 'unseen_fold34']:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        y = df[lc].values[:n].astype(int)
        p = df[pc].values[:n].astype(float)
        d = d[:n].astype(float)
        prev = y.mean()
        theta = max(2 * prev - 1, min(2 * prev, 0.5))
        centers, ppvs, npvs = compute_ppv_npv(y, p, d, N_BINS, theta)
        test_sets[ts] = (centers, ppvs, npvs)

    if test_sets:
        plot_ppv_npv(test_sets, f'PPV/NPV: {display} (TCR CT)',
                     f'fig5_ppv_npv_tcr_ct_{model}')

# ═══════════════════════════════════════════
# BCR CT (fold4 cal + external, split by domain)
# ═══════════════════════════════════════════
print("\n=== BCR CT PPV/NPV curves ===")
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
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
    parts = [cal]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)

    test_sets = {}
    for domain in ['sars', 'flu']:
        dom_df = pooled[pooled['data_source'] == domain]
        y = dom_df['rbd'].values.astype(int)
        p = dom_df['pred_prob'].values.astype(float)
        d = dom_df['distance'].values.astype(float)
        if len(y) < 50:
            continue
        prev = y.mean()
        theta = max(2 * prev - 1, min(2 * prev, 0.5))
        centers, ppvs, npvs = compute_ppv_npv(y, p, d, N_BINS, theta)
        test_sets[domain] = (centers, ppvs, npvs)

    if test_sets:
        plot_ppv_npv(test_sets, f'PPV/NPV: {display} (BCR CT)',
                     f'fig5_ppv_npv_bcr_ct_{model}')


print("\n=== All PPV/NPV panels generated ===")
