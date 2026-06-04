#!/usr/bin/env python3
"""Fig 2 panels o-p: Cross-model consistency for CV (AP and AUROC).

Horizontal boxplots showing per-fold |r| distribution for each model.
TCR CV: 5 models × 5 folds. BCR CV: 5 models × 5 folds.
Two groups (TCR blue, BCR orange) with legend.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style
from calipper.general_evaluator import safe_metric

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig2_cache')
BCR_CACHE = os.path.join(RESULTS, 'fig2_bcr_cache')
N_BINS = 8

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DISPLAY = {'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
               'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT'}

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_DISPLAY = {'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
               'mint': 'MINT', 'rleaai': 'RLEAAI'}

TCR_COLOR = '#1f77b4'
BCR_COLOR = '#ff7f0e'


def load_tcr_cv_bins(model, fold, metric_key):
    """Load binned data for TCR CV from cached npz or compute from predictions."""
    # Try bins cache first
    bins_path = os.path.join(BCR_CACHE, f'tcr_cv_{model}_fold{fold}_bins.npz')
    if not os.path.exists(bins_path):
        bins_path = None

    # Load from predictions + distances
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    if not os.path.exists(test_path):
        return None
    parts = [pd.read_csv(test_path)]
    for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fold_dir, vname)
        if os.path.exists(vp):
            parts.append(pd.read_csv(vp))
            break
    te = pd.concat(parts, ignore_index=True)
    lc = 'binder' if 'binder' in te.columns else 'y_true'
    pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
    y = te[lc].values.astype(int)
    p = te[pc].values.astype(float)

    dist_path = os.path.join(CACHE, f'{model}_cv_fold{fold}_combined_dist.npy')
    if not os.path.exists(dist_path):
        dist_path = os.path.join(CACHE, f'{model}_cv_fold{fold}_dist.npy')
    if not os.path.exists(dist_path):
        return None
    d = np.load(dist_path)
    n = min(len(d), len(y))
    y, p, d = y[:n], p[:n], d[:n]

    si = np.argsort(d)
    bs = len(si) // N_BINS
    bv, bd = [], []
    metric_name = 'ap' if metric_key == 'by_ap' else 'aucroc'
    for i in range(N_BINS):
        s, e = i * bs, (len(si) if i == N_BINS - 1 else (i + 1) * bs)
        idx = si[s:e]
        bv.append(safe_metric(metric_name, y[idx], p[idx]))
        bd.append(d[idx].mean())
    v, dd = np.array(bv), np.array(bd)
    valid = ~np.isnan(v)
    if valid.sum() >= 4:
        r, _ = pearsonr(dd[valid], v[valid])
        return r
    return None


def load_bcr_cv_bins(model, fold, metric_key):
    """Load binned correlations for BCR CV."""
    bins_path = os.path.join(BCR_CACHE, f'bcr_cv_{BCR_DISPLAY[model].replace("-", "_").lower()}_fold{fold}_bins.npz')
    # Try alternate naming
    for name_variant in [model, BCR_DISPLAY[model].replace('-', '_').lower(),
                          f'{BCR_DISPLAY[model].replace("-", "_")}']:
        bp = os.path.join(BCR_CACHE, f'bcr_cv_{name_variant}_fold{fold}_bins.npz')
        if os.path.exists(bp):
            bins_path = bp
            break

    if os.path.exists(bins_path):
        data = np.load(bins_path, allow_pickle=True)
        bx = data['bx']
        by = data[metric_key]
        valid = ~np.isnan(by)
        if valid.sum() >= 4:
            r, _ = pearsonr(bx[valid], by[valid])
            return r
    return None


def plot_consistency(metric_key, metric_label, out_name):
    """Plot horizontal boxplot of per-fold |r| for each model, TCR + BCR."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    all_labels = []
    all_data = []
    all_colors = []
    y_positions = []
    y_idx = 0

    # TCR models
    for model in TCR_MODELS:
        fold_rs = []
        for fold in range(5):
            r = load_tcr_cv_bins(model, fold, metric_key)
            if r is not None:
                fold_rs.append(abs(r))
        all_labels.append(TCR_DISPLAY[model])
        all_data.append(fold_rs if fold_rs else [0])
        all_colors.append(TCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

    y_idx += 0.5  # gap between TCR and BCR

    # BCR models
    for model in BCR_MODELS:
        fold_rs = []
        for fold in range(5):
            r = load_bcr_cv_bins(model, fold, metric_key)
            if r is not None:
                fold_rs.append(abs(r))
        all_labels.append(BCR_DISPLAY[model])
        all_data.append(fold_rs if fold_rs else [0])
        all_colors.append(BCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

    # Horizontal boxplots
    for i, (pos, data, color) in enumerate(zip(y_positions, all_data, all_colors)):
        bp = ax.boxplot([data], positions=[pos], widths=0.6, vert=False,
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(1.2)
        # Strip dots
        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(data))
        ax.scatter(data, np.full(len(data), pos) + jitter,
                   c=color, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel(f'|Pearson r| ({metric_label} degradation)', fontsize=9)
    ax.set_title(f'Cross-model consistency\n(CV, {metric_label})', fontweight='bold', fontsize=12)
    ax.set_xlim(0.0, 1.05)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend([Patch(facecolor=TCR_COLOR, alpha=0.5),
               Patch(facecolor=BCR_COLOR, alpha=0.5)],
              ['TCR', 'BCR'], fontsize=8, loc='upper left', framealpha=0.8)

    out_dir = os.path.join(PANEL_DIR, 'lev-logtransf')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, out_name)
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}.png')


# Check BCR cache naming
print("BCR cache files:")
import glob
bcr_bins = glob.glob(os.path.join(BCR_CACHE, 'bcr_cv_*_bins.npz'))
for b in sorted(bcr_bins)[:10]:
    print(f"  {os.path.basename(b)}")

plot_consistency('by_ap', 'AP', 'fig2_cross_model_consistency_ap')
plot_consistency('by_auroc', 'AUROC', 'fig2_cross_model_consistency_auroc')

print("\nDone.")
