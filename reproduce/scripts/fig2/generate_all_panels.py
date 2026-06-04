#!/usr/bin/env python3
"""Generate ALL individual panels for Figs 2-6.

Each panel saved as separate PDF+PNG in panels/fig{N}/fig{N}_{label}_{desc}.pdf
All from cached data — no distance recomputation.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ → fig2/ → panels/ → designed_figures/ → Manuscript/ → general_eval/
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import (
    MODEL_COLORS, MODEL_DISPLAY, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY,
    METRIC_COLORS, METRIC_DISPLAY, SOURCE_COLORS, SOURCE_DISPLAY,
    apply_publication_style, DPI, FONT_LABEL, FONT_TICK, FONT_LEGEND,
)
from calipper.general_evaluator import safe_metric
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')
BCR_CACHE = os.path.join(RESULTS, 'fig2_bcr_cache' if DIST_TYPE == 'lev-log' else 'fig2_bcr_cache_blosumsqrt')
FIG34_CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
# Output to panels/fig{N}/{dist_subdir}/
PANEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr_net', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
BCR_CT_SETS = ['A1-A11', 'unseen', 'flu']  # fold4cal pipeline (no BNT162b2/guoyu — 100% training overlap)
TCR_CT_DISPLAY = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen', 'v3_combined': 'v3',
                   'v4_combined': 'v4', 'mcpas': 'McPAS', 'iedb_sars': 'IEDB'}
TCR_CT_COLORS = {'seen_test': '#2ecc71', 'unseen_fold34': '#e74c3c', 'v3_combined': '#3498db',
                  'v4_combined': '#f39c12', 'mcpas': '#9b59b6', 'iedb_sars': '#e67e22'}
BCR_CT_COLORS = {'A1-A11': '#e74c3c', 'unseen': '#f39c12', 'flu': '#3498db'}
FOLD_COLORS = {0: '#a3c4e0', 1: '#ffc68a', 2: '#a3d9a3', 3: '#e8a3a3', 4: '#c4b3d9'}

PANEL_W, PANEL_H = 3.0, 2.5  # inches per panel


def save_panel(fig, fig_num, label, desc):
    """Save panel to panels/fig{N}/{dist_subdir}/fig{N}_{desc}.pdf/png."""
    d = os.path.join(PANEL_DIR, f'fig{fig_num}', DIST_SUBDIR[DIST_TYPE])
    os.makedirs(d, exist_ok=True)
    base = f'fig{fig_num}_{desc}'
    fig.savefig(os.path.join(d, base + '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(d, base + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {desc}')


def make_panel():
    """Create a single-panel figure."""
    fig, ax = plt.subplots(1, 1, figsize=(PANEL_W, PANEL_H))
    return fig, ax


def load_tcr_cv_bins():
    """Load TCR CV binned data — test+val combined (matches original fig2 design)."""
    data = {}
    for model in TCR_MODELS:
        for fold in range(5):
            fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
            test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
            if not os.path.exists(test_path):
                continue
            # Load test + val combined
            parts = [pd.read_csv(test_path)]
            for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
                vp = os.path.join(fold_dir, vname)
                if os.path.exists(vp):
                    parts.append(pd.read_csv(vp))
                    break
            te = pd.concat(parts, ignore_index=True)
            # Use dist_config-aware distance files
            if DIST_TYPE == 'blosum-sqrt':
                dist_path = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_blosumsqrt_dist.npy')
            else:
                dist_path = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_combined_dist.npy')
            if not os.path.exists(dist_path):
                dist_path = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_dist.npy')
            if not os.path.exists(dist_path):
                continue
            d = np.load(dist_path)
            if len(d) != len(te):
                d = d[:len(te)]  # truncate if slight mismatch
            lc = 'binder' if 'binder' in te.columns else 'y_true'
            pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
            si = np.argsort(d); bs = len(si) // 8
            bx, by_auc, by_ap = [], [], []
            for i in range(8):
                s = i * bs; e = len(si) if i == 7 else (i+1)*bs
                idx = si[s:e]
                bx.append(d[idx].mean())
                by_auc.append(safe_metric('aucroc', te[lc].values[idx].astype(int), te[pc].values[idx].astype(float)))
                by_ap.append(safe_metric('ap', te[lc].values[idx].astype(int), te[pc].values[idx].astype(float)))
            data[(model, fold)] = {'bx': np.array(bx), 'by_auroc': np.array(by_auc), 'by_ap': np.array(by_ap)}
    return data


def load_bcr_cv_bins():
    data = {}
    for model in BCR_MODELS:
        for fold in range(5):
            path = os.path.join(BCR_CACHE, f'bcr_cv_{model}_fold{fold}_bins.npz')
            if os.path.exists(path):
                d = np.load(path)
                data[(model, fold)] = {k: d[k] for k in d.files}
    return data


def plot_cross_model_panel(data, models, colors, display, metric_key, ylabel, title):
    """Plot degradation curves. Uses MEAN of per-fold r (not pooled r across folds)."""
    fig, ax = make_panel()
    for model in models:
        fold_curves = []
        for fold in range(5):
            key = (model, fold)
            if key not in data: continue
            d = data[key]
            bx, by = d['bx'], d[metric_key]
            valid = ~np.isnan(by)
            if valid.sum() >= 3:
                r_fold, _ = pearsonr(bx[valid], by[valid])
                fold_curves.append((bx[valid], by[valid], r_fold))
        if not fold_curves: continue
        # Mean of per-fold r values
        mean_r = np.mean([c[2] for c in fold_curves])
        # Interpolate to common x for mean±std band
        x_common = np.mean([c[0] for c in fold_curves], axis=0)
        interp_ys = [np.interp(x_common, c[0], c[1]) for c in fold_curves]
        ys = np.array(interp_ys)
        mean_y = np.nanmean(ys, axis=0); std_y = np.nanstd(ys, axis=0)
        vx = ~np.isnan(mean_y)
        ax.plot(x_common[vx], mean_y[vx], '-', color=colors[model], linewidth=1.5,
                label=f'{display[model]} (r={mean_r:.2f})', alpha=0.9)
        ax.fill_between(x_common[vx], (mean_y-std_y)[vx], (mean_y+std_y)[vx], color=colors[model], alpha=0.1)
    ax.set_xlabel('S2DD distance'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    return fig


# ═══════════════════════════════════════════
print("=== Loading data ===")
tcr_cv = load_tcr_cv_bins()
bcr_cv = load_bcr_cv_bins()
print(f"  TCR CV: {len(tcr_cv)} entries, BCR CV: {len(bcr_cv)} entries")

# BCR CT bins
bcr_ct = {}
for fold in range(5):
    for ts in BCR_CT_SETS:
        for suffix in ['_bins', '_unseenheavy_bins']:
            path = os.path.join(BCR_CACHE, f'bcr_ct_fold{fold}_{ts}{suffix}.npz')
            if os.path.exists(path):
                d = np.load(path)
                key = f'{fold}_{ts}' if 'unseenheavy' not in suffix else f'{fold}_{ts}_uh'
                bcr_ct[key] = {k: d[k] for k in d.files}

bcr_hm = np.load(os.path.join(BCR_CACHE, 'bcr_heatmap_data.npz'))
bcr_summary = np.load(os.path.join(BCR_CACHE, 'bcr_summary_stats.npz'))
bcr_diversity = np.load(os.path.join(BCR_CACHE, 'bcr_diversity_effect.npz'))
print("  All cache loaded\n")


# ═══════════════════════════════════════════
# FIG 2 PANELS
# ═══════════════════════════════════════════
print("=== Fig 2 panels ===")

# Row 1: Cross-model CV
fig = plot_cross_model_panel(tcr_cv, TCR_MODELS, MODEL_COLORS, MODEL_DISPLAY, 'by_ap', 'AP', 'TCR CV AP (5 models)')
save_panel(fig, 2, 'a', 'tcr_cv_ap')

fig = plot_cross_model_panel(tcr_cv, TCR_MODELS, MODEL_COLORS, MODEL_DISPLAY, 'by_auroc', 'AUROC', 'TCR CV AUROC (5 models)')
save_panel(fig, 2, 'b', 'tcr_cv_auroc')

fig = plot_cross_model_panel(bcr_cv, BCR_MODELS, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY, 'by_ap', 'AP', 'BCR CV AP (5 models)')
save_panel(fig, 2, 'c', 'bcr_cv_ap')

fig = plot_cross_model_panel(bcr_cv, BCR_MODELS, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY, 'by_auroc', 'AUROC', 'BCR CV AUROC (5 models)')
save_panel(fig, 2, 'd', 'bcr_cv_auroc')

# Row 2: Per-test-set CT (TCR ATM-TCR + BCR XBCR-net)
for metric, ylabel, mk in [('by_ap', 'AP', 'ap'), ('by_auroc', 'AUROC', 'auroc')]:
    label = 'e' if mk == 'ap' else 'f'
    fig, ax = make_panel()
    for ts in TCR_CT_SETS:
        pred_path = os.path.join(RESULTS, 'atm_tcr', 'cross_test_logdist', 'predictions',
                                  f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_CACHE, f'atm_tcr_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path): continue
        te = pd.read_csv(pred_path); d = np.load(dist_path)
        if len(d) != len(te): d = d[:len(te)]
        lc = 'binder' if 'binder' in te.columns else 'y_true'
        pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
        si = np.argsort(d); bs = len(si) // 8
        bx, by = [], []
        for i in range(8):
            s = i * bs; e = len(si) if i == 7 else (i+1)*bs
            idx = si[s:e]; bx.append(d[idx].mean())
            by.append(safe_metric(mk.replace('auroc', 'aucroc'), te[lc].values[idx].astype(int), te[pc].values[idx].astype(float)))
        bx, by = np.array(bx), np.array(by)
        valid = ~np.isnan(by)
        if valid.sum() >= 3:
            r_val, _ = pearsonr(bx[valid], by[valid])
            ax.plot(bx[valid], by[valid], 'o-', color=TCR_CT_COLORS.get(ts, '#888'), markersize=3,
                    linewidth=1, label=f'{TCR_CT_DISPLAY.get(ts, ts)} ({r_val:.2f})')
    ax.set_xlabel('S2DD distance'); ax.set_ylabel(ylabel)
    ax.set_title(f'TCR CT {ylabel} (ATM-TCR)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    save_panel(fig, 2, label, f'tcr_ct_{mk}')

# BCR CT degradation — per-model panels, matching TCR CT style
# Each model gets its own panel showing per-test-set degradation lines
# fold4cal data (fold95 model for XBCR-net, model_weights.pt for others)
FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_CT_MODEL_DIRS = {'xbcr_net': 'xbcr', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                      'mint': 'mint', 'rleaai': 'rleaai'}

for bcr_model_key, bcr_model_dir in BCR_CT_MODEL_DIRS.items():
    model_disp = BCR_MODEL_DISPLAY[bcr_model_key]
    for mk, ylabel in [('ap', 'AP'), ('auroc', 'AUROC')]:
        fig, ax = make_panel()
        for ts in BCR_CT_SETS:
            fp = os.path.join(FOLD4CAL, bcr_model_dir, f'{ts}_predictions.csv')
            if not os.path.exists(fp): continue
            df = pd.read_csv(fp)
            y = df['rbd'].values.astype(int)
            p = df['pred_prob'].values.astype(float)
            # BLOSUM-sqrt: load from sidecar .npy
            if DIST_TYPE == 'blosum-sqrt':
                npy = os.path.join(FOLD4CAL, bcr_model_dir, f'{ts}_blosumsqrt_dist.npy')
                d = np.load(npy).astype(float)[:len(df)] if os.path.exists(npy) else df['distance'].values.astype(float)
            else:
                d = df['distance'].values.astype(float)
            si = np.argsort(d); bs = len(si) // 8
            bx, by = [], []
            for i in range(8):
                s = i * bs; e = len(si) if i == 7 else (i + 1) * bs
                idx = si[s:e]; bx.append(d[idx].mean())
                by.append(safe_metric(mk.replace('auroc', 'aucroc'), y[idx], p[idx]))
            bx, by = np.array(bx), np.array(by)
            valid = ~np.isnan(by)
            if valid.sum() >= 3:
                r_val, _ = pearsonr(bx[valid], by[valid])
                ax.plot(bx[valid], by[valid], 'o-', color=BCR_CT_COLORS.get(ts, '#888'),
                        markersize=3, linewidth=1, label=f'{ts} ({r_val:.2f})')
        ax.set_xlabel('S2DD distance'); ax.set_ylabel(ylabel)
        ax.set_title(f'{model_disp} CT {ylabel}', fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        save_panel(fig, 2, '', f'bcr_ct_{mk}_{bcr_model_dir}')

# Row 3: Per-model CT degradation — Combined (all samples) + Unseen-epitope-only
# "Unseen" = filter each test set to ONLY samples whose epitope is NOT in training
# This is the correct definition: removes seen-epitope samples from mixed test sets
TCR_CT_ALL_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']

# Load training epitopes for unseen filtering
_train_path = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'splits', 'train.csv')
_train_epitopes = set(pd.read_csv(_train_path)['peptide'].unique()) if os.path.exists(_train_path) else set()
print(f"  Training epitopes: {len(_train_epitopes)} unique")

def _bin_degradation(d_arr, y_arr, p_arr, metric_key, n_bins=8):
    """Bin by distance and compute metric per bin. Returns (bx, by) arrays."""
    si = np.argsort(d_arr); bs = len(si) // n_bins
    if bs < 5: return None, None
    bx, by = [], []
    for i in range(n_bins):
        s = i * bs; e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]; bx.append(d_arr[idx].mean())
        by.append(safe_metric(metric_key, y_arr[idx], p_arr[idx]))
    return np.array(bx), np.array(by)

for model in TCR_MODELS:
    model_disp = MODEL_DISPLAY[model]

    # Pre-compute combined curves for all test sets (needed as background in unseen panels)
    combined_curves = {}  # (ts, mk) -> (bx, by)
    for ts in TCR_CT_ALL_SETS:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                  f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path): continue
        te = pd.read_csv(pred_path); d = np.load(dist_path)
        if len(d) != len(te): d = d[:len(te)]
        lc = 'binder' if 'binder' in te.columns else 'y_true'
        pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
        for mk in ['auroc', 'ap']:
            bx, by = _bin_degradation(d, te[lc].values.astype(int), te[pc].values.astype(float),
                                       mk.replace('auroc', 'aucroc'))
            if bx is not None:
                combined_curves[(ts, mk)] = (bx, by)

    for scope_name, scope_tag, filter_unseen in [
        ('Combined', 'combined', False),
        ('Unseen epitopes', 'unseen', True),
    ]:
        for mk, ylabel in [('auroc', 'AUROC'), ('ap', 'AP')]:
            fig, ax = make_panel()

            # For unseen panels: draw ALL combined curves as thin dashed background first
            if filter_unseen:
                for ts in TCR_CT_ALL_SETS:
                    if (ts, mk) not in combined_curves: continue
                    bx, by = combined_curves[(ts, mk)]
                    valid = ~np.isnan(by)
                    if valid.sum() >= 3:
                        ax.plot(bx[valid], by[valid], '--', color=TCR_CT_COLORS.get(ts, '#888'),
                                linewidth=1.1, alpha=0.5, dashes=(4, 2.5), zorder=1)

            for ts in TCR_CT_ALL_SETS:
                pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                          f'{ts}_predictions_with_label.csv')
                dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
                if not os.path.exists(pred_path) or not os.path.exists(dist_path): continue
                te = pd.read_csv(pred_path); d = np.load(dist_path)
                if len(d) != len(te): d = d[:len(te)]
                lc = 'binder' if 'binder' in te.columns else 'y_true'
                pc = 'prediction' if 'prediction' in te.columns else 'y_prob'
                ep_col = 'peptide' if 'peptide' in te.columns else 'Epitope'

                if filter_unseen and ep_col in te.columns:
                    unseen_mask = ~te[ep_col].isin(_train_epitopes)
                    if unseen_mask.sum() < 30: continue
                    y_arr = te[lc].values[unseen_mask].astype(int)
                    p_arr = te[pc].values[unseen_mask].astype(float)
                    d_arr = d[unseen_mask]
                else:
                    y_arr = te[lc].values.astype(int)
                    p_arr = te[pc].values.astype(float)
                    d_arr = d

                bx, by = _bin_degradation(d_arr, y_arr, p_arr, mk.replace('auroc', 'aucroc'))
                if bx is None: continue
                valid = ~np.isnan(by)
                if valid.sum() >= 3:
                    r_val, _ = pearsonr(bx[valid], by[valid])
                    if filter_unseen:
                        # Unseen panels: no r in legend, solid line with markers
                        ax.plot(bx[valid], by[valid], 'o-', color=TCR_CT_COLORS.get(ts, '#888'),
                                markersize=3, linewidth=1.2,
                                label=f'{TCR_CT_DISPLAY.get(ts, ts)}')
                    else:
                        # Combined panels: show r in legend
                        ax.plot(bx[valid], by[valid], 'o-', color=TCR_CT_COLORS.get(ts, '#888'),
                                markersize=3, linewidth=1,
                                label=f'{TCR_CT_DISPLAY.get(ts, ts)} ({r_val:.2f})')

            ax.set_xlabel('S2DD distance'); ax.set_ylabel(ylabel)
            ax.set_title(f'{model_disp} CT {ylabel}\n({scope_name})', fontweight='bold', fontsize=12)
            handles, labels = ax.get_legend_handles_labels()
            if filter_unseen:
                from matplotlib.lines import Line2D
                handles.append(Line2D([0], [0], color='#888', linestyle='--',
                                       dashes=(4, 2.5), linewidth=1.1, alpha=0.7,
                                       label='Combined (all epitopes)'))
            ax.legend(handles=handles, fontsize=8, loc='upper right',
                      framealpha=0.85)
            save_panel(fig, 2, '', f'tcr_ct_{mk}_{scope_tag}_{model}')

# Row 4: Heatmaps — split-cell design showing r (upper-left) and ρ (lower-right)
from scipy.stats import spearmanr
from matplotlib.patches import Polygon

def plot_split_heatmap(ax, cv_data, models, display, metric_key, title):
    """Split-cell heatmap: r (upper-left triangle) and ρ (lower-right triangle)."""
    n = len(models)
    hm_r = np.full((n, 5), np.nan)
    hm_rho = np.full((n, 5), np.nan)
    for i, model in enumerate(models):
        for fold in range(5):
            key = (model, fold)
            if key not in cv_data: continue
            d = cv_data[key]
            bx, by = d['bx'], d[metric_key]
            valid = ~np.isnan(by)
            if valid.sum() >= 3:
                hm_r[i, fold], _ = pearsonr(bx[valid], by[valid])
                hm_rho[i, fold], _ = spearmanr(bx[valid], by[valid])

    im = ax.imshow(hm_r, cmap='RdBu_r', vmin=-1, vmax=-0.5, aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'F{i}' for i in range(5)])
    ax.set_yticks(range(n)); ax.set_yticklabels([display[m] for m in models], fontsize=7)

    for i in range(n):
        for j in range(5):
            # Draw diagonal split line bottom-left to top-right (matching old panel)
            ax.plot([j-0.5, j+0.5], [i+0.5, i-0.5], 'w-', linewidth=1.0, alpha=0.8)
            # Upper-left triangle: Pearson r
            if not np.isnan(hm_r[i, j]):
                c = 'white' if hm_r[i, j] < -0.8 else 'black'
                ax.text(j - 0.15, i - 0.15, f'{hm_r[i,j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color=c, fontweight='bold')
            # Lower-right triangle: Spearman ρ
            if not np.isnan(hm_rho[i, j]):
                c = 'white' if hm_rho[i, j] < -0.8 else 'black'
                ax.text(j + 0.15, i + 0.15, f'{hm_rho[i,j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color=c)
    ax.set_title(title, fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

for cv_data, models, display, metric_key, title, desc in [
    (tcr_cv, TCR_MODELS, MODEL_DISPLAY, 'by_ap', 'TCR AP\u2013S2DD r/\u03c1', 'tcr_heatmap_ap'),
    (bcr_cv, BCR_MODELS, BCR_MODEL_DISPLAY, 'by_ap', 'BCR AP\u2013S2DD r/\u03c1', 'bcr_heatmap_ap'),
]:
    fig, ax = make_panel()
    plot_split_heatmap(ax, cv_data, models, display, metric_key, title)
    save_panel(fig, 2, '', desc)

# Cross-model dot plot
fig, ax = make_panel()
y_idx = 0; all_labels = []
tcr_summary_path = os.path.join(RESULTS, 'cross_model_comparison_v2', 'model_comparison_summary.csv')
if os.path.exists(tcr_summary_path):
    ts_df = pd.read_csv(tcr_summary_path)
    # Map model names: summary uses display names
    tcr_model_name_map = {'NetTCR': 'nettcr', 'ATM-TCR': 'atm_tcr', 'BLOSUM-RF': 'blosum_rf',
                           'ERGO-II': 'ergo_ii', 'TCR-BERT': 'tcrbert'}
    for model in TCR_MODELS:
        # Find matching row (summary uses display name or raw name)
        disp = MODEL_DISPLAY[model]
        sub = ts_df[(ts_df['model'] == disp) | (ts_df['model'] == model)]
        if 'scope' in sub.columns:
            sub = sub[sub['scope'] == 'combined']
        if len(sub) > 0:
            mean_r = float(sub.iloc[0]['mean_abs_r'])
            std_r = float(sub.iloc[0]['std']) if 'std' in sub.columns else 0
            ax.errorbar(mean_r, y_idx, xerr=std_r,
                       fmt='o', color=MODEL_COLORS[model], markersize=5, capsize=2, linewidth=0.8, alpha=0.85)
        all_labels.append(MODEL_DISPLAY[model]); y_idx += 1
for model in BCR_MODELS:
    # Average across metrics for a single point per model
    rs = []
    for metric in ['aucroc', 'ap', 'f1']:
        rk = f'r_{model}_{metric}'
        if rk in bcr_summary:
            rs.append(float(bcr_summary[rk]))
    if rs:
        ax.errorbar(np.mean(rs), y_idx, xerr=np.std(rs),
                   fmt='s', color=BCR_MODEL_COLORS[model], markersize=5, capsize=2, linewidth=0.8, alpha=0.85)
    all_labels.append(BCR_MODEL_DISPLAY[model]); y_idx += 1
ax.set_yticks(range(len(all_labels))); ax.set_yticklabels(all_labels, fontsize=5)
ax.set_xlabel('Mean |Pearson r|'); ax.set_title('Cross-model consistency', fontweight='bold')
ax.set_xlim(0.1, 1.05); ax.invert_yaxis()
ax.axhline(len(TCR_MODELS) - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
save_panel(fig, 2, 'o', 'cross_model_consistency')

# Diversity effect
fig, ax = make_panel()
metric_map_div = {'AUROC': 'aucroc', 'AP': 'ap', 'F1': 'f1'}
so = {'AUROC': float(bcr_diversity['sars_only_auroc']), 'AP': float(bcr_diversity['sars_only_ap']), 'F1': float(bcr_diversity['sars_only_f1'])}
sf = {'AUROC': float(bcr_diversity['sars_flu_auroc']), 'AP': float(bcr_diversity['sars_flu_ap']), 'F1': float(bcr_diversity['sars_flu_f1'])}
for m in ['AUROC', 'AP', 'F1']:
    color = METRIC_COLORS[metric_map_div[m]]
    ax.plot([0, 1], [so[m], sf[m]], '-', color=color, linewidth=1.5, alpha=0.7)
    ax.scatter([0, 1], [so[m], sf[m]], s=50, c=color, edgecolors='white', linewidth=0.6, zorder=5)
    ax.annotate(f'+{sf[m]-so[m]:.2f}', xy=(0.5, (so[m]+sf[m])/2), fontsize=7, color=color, fontweight='bold', ha='center')
ax.set_xticks([0, 1]); ax.set_xticklabels(['SARS-only\n(37 ag)', 'SARS+Flu\n(80 ag)'])
ax.set_ylabel('Mean |Pearson r|'); ax.set_title('Antigen diversity effect', fontweight='bold')
ax.set_ylim(0, 1.05); ax.legend(['AUROC', 'AP', 'F1'], fontsize=6, loc='lower right')
save_panel(fig, 2, 'p', 'diversity_effect')

# ═══════════════════════════════════════════
# FIG 6 PANELS (from retrospective data)
# ═══════════════════════════════════════════
print("\n=== Fig 6 panels ===")
# These are already in new_fig6_retrospective.py — but as individual panels:

# Key panels that need individual files
ROOT = INPUT_DIR
da_deg = pd.read_csv(f'{INPUT_DIR}/results/deepantigen_retrospective/s2dd_degradation/binned_s2dd_sw.csv')
pp_deg = pd.read_csv(f'{INPUT_DIR}/results/panpep_retrospective/s2dd_degradation/majority_degradation_curves.csv')
aa_deg = pd.read_csv(f'{INPUT_DIR}/results/antibioticsai_retrospective/s2dd_degradation/main_degradation.csv')

def fit_deg(x, y):
    z = np.polyfit(x, y, 2)
    xs = np.linspace(x.min(), x.max(), 100)
    return xs, np.polyval(z, xs), np.std(y - np.polyval(z, x))

# Fig 6a: deepAntigen AP degradation
fig, ax = make_panel()
x, y = da_deg['mean_val'].values, da_deg['ap'].values
r, _ = pearsonr(x, y)
ax.scatter(x, y, color='#4C72B0', s=30, edgecolor='white', linewidth=0.3, zorder=5)
xs, ys, se = fit_deg(x, y)
ax.plot(xs, ys, color='#4C72B0', linewidth=1.2)
ax.fill_between(xs, ys-se, ys+se, color='#4C72B0', alpha=0.1)
ax.text(0.05, 0.08, f'r = {r:.3f}', transform=ax.transAxes, fontsize=7, fontweight='bold', color='#4C72B0',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
ax.set_xlabel('S2DD distance (BLOSUM-SW)'); ax.set_ylabel('AP')
ax.set_title('deepAntigen (TCR)', fontweight='bold')
save_panel(fig, 6, 'a', 'deepantigen_ap_degradation')

# Fig 6b: PanPep AP degradation
fig, ax = make_panel()
x, y = pp_deg['mean_dist'].values, pp_deg['ap'].values
r, _ = pearsonr(x, y)
ax.scatter(x, y, color='#55A868', s=30, edgecolor='white', linewidth=0.3, zorder=5)
xs, ys, se = fit_deg(x, y)
ax.plot(xs, ys, color='#55A868', linewidth=1.2)
ax.fill_between(xs, ys-se, ys+se, color='#55A868', alpha=0.1)
ax.text(0.05, 0.08, f'r = {r:.3f}', transform=ax.transAxes, fontsize=7, fontweight='bold', color='#55A868',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
ax.set_xlabel('S2DD distance (Levenshtein)'); ax.set_ylabel('AP')
ax.set_title('PanPep (TCR meta-learn)', fontweight='bold')
save_panel(fig, 6, 'b', 'panpep_ap_degradation')

# Fig 6c: AntibioticsAI F1 degradation
fig, ax = make_panel()
x, y = aa_deg['mean_dist'].values, aa_deg['f1'].values
r, _ = pearsonr(x, y)
ax.scatter(x, y, color='#8172B3', s=30, edgecolor='white', linewidth=0.3, zorder=5)
xs, ys, se = fit_deg(x, y)
ax.plot(xs, np.maximum(ys, 0), color='#8172B3', linewidth=1.2)
ax.fill_between(xs, np.maximum(ys-se, -0.03), ys+se, color='#8172B3', alpha=0.1)
ax.text(0.05, 0.92, f'r = {r:.3f}', transform=ax.transAxes, fontsize=7, fontweight='bold', color='#8172B3', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
ax.set_xlabel('S2DD distance (Morgan FP)'); ax.set_ylabel('F1'); ax.set_ylim(-0.05, 0.75)
ax.set_title('AntibioticsAI (drug disc.)', fontweight='bold')
save_panel(fig, 6, 'c', 'antibioticsai_f1_degradation')

# Fig 6d: BigMHC flat
bm_deg = pd.read_csv(f'{INPUT_DIR}/results/bigmhc_retrospective/s2dd_degradation/all_models_degradation.csv')
fig, ax = make_panel()
model_colors_bm = ['#C44E52', '#E07B7B', '#D4A0A0', '#B05555', '#8B3A3A', '#A66666']
bm_rs = []
for i, m in enumerate(bm_deg['model'].unique()):
    sub = bm_deg[bm_deg['model'] == m]
    r_m, _ = pearsonr(sub['mean_dist'], sub['aucroc'])
    bm_rs.append(r_m)
    ax.plot(np.arange(1, 9), sub['aucroc'].values, 'o-', color=model_colors_bm[i], markersize=2,
            linewidth=0.7, alpha=0.6, label=m.replace('BigMHC_', ''))
ax.axhline(0.5, color='gray', linewidth=0.4, linestyle=':', alpha=0.5)
ax.text(0.05, 0.08, f'mean r = {np.mean(bm_rs):+.3f}\n(no degradation)', transform=ax.transAxes, fontsize=6,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3f3', edgecolor='#C44E52', alpha=0.9, linewidth=0.3))
ax.set_xlabel('Distance bin'); ax.set_ylabel('AUROC'); ax.set_ylim(0.38, 0.68)
ax.legend(fontsize=9, loc='upper right', ncol=2); ax.set_title('BigMHC (immunogenicity)', fontweight='bold')
save_panel(fig, 6, 'd', 'bigmhc_flat')

print("\n=== Panel generation complete ===")
# Count all panels
total = 0
for fig_dir in sorted(os.listdir(PANEL_DIR)):
    d = os.path.join(PANEL_DIR, fig_dir)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.endswith('.pdf')])
        total += n
        print(f"  {fig_dir}: {n} panels")
print(f"  Total: {total} panels")
