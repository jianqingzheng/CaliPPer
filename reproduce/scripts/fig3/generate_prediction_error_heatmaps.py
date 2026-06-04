#!/usr/bin/env python3
"""Fig3 dataset-level prediction error heatmaps (12 panels).

Circle-grid visualization:
  - Circle SIZE = ground truth metric value (actual performance)
  - Circle COLOR = absolute prediction error |predicted - actual|
  - One circle per (model × fold/test set) cell

Panels:
  fig3_{tcr,bcr}_{cv,ct}_pred_error_{aucroc,ap,f1}.{pdf,png}
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.dirname(SCRIPT_DIR)
DESIGNED_DIR = os.path.dirname(os.path.dirname(_FIG_DIR))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
sys.path.insert(0, DESIGNED_DIR)
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))

from style_config import (apply_publication_style,
                           MODEL_DISPLAY, BCR_MODEL_DISPLAY)
from calipper.general_evaluator import safe_metric
from calipper.core import predict_metric as s2dd_predict
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE

apply_publication_style()

PANEL_DIR = os.path.join(FIG_DIR, 'fig3', DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
# Only v3+v4 as calibration: simulates real-case where user calibrates on standard
# validation splits. Seen/unseen/IEDB/McPAS excluded (anomalous or external).
TCR_EXCLUDE_FROM_CAL = {'iedb_sars', 'mcpas', 'seen_test', 'unseen_fold34'}
TCR_CT_DISPLAY = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen', 'v3_combined': 'v3',
                   'v4_combined': 'v4', 'mcpas': 'McPAS', 'iedb_sars': 'IEDB'}
LABEL_COL = {'nettcr': ('binder', 'prediction'), 'atm_tcr': ('y_true', 'y_prob'),
             'blosum_rf': ('binder', 'prediction'), 'ergo_ii': ('y_true', 'y_prob'),
             'tcrbert': ('y_true', 'y_prob')}

BCR_MODELS = ['xbcr_net', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_CT_SETS_MAP = {
    'xbcr_net': ['A1-A11', 'unseen', 'flu'],
    'deepaai':  ['A1-A11', 'unseen', 'flu'],
    'mambaaai': ['A1-A11', 'unseen', 'flu'],
    'mint':     ['A1-A11', 'unseen', 'flu'],
    'rleaai':   ['A1-A11', 'unseen', 'flu'],
}
ALL_BCR_CT = ['A1-A11', 'unseen', 'flu']

METRICS = ['aucroc', 'ap', 'f1']
MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}


# ─────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────

def load_tcr_cv_fold(model, fold):
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    if not os.path.exists(test_path): return None
    parts = [pd.read_csv(test_path)]
    for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fold_dir, vname)
        if os.path.exists(vp): parts.append(pd.read_csv(vp)); break
    df = pd.concat(parts, ignore_index=True)
    if DIST_TYPE == 'blosum-sqrt':
        suffixes = ['_blosumsqrt_dist']
    else:
        suffixes = ['_combined_dist', '_dist']
    for suffix in suffixes:
        dp = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dp):
            d = np.load(dp); n = min(len(d), len(df))
            lc_act = lc if lc in df.columns else ('binder' if 'binder' in df.columns else 'y_true')
            pc_act = pc if pc in df.columns else ('prediction' if 'prediction' in df.columns else 'y_prob')
            return (df[lc_act].values[:n].astype(int),
                    df[pc_act].values[:n].astype(float), d[:n])
    return None


def load_tcr_ct(model, ts):
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                              f'{ts}_predictions_with_label.csv')
    dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
    if not os.path.exists(pred_path) or not os.path.exists(dist_path): return None
    te = pd.read_csv(pred_path); d = np.load(dist_path); n = min(len(d), len(te))
    lc_act = lc if lc in te.columns else ('binder' if 'binder' in te.columns else 'y_true')
    pc_act = pc if pc in te.columns else ('prediction' if 'prediction' in te.columns else 'y_prob')
    return (te[lc_act].values[:n].astype(int),
            te[pc_act].values[:n].astype(float), d[:n])


def load_bcr_cv_fold(model_key, fold):
    model_dir = 'xbcr' if model_key == 'xbcr_net' else model_key
    test_path = os.path.join(RESULTS, model_dir, 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
    if not os.path.exists(test_path): return None
    te = pd.read_csv(test_path)
    if 'distance' not in te.columns: return None
    pc = 'pred_prob' if 'pred_prob' in te.columns else 'output'
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        npy = os.path.join(RESULTS, 'fig2_cache', f'{model_dir}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        d = np.load(npy).astype(float)[:len(te)] if os.path.exists(npy) else te['distance'].values.astype(float)
    else:
        d = te['distance'].values.astype(float)
    return (te['rbd'].values.astype(int), te[pc].values.astype(float), d)


def load_bcr_ct(model_key, ts):
    model_dir = 'xbcr' if model_key == 'xbcr_net' else model_key
    model_ct_dir = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal', model_dir)
    pred_path = os.path.join(model_ct_dir, f'{ts}_predictions.csv')
    if not os.path.exists(pred_path): return None
    te = pd.read_csv(pred_path)
    if 'distance' not in te.columns: return None
    pc = 'pred_prob' if 'pred_prob' in te.columns else 'output'
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        npy = os.path.join(model_ct_dir, f'{ts}_blosumsqrt_dist.npy')
        d = np.load(npy).astype(float)[:len(te)] if os.path.exists(npy) else te['distance'].values.astype(float)
    else:
        d = te['distance'].values.astype(float)
    return (te['rbd'].values.astype(int), te[pc].values.astype(float), d)


# ─────────────────────────────────────────────────────────────────
# Degradation correlation computation
# ─────────────────────────────────────────────────────────────────

def compute_degradation(y, p, d, metric, n_bins=8):
    """Bin by distance, compute metric per bin, return Pearson r and slope."""
    from scipy.stats import pearsonr as _pr
    si = np.argsort(d)
    bs = max(len(si) // n_bins, 1)
    if bs < 10:
        return np.nan, 0.0
    bin_d, bin_m = [], []
    for i in range(n_bins):
        s = i * bs; e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        m_val = safe_metric(metric, y[idx], p[idx])
        if not np.isnan(m_val):
            bin_d.append(d[idx].mean())
            bin_m.append(m_val)
    if len(bin_d) < 4:
        return np.nan, 0.0
    bd, bm = np.array(bin_d), np.array(bin_m)
    try:
        r, _ = _pr(bd, bm)
    except Exception as _e_pr:
        import sys as _s_pr
        print(f"  ⚠ FALLBACK [prediction_error_heatmaps]: pearsonr failed ({type(_e_pr).__name__}: {_e_pr}); leaving r=NaN", file=_s_pr.stderr, flush=True)
        r = np.nan
    # Linear regression slope (normalized)
    if np.std(bd) > 1e-12:
        slope = np.polyfit(bd, bm, 1)[0]
    else:
        slope = 0.0
    return r, slope


# ─────────────────────────────────────────────────────────────────
# Ellipse-grid heatmap
# ─────────────────────────────────────────────────────────────────

def plot_ellipse_grid(ax, actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                       row_labels, col_labels, title,
                       metric_name='AUROC', max_error=0.3, show_legend=True,
                       legend_position='right', legend_parts=None,
                       show_colorbar=True):
    """Ellipse-grid visualization:
      - Ellipse AREA ∝ ground truth metric (bigger = better model)
      - Ellipse COLOR = |prediction error| (green=low, red=high)
      - Ellipse ASPECT RATIO = |degradation r| (elongated = strong correlation)
      - Ellipse ROTATION = slope direction (\ = negative slope, / = positive)
      - TEXT inside = absolute error value
    """
    from matplotlib.patches import Ellipse
    n_rows, n_cols = actual_mat.shape

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=max_error)

    base_area = 0.60  # max semi-major when actual=1.0
    import matplotlib.patheffects as pe

    for row in range(n_rows):
        for col in range(n_cols):
            actual = actual_mat[row, col]
            error = error_mat[row, col]
            deg_r = deg_r_mat[row, col]
            slope = deg_slope_mat[row, col]

            if np.isnan(actual) or np.isnan(error):
                ax.text(col, row, '—', ha='center', va='center', fontsize=6,
                        color='#ccc', fontweight='bold')
                continue

            # AREA proportional to actual metric (sqrt for radius)
            # area = pi * a * b ∝ actual → a ∝ sqrt(actual)
            # Floor at 0.4 so even low-metric cells are visible
            scale = max(0.4, np.sqrt(max(0.1, actual)))
            semi_major = base_area * scale

            # Aspect ratio from |degradation r|: |r|=1 → 3:1, |r|=0 → 1:1
            abs_r = abs(deg_r) if not np.isnan(deg_r) else 0.0
            aspect = 1.0 - 0.65 * abs_r  # 1.0 (circle) → 0.35 (very elongated)
            semi_minor = semi_major * aspect

            # Rotation: slope direction amplified to visible angle
            # Typical data: d_range ≈ 3-5, metric_range ≈ 0.2-0.5
            # Normalize slope by typical ratio (d/m ≈ 10) then amplify 1.5×
            # Negate for y-axis inversion
            if not np.isnan(slope) and abs(slope) > 1e-6:
                norm_slope = slope * 10.0  # normalize metric/distance units
                angle = -np.degrees(np.arctan(norm_slope * 1.5))
                angle = np.clip(angle, -60, 60)
            else:
                angle = 0.0

            ellipse = Ellipse((col, row), width=2*semi_major, height=2*semi_minor,
                              angle=angle,
                              facecolor=cmap(norm(error)),
                              edgecolor='#444', linewidth=0.5, zorder=5)
            ax.add_patch(ellipse)

            # Text: absolute error value with white outline for readability
            ax.text(col, row, f'{error:.3f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # Grid lines
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color='#eee', linewidth=0.5, zorder=1)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color='#eee', linewidth=0.5, zorder=1)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    ax.set_aspect('equal')

    # Colorbar for error
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
        cbar.set_label('|Prediction error|', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    # ── Ellipse legend ──
    # legend_parts: list of parts to show. Default (None) = all.
    #   'size'  = ellipse size legend
    #   'ratio' = axis ratio (|r|) legend
    #   'tilt'  = tilt/slope direction legend
    # legend_position: 'right' (default) or 'bottom'
    if show_legend:
        from matplotlib.patches import Ellipse as _E

        if legend_parts is None:
            legend_parts = ['size', 'ratio', 'tilt']

        if legend_position == 'bottom':
            # Draw legend items horizontally below the grid
            leg_y = n_rows + 0.4
            x_cursor = 0.0
            spacing = n_cols / max(len(legend_parts), 1)

            for pi, part in enumerate(legend_parts):
                x0 = pi * spacing

                if part == 'size':
                    ax.text(x0 + 0.0, leg_y, 'Ellipse legend', fontsize=7,
                            fontweight='bold', va='top', clip_on=False)
                    for ki, (val, lab) in enumerate([(0.8, f'{metric_name}=0.8'),
                                                      (0.4, f'{metric_name}=0.4')]):
                        xx = x0 + ki * 1.0
                        r = base_area * val * 0.7
                        ax.add_patch(_E((xx + 0.15, leg_y + 0.55), 2*r, 2*r,
                                        facecolor='#ddd', edgecolor='#666',
                                        linewidth=0.4, clip_on=False, zorder=10))
                        ax.text(xx + 0.55, leg_y + 0.55, lab, fontsize=6,
                                va='center', clip_on=False)

                elif part == 'ratio':
                    ax.text(x0 + 0.0, leg_y, 'Axis ratio = |r|', fontsize=7,
                            fontstyle='italic', va='top', clip_on=False)
                    ax.add_patch(_E((x0 + 0.15, leg_y + 0.55), 0.25, 0.25,
                                    facecolor='#ddd', edgecolor='#666',
                                    linewidth=0.4, clip_on=False, zorder=10))
                    ax.text(x0 + 0.55, leg_y + 0.55, '|r|≈0', fontsize=6,
                            va='center', clip_on=False)
                    ax.add_patch(_E((x0 + 1.15, leg_y + 0.55), 0.35, 0.12, angle=-40,
                                    facecolor='#ddd', edgecolor='#666',
                                    linewidth=0.4, clip_on=False, zorder=10))
                    ax.text(x0 + 1.55, leg_y + 0.55, '|r|≈1', fontsize=6,
                            va='center', clip_on=False)

                elif part == 'tilt':
                    ax.text(x0 + 0.0, leg_y, 'Tilt = slope sign', fontsize=7,
                            fontstyle='italic', va='top', clip_on=False)
                    ax.add_patch(_E((x0 + 0.15, leg_y + 0.55), 0.3, 0.1, angle=-40,
                                    facecolor='#8fbc8f', edgecolor='#444',
                                    linewidth=0.4, clip_on=False, zorder=10))
                    ax.text(x0 + 0.55, leg_y + 0.55, '\\ degrad.', fontsize=6,
                            va='center', clip_on=False)
                    ax.add_patch(_E((x0 + 1.15, leg_y + 0.55), 0.3, 0.1, angle=40,
                                    facecolor='#f0c080', edgecolor='#444',
                                    linewidth=0.4, clip_on=False, zorder=10))
                    ax.text(x0 + 1.55, leg_y + 0.55, '/ no degrad.', fontsize=6,
                            va='center', clip_on=False)

        else:
            # Original right-side legend
            leg_x0 = n_cols + 0.3
            leg_y0 = n_rows * 0.15

            if 'size' in legend_parts:
                ax.text(leg_x0 + 0.55, leg_y0 - 0.35, 'Ellipse legend', fontsize=5,
                        fontweight='bold', ha='center', va='bottom')
                for ki, (val, lab) in enumerate([(0.8, f'{metric_name}=0.8'),
                                                  (0.4, f'{metric_name}=0.4')]):
                    yy = leg_y0 + ki * 0.55
                    r = base_area * val * 0.7
                    ax.add_patch(_E((leg_x0 + 0.15, yy), 2*r, 2*r, facecolor='#ddd',
                                    edgecolor='#666', linewidth=0.4, clip_on=False, zorder=10))
                    ax.text(leg_x0 + 0.65, yy, lab, fontsize=4, va='center', clip_on=False)

            if 'ratio' in legend_parts:
                yy_base = leg_y0 + 1.3
                ax.text(leg_x0 + 0.55, yy_base - 0.25, 'Axis ratio = |r|', fontsize=7,
                        ha='center', va='bottom', fontstyle='italic', clip_on=False)
                ax.add_patch(_E((leg_x0 + 0.15, yy_base + 0.15), 0.28, 0.28, facecolor='#ddd',
                                edgecolor='#666', linewidth=0.4, clip_on=False, zorder=10))
                ax.text(leg_x0 + 0.65, yy_base + 0.15, '|r|≈0', fontsize=4, va='center', clip_on=False)
                ax.add_patch(_E((leg_x0 + 0.15, yy_base + 0.6), 0.38, 0.13, angle=-40,
                                facecolor='#ddd', edgecolor='#666', linewidth=0.4, clip_on=False, zorder=10))
                ax.text(leg_x0 + 0.65, yy_base + 0.6, '|r|≈1', fontsize=4, va='center', clip_on=False)

            if 'tilt' in legend_parts:
                yy_tilt = leg_y0 + 2.5
                ax.text(leg_x0 + 0.55, yy_tilt - 0.25, 'Tilt = slope sign', fontsize=7,
                        ha='center', va='bottom', fontstyle='italic', clip_on=False)
                ax.add_patch(_E((leg_x0 + 0.15, yy_tilt + 0.15), 0.35, 0.12, angle=-40,
                                facecolor='#8fbc8f', edgecolor='#444', linewidth=0.4, clip_on=False, zorder=10))
                ax.text(leg_x0 + 0.65, yy_tilt + 0.15, '\\ degrad.', fontsize=4, va='center', clip_on=False)
                ax.add_patch(_E((leg_x0 + 0.15, yy_tilt + 0.6), 0.35, 0.12, angle=40,
                                facecolor='#f0c080', edgecolor='#444', linewidth=0.4, clip_on=False, zorder=10))
                ax.text(leg_x0 + 0.65, yy_tilt + 0.6, '/ no degrad.', fontsize=4, va='center', clip_on=False)


def save_ellipse_heatmap(actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                          row_labels, col_labels, title, out_path,
                          metric_name='AUROC', show_legend=True,
                          legend_position='right', legend_parts=None,
                          show_colorbar=True, fig_height=None):
    n_cols = actual_mat.shape[1]
    if not show_legend:
        fig_w = max(4.0, 1.5 + n_cols * 0.8)
        fig_h = max(3.5, 1.0 + actual_mat.shape[0] * 0.65)
    elif legend_position == 'bottom':
        fig_w = max(4.0, 1.5 + n_cols * 0.8)
        fig_h = max(4.5, 1.0 + actual_mat.shape[0] * 0.65 + 1.2)
    else:
        fig_w = max(5.0, 2.5 + n_cols * 0.8)
        fig_h = max(3.5, 1.0 + actual_mat.shape[0] * 0.65)
    if fig_height is not None:
        fig_h = fig_height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_ellipse_grid(ax, actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                       row_labels, col_labels, title, metric_name=metric_name,
                       show_legend=show_legend, legend_position=legend_position,
                       legend_parts=legend_parts, show_colorbar=show_colorbar)
    fig.savefig(out_path + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# TCR CV: halfsplit → dataset-level prediction
# ─────────────────────────────────────────────────────────────────

def compute_tcr_cv():
    print("=== TCR CV prediction error ===")
    for metric in METRICS:
        actual_mat = np.full((len(TCR_MODELS), 5), np.nan)
        error_mat = np.full((len(TCR_MODELS), 5), np.nan)
        deg_r_mat = np.full((len(TCR_MODELS), 5), np.nan)
        deg_slope_mat = np.full((len(TCR_MODELS), 5), 0.0)
        for i, model in enumerate(TCR_MODELS):
            for fold in range(5):
                data = load_tcr_cv_fold(model, fold)
                if data is None: continue
                y, p, d = data
                si = np.argsort(d)
                cal_idx, test_idx = si[::2], si[1::2]
                cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
                result = s2dd_predict(cal_data, p[test_idx], d[test_idx], metrics=[metric])
                actual = safe_metric(metric, y[test_idx], p[test_idx])
                predicted = result['estimated'].get(metric, np.nan)
                if not np.isnan(actual) and not np.isnan(predicted):
                    actual_mat[i, fold] = actual
                    error_mat[i, fold] = abs(predicted - actual)
                # Degradation on full fold data
                r, slope = compute_degradation(y, p, d, metric)
                deg_r_mat[i, fold] = r
                deg_slope_mat[i, fold] = slope

        row_labels = [MODEL_DISPLAY[m] for m in TCR_MODELS]
        col_labels = [f'F{i}' for i in range(5)]
        out = os.path.join(PANEL_DIR, f'fig3_tcr_cv_pred_error_{metric}')
        save_ellipse_heatmap(actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                              row_labels, col_labels,
                              f'TCR CV {MDISP[metric]} prediction', out, MDISP[metric])
        mean_err = np.nanmean(error_mat)
        print(f'  {metric}: mean error={mean_err:.3f}')


# ─────────────────────────────────────────────────────────────────
# TCR CT: LOO → dataset-level prediction
# ─────────────────────────────────────────────────────────────────

def compute_tcr_ct():
    print("\n=== TCR CT prediction error ===")
    for metric in METRICS:
        actual_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), np.nan)
        error_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), np.nan)
        deg_r_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), np.nan)
        deg_slope_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), 0.0)
        for i, model in enumerate(TCR_MODELS):
            ct_all = {}
            for ts in TCR_CT_SETS:
                data = load_tcr_ct(model, ts)
                if data is not None:
                    ct_all[ts] = data
            for j, ts in enumerate(TCR_CT_SETS):
                if ts not in ct_all: continue
                test_y, test_p, test_d = ct_all[ts]
                # Degradation on this test set
                r, slope = compute_degradation(test_y, test_p, test_d, metric)
                deg_r_mat[i, j] = r
                deg_slope_mat[i, j] = slope
                # Prediction (exclude external sets from calibration)
                others = [p for p in ct_all if p != ts and p not in TCR_EXCLUDE_FROM_CAL]
                if not others: continue
                cal_data = {p: ct_all[p] for p in others}
                result = s2dd_predict(cal_data, test_p, test_d, metrics=[metric])
                actual = safe_metric(metric, test_y, test_p)
                predicted = result['estimated'].get(metric, np.nan)
                if not np.isnan(actual) and not np.isnan(predicted):
                    actual_mat[i, j] = actual
                    error_mat[i, j] = abs(predicted - actual)

        row_labels = [MODEL_DISPLAY[m] for m in TCR_MODELS]
        col_labels = [TCR_CT_DISPLAY[ts] for ts in TCR_CT_SETS]
        out = os.path.join(PANEL_DIR, f'fig3_tcr_ct_pred_error_{metric}')
        save_ellipse_heatmap(actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                              row_labels, col_labels,
                              f'TCR CT {MDISP[metric]} prediction', out, MDISP[metric],
                              show_legend=True, legend_position='bottom',
                              legend_parts=['size', 'ratio'], fig_height=5.5)
        mean_err = np.nanmean(error_mat)
        print(f'  {metric}: mean error={mean_err:.3f}')


# ─────────────────────────────────────────────────────────────────
# BCR CV: halfsplit → dataset-level prediction
# ─────────────────────────────────────────────────────────────────

def compute_bcr_cv():
    print("\n=== BCR CV prediction error ===")
    for metric in METRICS:
        actual_mat = np.full((len(BCR_MODELS), 5), np.nan)
        error_mat = np.full((len(BCR_MODELS), 5), np.nan)
        deg_r_mat = np.full((len(BCR_MODELS), 5), np.nan)
        deg_slope_mat = np.full((len(BCR_MODELS), 5), 0.0)
        for i, model in enumerate(BCR_MODELS):
            for fold in range(5):
                data = load_bcr_cv_fold(model, fold)
                if data is None: continue
                y, p, d = data
                r, slope = compute_degradation(y, p, d, metric)
                deg_r_mat[i, fold] = r
                deg_slope_mat[i, fold] = slope
                si = np.argsort(d)
                cal_idx, test_idx = si[::2], si[1::2]
                cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
                result = s2dd_predict(cal_data, p[test_idx], d[test_idx], metrics=[metric])
                actual = safe_metric(metric, y[test_idx], p[test_idx])
                predicted = result['estimated'].get(metric, np.nan)
                if not np.isnan(actual) and not np.isnan(predicted):
                    actual_mat[i, fold] = actual
                    error_mat[i, fold] = abs(predicted - actual)

        row_labels = [BCR_MODEL_DISPLAY[m] for m in BCR_MODELS]
        col_labels = [f'F{i}' for i in range(5)]
        out = os.path.join(PANEL_DIR, f'fig3_bcr_cv_pred_error_{metric}')
        save_ellipse_heatmap(actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                              row_labels, col_labels,
                              f'BCR CV {MDISP[metric]} prediction', out, MDISP[metric])
        mean_err = np.nanmean(error_mat)
        print(f'  {metric}: mean error={mean_err:.3f}')


# ─────────────────────────────────────────────────────────────────
# BCR CT: fold4-as-cal → dataset-level prediction (updated 2026-04-25)
# ─────────────────────────────────────────────────────────────────

def compute_bcr_ct():
    print("\n=== BCR CT prediction error (fold4-as-cal) ===")
    FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
    for metric in METRICS:
        actual_mat = np.full((len(BCR_MODELS), len(ALL_BCR_CT)), np.nan)
        error_mat = np.full((len(BCR_MODELS), len(ALL_BCR_CT)), np.nan)
        deg_r_mat = np.full((len(BCR_MODELS), len(ALL_BCR_CT)), np.nan)
        deg_slope_mat = np.full((len(BCR_MODELS), len(ALL_BCR_CT)), 0.0)
        for i, model in enumerate(BCR_MODELS):
            model_dir = 'xbcr' if model == 'xbcr_net' else model
            # Load model-specific cal from fold4-as-cal pipeline
            cal_ct_dir = os.path.join(FOLD4CAL, model_dir)
            cal_path = os.path.join(cal_ct_dir, 'cal_predictions.csv')
            if not os.path.exists(cal_path): continue
            cal = pd.read_csv(cal_path)
            if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                npy = os.path.join(cal_ct_dir, 'cal_predictions_blosumsqrt_dist.npy')
                cal_d = np.load(npy).astype(float)[:len(cal)] if os.path.exists(npy) else cal['distance'].values.astype(float)
            else:
                cal_d = cal['distance'].values.astype(float)
            cal_data = {'fold4_test': (cal['rbd'].values.astype(int),
                                        cal['pred_prob'].values.astype(float),
                                        cal_d)}
            for j, ts in enumerate(ALL_BCR_CT):
                data = load_bcr_ct(model, ts)
                if data is None: continue
                test_y, test_p, test_d = data
                r, slope = compute_degradation(test_y, test_p, test_d, metric)
                deg_r_mat[i, j] = r
                deg_slope_mat[i, j] = slope
                result = s2dd_predict(cal_data, test_p, test_d, metrics=[metric])
                actual = safe_metric(metric, test_y, test_p)
                predicted = result['estimated'].get(metric, np.nan)
                if not np.isnan(actual) and not np.isnan(predicted):
                    actual_mat[i, j] = actual
                    error_mat[i, j] = abs(predicted - actual)

        row_labels = [BCR_MODEL_DISPLAY[m] for m in BCR_MODELS]
        col_labels = ALL_BCR_CT
        out = os.path.join(PANEL_DIR, f'fig3_bcr_ct_pred_error_{metric}')
        save_ellipse_heatmap(actual_mat, error_mat, deg_r_mat, deg_slope_mat,
                              row_labels, col_labels,
                              f'BCR CT {MDISP[metric]} prediction', out, MDISP[metric],
                              show_legend=True, legend_position='bottom',
                              legend_parts=['tilt'], show_colorbar=False,
                              fig_height=5.5)
        mean_err = np.nanmean(error_mat)
        print(f'  {metric}: mean error={mean_err:.3f}')


if __name__ == '__main__':
    compute_tcr_cv()
    compute_tcr_ct()
    compute_bcr_cv()
    compute_bcr_ct()
    print("\nDone. Generated 12 circle-grid prediction error panels.")
