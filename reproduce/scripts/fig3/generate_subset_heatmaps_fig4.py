#!/usr/bin/env python3
"""Fig3 prediction correlation heatmaps (12 panels).

For each (receptor × split × metric) combination:
  - Rows: models (5 TCR or 5 BCR models)
  - Cols: folds (F0-F4 for CV) or test sets (CT)
  - Upper-left triangle: Pearson r (predicted vs actual across bins)
  - Lower-right triangle: Spearman ρ

Style: matches fig2_tcr_heatmap_ap_split.png (RdBu_r, white edges, split cells).

Panels produced:
  fig3_tcr_cv_heatmap_{aucroc,ap,f1}_split.{pdf,png}
  fig3_tcr_ct_heatmap_{aucroc,ap,f1}_split.{pdf,png}
  fig3_bcr_cv_heatmap_{aucroc,ap,f1}_split.{pdf,png}
  fig3_bcr_ct_heatmap_{aucroc,ap,f1}_split.{pdf,png}
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

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
from calipper.core import adaptive_n_bins
from PAPE.pape_core import (estimate_importance_weights, fit_weighted_calibration,
                             apply_calibration, estimate_metric as pape_eq4)
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE

apply_publication_style()

PANEL_DIR = os.path.join(FIG_DIR, 'fig3', DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
TCR_CT_DISPLAY = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen', 'v3_combined': 'v3',
                   'v4_combined': 'v4', 'mcpas': 'McPAS', 'iedb_sars': 'IEDB'}
LABEL_COL = {'nettcr': ('binder', 'prediction'), 'atm_tcr': ('binder', 'prediction'),
             'blosum_rf': ('binder', 'prediction'), 'ergo_ii': ('binder', 'prediction'),
             'tcrbert': ('binder', 'prediction')}
TCR_CT_CAL_SETS = {'v3_combined', 'v4_combined'}  # only v3+v4 for calibration

BCR_MODELS_CV = ['xbcr_net', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_CT_SETS = {
    'xbcr_net': ['A1-A11', 'unseen', 'flu'],
    'deepaai':  ['A1-A11', 'unseen', 'flu'],
    'mambaaai': ['A1-A11', 'unseen', 'flu'],
    'mint':     ['A1-A11', 'unseen', 'flu'],
    'rleaai':   ['A1-A11', 'unseen', 'flu'],
}
BCR_CT_DISPLAY = {'A1-A11': 'A1-A11', 'unseen': 'unseen', 'flu': 'Influenza'}

METRICS = ['aucroc', 'ap', 'f1']
MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}
from calipper.core import predict_subset_metric


# ─────────────────────────────────────────────────────────────────
# Data loaders (reuse from existing scripts)
# ─────────────────────────────────────────────────────────────────

def load_tcr_cv_fold(model, fold):
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    if not os.path.exists(test_path): return None
    parts = [pd.read_csv(test_path)]
    for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fold_dir, vname)
        if os.path.exists(vp):
            parts.append(pd.read_csv(vp)); break
    df = pd.concat(parts, ignore_index=True)
    if DIST_TYPE == 'blosum-sqrt':
        suffixes = ['_blosumsqrt_dist']
    else:
        suffixes = ['_combined_dist', '_dist']
    for suffix in suffixes:
        dp = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dp):
            d = np.load(dp)
            n = min(len(d), len(df))
            lc_act = lc if lc in df.columns else ('binder' if 'binder' in df.columns else 'y_true')
            pc_act = pc if pc in df.columns else ('prediction' if 'prediction' in df.columns else 'y_prob')
            return {'y': df[lc_act].values[:n].astype(int),
                    'p': df[pc_act].values[:n].astype(float),
                    'd': d[:n]}
    return None


def load_tcr_ct(model, ts):
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                              f'{ts}_predictions_with_label.csv')
    dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
    if not os.path.exists(pred_path) or not os.path.exists(dist_path): return None
    te = pd.read_csv(pred_path); d = np.load(dist_path)
    n = min(len(d), len(te))
    lc_act = lc if lc in te.columns else ('binder' if 'binder' in te.columns else 'y_true')
    pc_act = pc if pc in te.columns else ('prediction' if 'prediction' in te.columns else 'y_prob')
    return {'y': te[lc_act].values[:n].astype(int),
            'p': te[pc_act].values[:n].astype(float),
            'd': d[:n]}


def load_bcr_cv_fold(model_key, fold):
    model_dir = 'xbcr' if model_key == 'xbcr_net' else model_key
    test_path = os.path.join(RESULTS, model_dir, 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
    if not os.path.exists(test_path): return None
    te = pd.read_csv(test_path)
    if 'distance' not in te.columns: return None
    pc = 'pred_prob' if 'pred_prob' in te.columns else 'output'
    # BLOSUM-sqrt: load from sidecar .npy
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        npy = os.path.join(RESULTS, 'fig2_cache', f'{model_dir}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        d = np.load(npy).astype(float)[:len(te)] if os.path.exists(npy) else te['distance'].values.astype(float)
    else:
        d = te['distance'].values.astype(float)
    return {'y': te['rbd'].values.astype(int),
            'p': te[pc].values.astype(float),
            'd': d}


def load_bcr_ct(model_key, ts):
    # fold4-as-cal pipeline (fold95 model, updated 2026-04-25)
    model_dir = 'xbcr' if model_key == 'xbcr_net' else model_key
    model_ct_dir = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal', model_dir)
    pred_path = os.path.join(model_ct_dir, f'{ts}_predictions.csv')
    if not os.path.exists(pred_path): return None
    te = pd.read_csv(pred_path)
    if 'distance' not in te.columns: return None
    pc = 'pred_prob' if 'pred_prob' in te.columns else 'output'
    # BLOSUM-sqrt: load from sidecar .npy
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        npy = os.path.join(model_ct_dir, f'{ts}_blosumsqrt_dist.npy')
        d = np.load(npy).astype(float)[:len(te)] if os.path.exists(npy) else te['distance'].values.astype(float)
    else:
        d = te['distance'].values.astype(float)
    return {'y': te['rbd'].values.astype(int),
            'p': te[pc].values.astype(float),
            'd': d}


# ─────────────────────────────────────────────────────────────────
# Bin-level prediction (halfsplit for CV, LOO for CT)
# ─────────────────────────────────────────────────────────────────

def predict_bins_halfsplit(data, metric, n_bins=8):
    """Within-fold halfsplit → bin-level predicted vs actual using v2.7."""
    from calipper.core import adaptive_n_bins
    y, p, d = data['y'], data['p'], data['d']
    si = np.argsort(d)
    cal_idx, test_idx = si[::2], si[1::2]

    cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
    test_y, test_p, test_d = y[test_idx], p[test_idx], d[test_idx]

    # Split test half into bins as subsets
    n_pos = int((y[cal_idx]==1).sum()); n_neg = int((y[cal_idx]==0).sum())
    nb = adaptive_n_bins(n_pos, n_neg)
    si_t = np.argsort(test_d)
    bs_t = max(len(si_t) // nb, 1)
    if bs_t < 20:
        return [], []

    test_subsets = {}
    for i in range(nb):
        s = i * bs_t; e = len(si_t) if i == nb - 1 else (i + 1) * bs_t
        idx = si_t[s:e]
        test_subsets[f'bin{i}'] = (test_y[idx], test_p[idx], test_d[idx])

    results = predict_subset_metric(cal_data, test_subsets, metrics=[metric])
    preds = [r['predicted'] for r in results]
    actuals = [r['actual'] for r in results]
    return preds, actuals


def predict_bins_loo(ct_data, held, metric, n_bins=8, cal_sets=None):
    """LOO: hold out 'held' test set, predict its bins using v2.7.

    cal_sets: if provided, restrict calibration to these sets only (e.g. TCR_CT_CAL_SETS).
    """
    from calipper.core import adaptive_n_bins
    if cal_sets is not None:
        others = [p for p in cal_sets if p != held and p in ct_data]
    else:
        others = [p for p in ct_data if p != held]
    if not others:
        return [], []

    cal_data = {p: (ct_data[p]['y'], ct_data[p]['p'], ct_data[p]['d']) for p in others}
    test = ct_data[held]
    test_y, test_p, test_d = test['y'], test['p'], test['d']

    # Split held-out test into bins as subsets
    cal_y = np.concatenate([v[0] for v in cal_data.values()])
    nb = adaptive_n_bins(int((cal_y==1).sum()), int((cal_y==0).sum()))
    si_t = np.argsort(test_d)
    bs_t = max(len(si_t) // nb, 1)
    if bs_t < 20:
        return [], []

    test_subsets = {}
    for i in range(nb):
        s = i * bs_t; e = len(si_t) if i == nb - 1 else (i + 1) * bs_t
        idx = si_t[s:e]
        test_subsets[f'bin{i}'] = (test_y[idx], test_p[idx], test_d[idx])

    results = predict_subset_metric(cal_data, test_subsets, metrics=[metric])
    preds = [r['predicted'] for r in results]
    actuals = [r['actual'] for r in results]
    return preds, actuals


# ─────────────────────────────────────────────────────────────────
# Split-triangle heatmap plotting (matches fig2 style)
# ─────────────────────────────────────────────────────────────────

def plot_split_heatmap(ax, r_mat, rho_mat, row_labels, col_labels, title,
                        cmap=None, vmin=-1, vmax=1):
    """Split-cell heatmap: upper-left=Pearson r, lower-right=Spearman ρ."""
    if cmap is None:
        cmap = plt.cm.RdBu_r
    n_rows, n_cols = r_mat.shape
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for row in range(n_rows):
        for col in range(n_cols):
            r_val = r_mat[row, col]
            rho_val = rho_mat[row, col] if rho_mat is not None else np.nan

            if not np.isnan(r_val):
                tri_ul = plt.Polygon(
                    [[col - 0.5, row - 0.5], [col + 0.5, row - 0.5], [col - 0.5, row + 0.5]],
                    closed=True, facecolor=cmap(norm(r_val)), edgecolor='white', linewidth=0.5)
                ax.add_patch(tri_ul)
            if not np.isnan(rho_val):
                tri_lr = plt.Polygon(
                    [[col + 0.5, row - 0.5], [col + 0.5, row + 0.5], [col - 0.5, row + 0.5]],
                    closed=True, facecolor=cmap(norm(rho_val)), edgecolor='white', linewidth=0.5)
                ax.add_patch(tri_lr)

            if not np.isnan(r_val):
                c_text = 'white' if abs(r_val) > 0.5 else 'black'
                ax.text(col - 0.18, row - 0.15, f'{r_val:.2f}',
                        ha='center', va='center', fontsize=5.5, color=c_text, fontweight='medium')
            if not np.isnan(rho_val):
                c_text = 'white' if abs(rho_val) > 0.5 else 'black'
                ax.text(col + 0.18, row + 0.15, f'{rho_val:.2f}',
                        ha='center', va='center', fontsize=5.5, color=c_text, fontweight='medium')

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    ax.set_aspect('auto')


def save_heatmap(r_mat, rho_mat, row_labels, col_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_split_heatmap(ax, r_mat, rho_mat, row_labels, col_labels, title)
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Correlation', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    fig.savefig(out_path + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# TCR CV — halfsplit prediction per fold
# ─────────────────────────────────────────────────────────────────

def compute_tcr_cv_heatmaps():
    """Rows=models, Cols=folds, cell=(r, ρ) between predicted and actual across bins."""
    print("=== TCR CV heatmaps ===")
    for metric in METRICS:
        r_mat = np.full((len(TCR_MODELS), 5), np.nan)
        rho_mat = np.full((len(TCR_MODELS), 5), np.nan)
        for i, model in enumerate(TCR_MODELS):
            for fold in range(5):
                data = load_tcr_cv_fold(model, fold)
                if data is None: continue
                n_pos = int((data['y'] == 1).sum()); n_neg = int((data['y'] == 0).sum())
                n_bins = adaptive_n_bins(n_pos, n_neg)
                preds, actuals = predict_bins_halfsplit(data, metric, n_bins=n_bins)
                if len(preds) >= 3:
                    try:
                        r_mat[i, fold] = pearsonr(preds, actuals)[0]
                        rho_mat[i, fold] = spearmanr(preds, actuals)[0]
                    except Exception as _e_pr:
                        import sys as _s_pr
                        print(f"  ⚠ FALLBACK [subset_heatmaps CV]: model={model} fold={fold} metric={metric} pearsonr/spearmanr failed ({type(_e_pr).__name__}: {_e_pr}); leaving NaN", file=_s_pr.stderr, flush=True)
        row_labels = [MODEL_DISPLAY[m] for m in TCR_MODELS]
        col_labels = [f'F{i}' for i in range(5)]
        out = os.path.join(PANEL_DIR, f'fig3_tcr_cv_heatmap_{metric}_split')
        save_heatmap(r_mat, rho_mat, row_labels, col_labels,
                     f'TCR CV {MDISP[metric]} r/ρ', out)
        print(f'  {metric}: mean r={np.nanmean(r_mat):.3f}, mean ρ={np.nanmean(rho_mat):.3f}')


def compute_tcr_ct_heatmaps():
    print("\n=== TCR CT heatmaps (v3+v4 cal only) ===")
    for metric in METRICS:
        r_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), np.nan)
        rho_mat = np.full((len(TCR_MODELS), len(TCR_CT_SETS)), np.nan)
        for i, model in enumerate(TCR_MODELS):
            ct_data = {ts: load_tcr_ct(model, ts) for ts in TCR_CT_SETS}
            ct_data = {k: v for k, v in ct_data.items() if v is not None}
            for j, ts in enumerate(TCR_CT_SETS):
                if ts not in ct_data: continue
                preds, actuals = predict_bins_loo(ct_data, ts, metric, n_bins=8,
                                                  cal_sets=TCR_CT_CAL_SETS)
                if len(preds) >= 3:
                    try:
                        r_mat[i, j] = pearsonr(preds, actuals)[0]
                        rho_mat[i, j] = spearmanr(preds, actuals)[0]
                    except Exception as _e_pr2:
                        import sys as _s_pr2
                        print(f"  ⚠ FALLBACK [subset_heatmaps CT]: model={model} ts={ts} metric={metric} pearsonr/spearmanr failed ({type(_e_pr2).__name__}: {_e_pr2}); leaving NaN", file=_s_pr2.stderr, flush=True)
        row_labels = [MODEL_DISPLAY[m] for m in TCR_MODELS]
        col_labels = [TCR_CT_DISPLAY[ts] for ts in TCR_CT_SETS]
        out = os.path.join(PANEL_DIR, f'fig3_tcr_ct_heatmap_{metric}_split')
        save_heatmap(r_mat, rho_mat, row_labels, col_labels,
                     f'TCR CT {MDISP[metric]} r/ρ', out)
        print(f'  {metric}: mean r={np.nanmean(r_mat):.3f}, mean ρ={np.nanmean(rho_mat):.3f}')


def compute_bcr_cv_heatmaps():
    print("\n=== BCR CV heatmaps ===")
    for metric in METRICS:
        r_mat = np.full((len(BCR_MODELS_CV), 5), np.nan)
        rho_mat = np.full((len(BCR_MODELS_CV), 5), np.nan)
        for i, model in enumerate(BCR_MODELS_CV):
            for fold in range(5):
                data = load_bcr_cv_fold(model, fold)
                if data is None: continue
                n_pos = int((data['y'] == 1).sum()); n_neg = int((data['y'] == 0).sum())
                n_bins = adaptive_n_bins(n_pos, n_neg)
                preds, actuals = predict_bins_halfsplit(data, metric, n_bins=n_bins)
                if len(preds) >= 3:
                    try:
                        r_mat[i, fold] = pearsonr(preds, actuals)[0]
                        rho_mat[i, fold] = spearmanr(preds, actuals)[0]
                    except Exception as _e_pr:
                        import sys as _s_pr
                        print(f"  ⚠ FALLBACK [subset_heatmaps CV]: model={model} fold={fold} metric={metric} pearsonr/spearmanr failed ({type(_e_pr).__name__}: {_e_pr}); leaving NaN", file=_s_pr.stderr, flush=True)
        row_labels = [BCR_MODEL_DISPLAY[m] for m in BCR_MODELS_CV]
        col_labels = [f'F{i}' for i in range(5)]
        out = os.path.join(PANEL_DIR, f'fig3_bcr_cv_heatmap_{metric}_split')
        save_heatmap(r_mat, rho_mat, row_labels, col_labels,
                     f'BCR CV {MDISP[metric]} r/ρ', out)
        print(f'  {metric}: mean r={np.nanmean(r_mat):.3f}, mean ρ={np.nanmean(rho_mat):.3f}')


def compute_bcr_ct_heatmaps():
    print("\n=== BCR CT heatmaps ===")
    # Union of all CT sets
    all_cts = ['A1-A11', 'unseen', 'flu']  # fold4cal pipeline (no BNT162b2/guoyu — 100% training overlap)
    for metric in METRICS:
        r_mat = np.full((len(BCR_MODELS_CV), len(all_cts)), np.nan)
        rho_mat = np.full((len(BCR_MODELS_CV), len(all_cts)), np.nan)
        for i, model in enumerate(BCR_MODELS_CV):
            ts_list = BCR_CT_SETS.get(model, [])
            ct_data = {ts: load_bcr_ct(model, ts) for ts in ts_list}
            ct_data = {k: v for k, v in ct_data.items() if v is not None}
            for j, ts in enumerate(all_cts):
                if ts not in ct_data: continue
                preds, actuals = predict_bins_loo(ct_data, ts, metric, n_bins=8)
                if len(preds) >= 3:
                    try:
                        r_mat[i, j] = pearsonr(preds, actuals)[0]
                        rho_mat[i, j] = spearmanr(preds, actuals)[0]
                    except Exception as _e_pr2:
                        import sys as _s_pr2
                        print(f"  ⚠ FALLBACK [subset_heatmaps CT]: model={model} ts={ts} metric={metric} pearsonr/spearmanr failed ({type(_e_pr2).__name__}: {_e_pr2}); leaving NaN", file=_s_pr2.stderr, flush=True)
        row_labels = [BCR_MODEL_DISPLAY[m] for m in BCR_MODELS_CV]
        col_labels = [BCR_CT_DISPLAY[ts] for ts in all_cts]
        out = os.path.join(PANEL_DIR, f'fig3_bcr_ct_heatmap_{metric}_split')
        save_heatmap(r_mat, rho_mat, row_labels, col_labels,
                     f'BCR CT {MDISP[metric]} r/ρ', out)
        print(f'  {metric}: mean r={np.nanmean(r_mat):.3f}, mean ρ={np.nanmean(rho_mat):.3f}')


if __name__ == '__main__':
    compute_tcr_cv_heatmaps()
    compute_tcr_ct_heatmaps()
    compute_bcr_cv_heatmaps()
    compute_bcr_ct_heatmaps()
    print("\nDone. Generated 12 heatmap panels.")
