#!/usr/bin/env python3
"""Generate pooled CV prediction scatter panels for TCR and BCR.

Style: color=model, shape=metric (matching CT pooled scatter panels).

TCR CV: 5 models × 5 folds, within-fold halfsplit prediction.
BCR CV: XBCR-net × 5 folds, within-fold halfsplit prediction.

Protocol: split each fold's test+val data into cal/test halves by
alternating distance sort, run PAPE+vbias on cal half, predict test
half bins.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import (apply_publication_style, MODEL_COLORS, MODEL_DISPLAY,
                           BCR_MODEL_COLORS, BCR_MODEL_DISPLAY)
from calipper.general_evaluator import safe_metric
from calipper.core import adaptive_n_bins, VBIAS_BETA_LAM
from PAPE.pape_core import (estimate_importance_weights, fit_weighted_calibration,
                             apply_calibration, estimate_metric as pape_eq4)
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE

apply_publication_style()

PANEL_DIR = os.path.join(FIG_DIR, 'fig3', DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')

METRICS = ['aucroc', 'ap', 'f1']
MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}
METRIC_MARKERS = {'aucroc': 'o', 'ap': '^', 'f1': 's'}
RESIDUAL_LAM = VBIAS_BETA_LAM  # from s2dd module (0.05)
PW, PH = 3.0, 2.5

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
LABEL_COL = {'nettcr': ('binder', 'prediction'), 'atm_tcr': ('y_true', 'y_prob'),
             'blosum_rf': ('binder', 'prediction'), 'ergo_ii': ('y_true', 'y_prob'),
             'tcrbert': ('y_true', 'y_prob')}


def load_tcr_cv_fold(model, fold):
    """Load test+val combined predictions + cached distances for one CV fold."""
    fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
    test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
    if not os.path.exists(test_path):
        return None, None
    parts = [pd.read_csv(test_path)]
    for vname in ['val_predictions_with_label.csv', 'val_predictions.csv']:
        vp = os.path.join(fold_dir, vname)
        if os.path.exists(vp):
            parts.append(pd.read_csv(vp))
            break
    df = pd.concat(parts, ignore_index=True)

    if DIST_TYPE == 'blosum-sqrt':
        suffixes = ['_blosumsqrt_dist']
    else:
        suffixes = ['_combined_dist', '_dist']
    for suffix in suffixes:
        dpath = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}{suffix}.npy')
        if os.path.exists(dpath):
            d = np.load(dpath)
            n = min(len(d), len(df))
            return df.iloc[:n], d[:n]
    return None, None


def halfsplit_predict(y, p, d, metric, n_bins=8):
    """Within-fold halfsplit: alternate by distance → cal/test, PAPE+vbias predict."""
    si = np.argsort(d)
    cal_idx, test_idx = si[::2], si[1::2]
    cal_y, cal_p, cal_d = y[cal_idx], p[cal_idx], d[cal_idx]
    test_y, test_p, test_d = y[test_idx], p[test_idx], d[test_idx]

    # DRE + calibrator
    w, _, _ = estimate_importance_weights(
        np.stack([cal_d, cal_p], 1), np.stack([test_d, test_p], 1))
    cal_model = fit_weighted_calibration(cal_p, cal_y, w)
    c_cal = apply_calibration(cal_model, cal_p)
    c_test = apply_calibration(cal_model, test_p)

    # Cal bins
    si_c = np.argsort(cal_d)
    bs = len(si_c) // n_bins
    if bs < 30:
        return []
    bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
    for i in range(n_bins):
        s = i * bs; e = len(si_c) if i == n_bins - 1 else (i + 1) * bs
        idx = si_c[s:e]
        a_m = safe_metric(metric, cal_y[idx], cal_p[idx])
        p_m = pape_eq4(c_cal[idx], cal_p[idx], metric, threshold=0.5)
        if not np.isnan(a_m) and not np.isnan(p_m):
            bin_d.append(cal_d[idx].mean())
            bin_mp.append(cal_p[idx].mean())
            bin_actual.append(a_m)
            bin_pape.append(p_m)

    if len(bin_actual) < 4:
        return []

    bin_d = np.array(bin_d); bin_mp = np.array(bin_mp)
    bin_actual = np.array(bin_actual); bin_pape = np.array(bin_pape)

    # Joint curve + β vbias correction on PAPE residuals (v2.7)
    from calipper.core import fit_best_curve, predict_best_curve
    residual = bin_actual - bin_pape
    if len(residual) >= 4:
        fit_res = fit_best_curve(bin_d, bin_mp, residual, lam=RESIDUAL_LAM)
    else:
        fit_res = {'params': None}

    # Predict test half bins
    si_t = np.argsort(test_d)
    bs_t = len(si_t) // n_bins
    if bs_t < 20:
        return []
    results = []
    for i in range(n_bins):
        s = i * bs_t; e = len(si_t) if i == n_bins - 1 else (i + 1) * bs_t
        idx = si_t[s:e]
        actual = safe_metric(metric, test_y[idx], test_p[idx])
        if np.isnan(actual):
            continue
        pape_bin = pape_eq4(c_test[idx], test_p[idx], metric, threshold=0.5)
        if fit_res['params'] is not None:
            correction = float(predict_best_curve(
                fit_res, np.array([test_d[idx].mean()]),
                np.array([test_p[idx].mean()]))[0])
        else:
            correction = 0.0
        results.append((float(np.clip(pape_bin + correction, 0, 1)), actual))
    return results


def make_pooled_scatter(pooled, model_colors, model_display, model_order,
                        title, out_path):
    """Generate pooled scatter with color=model, shape=metric."""
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    all_p, all_a = [], []
    plotted_models, plotted_metrics = set(), set()

    for m in METRICS:
        for mdl in model_order:
            pts = [(e[0], e[1]) for e in pooled[m] if e[2] == mdl]
            if not pts:
                continue
            p_arr = [x[0] for x in pts]
            a_arr = [x[1] for x in pts]
            ax.scatter(a_arr, p_arr,
                       c=model_colors[mdl], marker=METRIC_MARKERS[m],
                       s=25, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=5)
            all_p.extend(p_arr); all_a.extend(a_arr)
            plotted_models.add(mdl)
            plotted_metrics.add(m)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, zorder=1)

    if all_p:
        r, _ = pearsonr(all_p, all_a)
        mae = np.mean(np.abs(np.array(all_p) - np.array(all_a)))
        ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}',
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    handles = []
    for mdl in model_order:
        if mdl in plotted_models:
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=model_colors[mdl], markersize=5,
                                  label=model_display[mdl]))
    for m in METRICS:
        if m in plotted_metrics:
            handles.append(Line2D([0], [0], marker=METRIC_MARKERS[m], color='w',
                                  markerfacecolor='gray', markersize=5,
                                  label=MDISP[m]))
    ax.legend(handles=handles, fontsize=5, loc='lower right', ncol=2,
              handletextpad=0.3, columnspacing=0.5)

    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    # Identical 0–1 square axes across all four scatter panels (d–g) so the
    # y=x diagonal and point clouds are directly comparable.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # full grid, behind data (Nature style)
    ax.tick_params(direction='out')
    ax.set_title(title, fontweight='bold', fontsize=12)
    fig.savefig(out_path + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    if all_p:
        print(f'  Pooled R={r:.3f}, MAE={mae:.3f}, n={len(all_p)}')


# ═══════════════════════════════════════════════════════════
# TCR CV dataset-level scatter (halfsplit → predict_metric on test half)
# ═══════════════════════════════════════════════════════════
print("=== TCR CV pooled scatter (dataset-level) ===")
from calipper.core import predict_metric as s2dd_predict
tcr_cv_pooled = {m: [] for m in METRICS}  # (predicted, actual, model)

for model in TCR_MODELS:
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    n_fold_ok = 0
    for fold in range(5):
        df, d = load_tcr_cv_fold(model, fold)
        if df is None or d is None:
            continue
        lc_act = lc if lc in df.columns else ('binder' if 'binder' in df.columns else 'y_true')
        pc_act = pc if pc in df.columns else ('prediction' if 'prediction' in df.columns else 'y_prob')
        y = df[lc_act].values.astype(int)
        p = df[pc_act].values.astype(float)

        # Halfsplit by distance
        si = np.argsort(d)
        cal_idx, test_idx = si[::2], si[1::2]
        cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
        test_p, test_d, test_y = p[test_idx], d[test_idx], y[test_idx]

        # Dataset-level prediction on test half
        result = s2dd_predict(cal_data, test_p, test_d, metrics=METRICS)
        for m in METRICS:
            actual_m = safe_metric(m, test_y, test_p)
            pred_m = result['estimated'].get(m, np.nan)
            if not np.isnan(actual_m) and not np.isnan(pred_m):
                tcr_cv_pooled[m].append((pred_m, actual_m, model))
        n_fold_ok += 1

    n_pts = sum(len([e for e in tcr_cv_pooled[m] if e[2] == model]) for m in METRICS)
    print(f'  {MODEL_DISPLAY[model]}: {n_fold_ok} folds, {n_pts} points')

make_pooled_scatter(
    tcr_cv_pooled, MODEL_COLORS, MODEL_DISPLAY, TCR_MODELS,
    f'TCR CV ({len(TCR_MODELS)} models)',
    os.path.join(PANEL_DIR, 'fig3_tcr_cv_scatter_pooled'))


# ═══════════════════════════════════════════════════════════
# BCR CV pooled scatter (ALL 5 models — 2-pathogen combined)
# ═══════════════════════════════════════════════════════════
print("\n=== BCR CV pooled scatter (5 models) ===")
bcr_cv_pooled = {m: [] for m in METRICS}

BCR_CV_MODELS = {
    'xbcr_net':  os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv'),
    'deepaai':   os.path.join(RESULTS, 'deepaai', 'combined_bind_ab_cv'),
    'mambaaai':  os.path.join(RESULTS, 'mambaaai', 'combined_bind_ab_cv'),
    'mint':      os.path.join(RESULTS, 'mint', 'combined_bind_ab_cv'),
    'rleaai':    os.path.join(RESULTS, 'rleaai', 'combined_bind_ab_cv'),
}

for mdl, cv_dir in BCR_CV_MODELS.items():
    n_fold_ok = 0
    for fold in range(5):
        test_path = os.path.join(cv_dir, f'fold{fold}', 'test.csv')
        if not os.path.exists(test_path):
            continue
        te = pd.read_csv(test_path)

        if 'distance' not in te.columns:
            continue

        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            npy_path = os.path.join(TCR_CACHE, f'{mdl}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
            if os.path.exists(npy_path):
                d = np.load(npy_path).astype(float)[:len(te)]
            else:
                d = te['distance'].values.astype(float)
        else:
            d = te['distance'].values.astype(float)
        y = te['rbd'].values.astype(int)
        p = te['pred_prob'].values.astype(float) if 'pred_prob' in te.columns else te['output'].values.astype(float)

        # Halfsplit → dataset-level prediction on test half
        si = np.argsort(d)
        cal_idx, test_idx = si[::2], si[1::2]
        cal_data = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
        test_p, test_d, test_y = p[test_idx], d[test_idx], y[test_idx]

        result = s2dd_predict(cal_data, test_p, test_d, metrics=METRICS)
        for m in METRICS:
            actual_m = safe_metric(m, test_y, test_p)
            pred_m = result['estimated'].get(m, np.nan)
            if not np.isnan(actual_m) and not np.isnan(pred_m):
                bcr_cv_pooled[m].append((pred_m, actual_m, mdl))
        n_fold_ok += 1

    n_pts = sum(len([e for e in bcr_cv_pooled[m] if e[2] == mdl]) for m in METRICS)
    print(f'  {BCR_MODEL_DISPLAY[mdl]}: {n_fold_ok} folds, {n_pts} points')

make_pooled_scatter(
    bcr_cv_pooled, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY, list(BCR_CV_MODELS.keys()),
    f'BCR CV ({len(BCR_CV_MODELS)} models)',
    os.path.join(PANEL_DIR, 'fig3_bcr_cv_scatter_pooled'))

print("\nDone.")
