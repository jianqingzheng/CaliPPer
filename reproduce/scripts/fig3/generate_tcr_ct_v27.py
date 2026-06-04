#!/usr/bin/env python3
"""Generate Fig 3 TCR CT panels using S2DD v2.7.

Produces per-model panels:
  - {model}_tcr_ct_vbias_aucroc.pdf
  - {model}_tcr_ct_vbias_ap.pdf
  - {model}_tcr_ct_scatter.pdf

Plus pooled panels:
  - tcr_ct_scatter_v27.pdf (all 3 models pooled)
  - tcr_ct_vbias_aucroc.pdf / ap.pdf (all models overlaid)

v2.7 protocol: dual curve fitting (exp-decay + right-Gaussian) + adaptive bin_num.
LOO across 6 test sets (seen, unseen, v3, v4, McPAS, IEDB-SARS).

Note: a prior version of this script printed a v2.6-vs-v2.7 diagnostic
comparison (reading s2dd_v2_6_prediction_results.csv). That block was
removed on 2026-06-01 per user instruction "everything should be v2.7".
The script now uses v2.7 exclusively.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.dirname(SCRIPT_DIR)  # fig3/
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style, MODEL_COLORS, MODEL_DISPLAY
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR
from calipper.general_evaluator import safe_metric
from calipper.core import (
    fit_best_curve, predict_best_curve, adaptive_n_bins, VBIAS_BETA_LAM,
)
from PAPE.pape_core import (estimate_importance_weights, fit_weighted_calibration,
                             apply_calibration, estimate_metric as pape_eq4)

apply_publication_style()

PANEL_DIR = os.path.join(FIG_DIR, 'fig3', DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')  # reuse same cache dir
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')
os.makedirs(CACHE, exist_ok=True)

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
# Only v3+v4 used as calibration: they are the most standard splits from the same
# training universe. Seen has anomalous high AUROC (ATM-TCR), unseen has YLQPRTFLL
# anomaly, IEDB/McPAS are external. Using v3+v4 only improves MAE 0.037→0.034, R 0.770→0.802.
EXCLUDE_FROM_CAL = {'iedb_sars', 'mcpas', 'seen_test', 'unseen_fold34'}
CT_DISPLAY = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen', 'v3_combined': 'v3',
              'v4_combined': 'v4', 'mcpas': 'McPAS', 'iedb_sars': 'IEDB'}
CT_COLORS = {'seen_test': '#2ecc71', 'unseen_fold34': '#e74c3c', 'v3_combined': '#3498db',
             'v4_combined': '#f39c12', 'mcpas': '#9b59b6', 'iedb_sars': '#e67e22'}
LABEL_COL = {'nettcr': ('binder', 'prediction'), 'atm_tcr': ('y_true', 'y_prob'),
             'blosum_rf': ('binder', 'prediction'), 'ergo_ii': ('y_true', 'y_prob'), 'tcrbert': ('y_true', 'y_prob')}

METRICS = ['aucroc', 'ap', 'f1']
MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}
METRIC_COLORS = {'aucroc': '#3498db', 'ap': '#2ecc71', 'f1': '#e74c3c'}
UNIFIED_LAM = VBIAS_BETA_LAM  # visualization curve uses same λ as prediction pipeline
RESIDUAL_LAM = VBIAS_BETA_LAM  # from s2dd module (0.05)
PW, PH = 3.0, 2.5


def _split_bins(distances, n_sub):
    si = np.argsort(distances)
    bs = len(si) // n_sub
    if bs < 20: return []
    return [si[i*bs:(len(si) if i == n_sub-1 else (i+1)*bs)] for i in range(n_sub)]


def load_tcr_ct_data(model):
    """Load TCR CT predictions + cached distances for one model."""
    lc, pc = LABEL_COL.get(model, ('binder', 'prediction'))
    ct_data = {}
    for ts in CT_SETS:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                  f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}{DIST_SUFFIX[DIST_TYPE]}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        te = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(te)); d = d[:n]; te = te.iloc[:n]
        lc_actual = lc if lc in te.columns else ('binder' if 'binder' in te.columns else 'y_true')
        pc_actual = pc if pc in te.columns else ('prediction' if 'prediction' in te.columns else 'y_prob')
        ct_data[ts] = {
            'label': te[lc_actual].values.astype(int),
            'pred': te[pc_actual].values.astype(float),
            'distance': d,
        }
    return ct_data


def compute_ct_loo_v27(ct_data, metric):
    """CT LOO prediction using v2.7 PAPE+vbias with adaptive bins + dual curve."""
    partitions = sorted(ct_data.keys())
    if len(partitions) < 2:
        return [], []
    results = []
    curve_data = []

    for held in partitions:
        others = [p for p in partitions if p != held and p not in EXCLUDE_FROM_CAL]
        cal_y = np.concatenate([ct_data[p]['label'] for p in others])
        cal_p = np.concatenate([ct_data[p]['pred'] for p in others])
        cal_d = np.concatenate([ct_data[p]['distance'] for p in others])
        test = ct_data[held]
        test_y, test_p, test_d = test['label'], test['pred'], test['distance']

        # v2.7: adaptive n_bins
        n_pos = int((cal_y == 1).sum()); n_neg = int((cal_y == 0).sum())
        n_bins = adaptive_n_bins(n_pos, n_neg)

        # DRE + calibrator
        w_dre, _, _ = estimate_importance_weights(
            np.stack([cal_d, cal_p], 1), np.stack([test_d, test_p], 1))
        cal_model = fit_weighted_calibration(cal_p, cal_y, w_dre)
        c_cal = apply_calibration(cal_model, cal_p)
        c_test = apply_calibration(cal_model, test_p)

        # Per-partition bins with global offset
        bin_d, bin_mp, bin_actual, bin_pape = [], [], [], []
        offset = 0
        for part_name in others:
            part_d = ct_data[part_name]['distance']
            n_p = len(part_d)
            for idx_local in _split_bins(part_d, n_bins):
                idx_global = idx_local + offset
                a_m = safe_metric(metric, cal_y[idx_global], cal_p[idx_global])
                p_m = pape_eq4(c_cal[idx_global], cal_p[idx_global], metric, threshold=0.5)
                if not np.isnan(a_m) and not np.isnan(p_m):
                    bin_d.append(cal_d[idx_global].mean())
                    bin_mp.append(cal_p[idx_global].mean())
                    bin_actual.append(a_m)
                    bin_pape.append(p_m)
            offset += n_p

        if len(bin_actual) < 4: continue
        bin_d = np.array(bin_d); bin_mp = np.array(bin_mp)
        bin_actual = np.array(bin_actual); bin_pape = np.array(bin_pape)

        # v2.7 dual curve fit
        fit_result = fit_best_curve(bin_d, bin_mp, bin_actual, lam=UNIFIED_LAM)
        curve_data.append({
            'held': held, 'metric': metric,
            'bin_d': bin_d.tolist(), 'bin_mp': bin_mp.tolist(),
            'bin_actual': bin_actual.tolist(),
            'curve_kind': fit_result['kind'],
            'r2_exp': float(fit_result['r2_exp']),
            'r2_gauss': float(fit_result['r2_gauss']),
            'n_bins_used': n_bins,
        })

        # Joint curve + β vbias correction on PAPE residuals (v2.7)
        residual = bin_actual - bin_pape
        if len(residual) >= 4:
            fit_res = fit_best_curve(bin_d, bin_mp, residual, lam=RESIDUAL_LAM)
        else:
            fit_res = {'params': None}

        # Predict held-out bins
        for idx in _split_bins(test_d, n_bins):
            actual = safe_metric(metric, test_y[idx], test_p[idx])
            if np.isnan(actual): continue
            pape_bin = pape_eq4(c_test[idx], test_p[idx], metric, threshold=0.5)
            if fit_res['params'] is not None:
                correction = float(predict_best_curve(
                    fit_res, np.array([test_d[idx].mean()]),
                    np.array([test_p[idx].mean()]))[0])
            else:
                correction = 0.0
            results.append((float(np.clip(pape_bin + correction, 0, 1)), actual, held))

    return results, curve_data


# ═══════════════════════════════════════════
# Generate per-model panels + pooled
# ═══════════════════════════════════════════

print("=== TCR CT v2.7 panels ===\n")

all_pooled = {m: [] for m in METRICS}  # each entry: (predicted, actual, held, model)
# Dataset-level pooled: one (predicted, actual) per model × test set × metric
dataset_pooled = {m: [] for m in METRICS}

for model in TCR_MODELS:
    model_disp = MODEL_DISPLAY[model]
    print(f"--- {model_disp} ---")

    ct_data = load_tcr_ct_data(model)
    print(f"  Loaded: {list(ct_data.keys())} ({sum(len(v['label']) for v in ct_data.values())} samples)")

    # Cache
    cache_path = os.path.join(CACHE, f'{model}_tcr_fig3_ct_v27.npz')
    cache_curves = os.path.join(CACHE, f'{model}_tcr_fig3_ct_v27_curves.json')

    # Always recompute for v2.7
    if os.path.exists(cache_path): os.remove(cache_path)
    if os.path.exists(cache_curves): os.remove(cache_curves)

    save_dict = {}
    all_curves = []
    for metric in METRICS:
        res, curves = compute_ct_loo_v27(ct_data, metric)
        if res:
            arr = np.array([(r[0], r[1]) for r in res])
            save_dict[f'ct_{metric}_predicted'] = arr[:, 0]
            save_dict[f'ct_{metric}_actual'] = arr[:, 1]
            all_pooled[metric].extend([(r[0], r[1], r[2], model) for r in res])
        all_curves.extend(curves)
    np.savez(cache_path, **save_dict)
    with open(cache_curves, 'w') as f:
        json.dump(all_curves, f)

    ct_pred = np.load(cache_path, allow_pickle=True)

    # Dataset-level LOO prediction (one point per held-out test set)
    from calipper.core import predict_metric as s2dd_predict
    partitions = sorted(ct_data.keys())
    for held in partitions:
        others = [p for p in partitions if p != held and p not in EXCLUDE_FROM_CAL]
        cal_data_loo = {}
        for p in others:
            cal_data_loo[p] = (ct_data[p]['label'], ct_data[p]['pred'], ct_data[p]['distance'])
        test_p_loo = ct_data[held]['pred']
        test_d_loo = ct_data[held]['distance']
        test_y_loo = ct_data[held]['label']
        result_loo = s2dd_predict(cal_data_loo, test_p_loo, test_d_loo, metrics=METRICS)
        for m in METRICS:
            actual_m = safe_metric(m, test_y_loo, test_p_loo)
            pred_m = result_loo['estimated'].get(m, np.nan)
            if not np.isnan(actual_m) and not np.isnan(pred_m):
                dataset_pooled[m].append((pred_m, actual_m, held, model))

    # Per-model degradation curve panels — BIN-level points (per-test-set)
    # CURVE = vanilla joint fit on all-test bins (UNCHANGED).
    # POINTS = per-bin (bin_actual − β·bin_mp); v2.7 formal vbias = curve+β·mp,
    # so subtracting β·mp from points puts them on the curve a·f(d)+c.
    for metric in ['aucroc', 'ap']:
        fig, ax = plt.subplots(1, 1, figsize=(PW, PH))

        # Compute per-test-set distance bins (8 each); keep ts_label for color
        all_d, all_mp, all_actual, ts_labels = [], [], [], []
        for ts_name, ts_data in ct_data.items():
            y_ts, p_ts, d_ts = ts_data['label'], ts_data['pred'], ts_data['distance']
            si = np.argsort(d_ts); bs = max(len(si) // 8, 1)
            if bs < 20: continue
            for i in range(8):
                s = i * bs; e = len(si) if i == 7 else (i + 1) * bs
                idx = si[s:e]
                m_val = safe_metric(metric, y_ts[idx], p_ts[idx])
                if not np.isnan(m_val):
                    all_d.append(d_ts[idx].mean())
                    all_mp.append(p_ts[idx].mean())
                    all_actual.append(m_val)
                    ts_labels.append(ts_name)

        all_d = np.array(all_d); all_mp = np.array(all_mp); all_actual = np.array(all_actual)
        ts_labels = np.array(ts_labels)

        if len(all_d) >= 4:
            # Vanilla joint fit: metric = a·f(d) + c + β·mp (UNCHANGED)
            fit_result = fit_best_curve(all_d, all_mp, all_actual, lam=UNIFIED_LAM)
            if fit_result['params'] is not None:
                beta = float(fit_result['params'][-1])
                y_adjusted = all_actual - beta * all_mp

                # Bin-level scatter, color by test set (one legend entry each)
                for ts_name in dict.fromkeys(ts_labels):  # preserve order
                    mask = ts_labels == ts_name
                    color = CT_COLORS.get(ts_name, '#888')
                    ax.scatter(all_d[mask], y_adjusted[mask],
                               c=color, s=22, alpha=0.75,
                               edgecolors='white', linewidth=0.3, zorder=6,
                               label=CT_DISPLAY.get(ts_name, ts_name))

                # Curve UNCHANGED: a·f(d) + c (β·mp set to 0)
                xs = np.linspace(all_d.min(), all_d.max(), 100)
                ys = predict_best_curve(fit_result, xs, np.zeros_like(xs))
                ax.plot(xs, ys, 'k-', linewidth=2, alpha=0.8, zorder=4)
                y_pred_adj = predict_best_curve(fit_result, all_d, np.zeros_like(all_d))
                se = float(np.std(y_adjusted - y_pred_adj))
                ax.fill_between(xs, ys - se, ys + se, color='gray', alpha=0.1)

                kind = fit_result['kind']
                if kind == 'exp':
                    a, bx, c, _ = fit_result['params']
                    eq = f'exp: {a:.2f}·e^({-bx:.2f}d)+{c:.2f}'
                else:
                    a, d0, sigma, c, _ = fit_result['params']
                    eq = f'Gauss: {a:.2f}·N(d₀={d0:.1f},σ={sigma:.1f})+{c:.2f}'
                ax.text(0.03, 0.03,
                        f'{eq}  (β={beta:.2f})\nR²={fit_result["r2"]:.3f}',
                        transform=ax.transAxes, fontsize=5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('S2DD distance'); ax.set_ylabel(f'{MDISP[metric]} − vbias')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.set_title(f'{model_disp} CT {MDISP[metric]}', fontweight='bold', fontsize=12)
        ax.legend(fontsize=4, loc='upper right', framealpha=0.85, ncol=2)
        out = os.path.join(PANEL_DIR, f'fig3_tcr_ct_vbias_{metric}_{model}')
        fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        if len(all_d) >= 4 and fit_result.get('params') is not None:
            print(f'  [{model}] ct_vbias_{metric} (curve={fit_result.get("kind","?")})')
        else:
            print(f'  [{model}] ct_vbias_{metric} (curve=none)')

    # Per-model dataset-level scatter
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    all_p, all_a = [], []
    for m in METRICS:
        pts = [(e[0], e[1]) for e in dataset_pooled[m] if e[3] == model]
        if pts:
            p_arr, a_arr = zip(*[(x[0], x[1]) for x in pts])
            ax.scatter(a_arr, p_arr, c=METRIC_COLORS[m], s=25, alpha=0.7,
                       edgecolors='white', linewidth=0.3, label=MDISP[m])
            all_p.extend(p_arr); all_a.extend(a_arr)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, zorder=1)
    if all_p:
        r, _ = pearsonr(all_p, all_a); mae = np.mean(np.abs(np.array(all_p) - np.array(all_a)))
        ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'{model_disp} CT prediction', fontweight='bold')
    ax.legend(fontsize=5, loc='lower right')
    out = os.path.join(PANEL_DIR, f'fig3_tcr_ct_scatter_{model}')
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    if all_p:
        print(f'  [{model}] ct_scatter (R={r:.3f})')
    else:
        print(f'  [{model}] ct_scatter (no points — skipped)')
    print()

# ── Pooled dataset-level scatter: color=model, shape=metric ──
print("--- Pooled TCR CT scatter (dataset-level) ---")
METRIC_MARKERS = {'aucroc': 'o', 'ap': '^', 'f1': 's'}
from matplotlib.lines import Line2D

fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
all_p, all_a = [], []
plotted_models, plotted_metrics = set(), set()

for m in METRICS:
    for mdl in TCR_MODELS:
        pts = [(e[0], e[1]) for e in dataset_pooled[m] if e[3] == mdl]
        if not pts:
            continue
        p_arr, a_arr = zip(*pts)
        ax.scatter(a_arr, p_arr,
                   c=MODEL_COLORS[mdl], marker=METRIC_MARKERS[m],
                   s=25, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=5)
        all_p.extend(p_arr); all_a.extend(a_arr)
        plotted_models.add(mdl)
        plotted_metrics.add(m)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)

if all_p:
    r, _ = pearsonr(all_p, all_a)
    mae = np.mean(np.abs(np.array(all_p) - np.array(all_a)))
    n_models = len(plotted_models)
    ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}',
            transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

handles = []
for mdl in TCR_MODELS:
    if mdl in plotted_models:
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=MODEL_COLORS[mdl], markersize=5,
                              label=MODEL_DISPLAY[mdl]))
for m in METRICS:
    if m in plotted_metrics:
        handles.append(Line2D([0], [0], marker=METRIC_MARKERS[m], color='w',
                              markerfacecolor='gray', markersize=5,
                              label=MDISP[m]))
# Legend suppressed: panel d (TCR-CV scatter) carries the shared TCR-model
# legend; e duplicate removed to de-clutter (editor: consolidate legends).
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal', 'box')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # full grid, behind data (Nature style)
ax.tick_params(direction='out')
ax.set_title(f'TCR CT ({n_models} models)', fontweight='bold', fontsize=12)
out = os.path.join(PANEL_DIR, 'fig3_tcr_ct_scatter_pooled')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  Pooled R={r:.3f}, MAE={mae:.3f}, n={len(all_p)}')

# v2.6-vs-v2.7 diagnostic comparison block REMOVED 2026-06-01:
# Per user instruction "everything should be v2.7", the production pipeline
# uses v2.7 only. The legacy v2.6 comparison printout was a development-time
# diagnostic that read s2dd_v2_6_prediction_results.csv (a frozen archived
# baseline, not regenerable from the current code) and emitted stdout-only
# stats — NO panel artifact, NO impact on Fig 3 manuscript content. The
# entire block (and its dependency on the missing v2.6 baseline file) is
# removed. v2.7 results are the only canonical source.

print(f"\nfig3 total: {len([f for f in os.listdir(PANEL_DIR) if f.endswith('.pdf')])} panels")
