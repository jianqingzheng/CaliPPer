#!/usr/bin/env python3
"""Generate Fig 3 BCR CT panels for ALL 5 BCR models using S2DD v2.7.

Protocol (fold4-as-cal design, verified 2026-04-25):
  Model:    fold4 model (trained on fold0/1/2/3, 14,355 2-pathogen binding samples)
  Cal:      fold4 test (3,655 samples) — model-specific pred_prob
  Test:     A1-A11 (281), unseen (1,256), flu (1,226) — model-specific pred_prob
  Distance: sigma_C 3-chain + weighted_max_znorm from fold4_train

  Data-level prediction: predict_metric(cal_data, test_p, test_d)
    → ONE predicted AUROC/AP per test set per model
    → 15 points on scatter (5 models × 3 test sets)

  Degradation curves: raw metric vs distance per test set
    → fit_best_curve (exp-decay or Gaussian) with β·mp vbias correction

3 invariants:
  1. SAME MODEL: cal + test pred_prob from same model weights
  2. SAME DISTANCE: all from fold4_train sigma_C 3-chain
  3. SAME TRAINING: overlap removal against fold4_train

Panels per model:
  - fig3_bcr_ct_vbias_aucroc_{model}.pdf  (degradation curve)
  - fig3_bcr_ct_vbias_ap_{model}.pdf
  - fig3_bcr_ct_scatter_{model}.pdf        (per-model dataset-level scatter)
Pooled:
  - fig3_bcr_ct_scatter_pooled.pdf         (5-model dataset-level scatter)
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, DIST_SUBDIR, BCR_DIST_MODE, get_bcr_ct_distance

PANEL_DIR = os.path.join(FIG_DIR, 'fig3', DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)
from calipper.general_evaluator import safe_metric
from calipper.core import (
    fit_best_curve, predict_best_curve, adaptive_n_bins,
    predict_metric as s2dd_predict, VBIAS_BETA_LAM,
)
from PAPE.pape_core import (estimate_importance_weights, fit_weighted_calibration,
                             apply_calibration, estimate_metric as pape_eq4)

apply_publication_style()

# ── Config ──────────────────────────────────────────────────────────────
RESULTS = os.path.join(INPUT_DIR, 'results')
# fold4-as-cal pipeline (verified correct by auditor 2026-04-25)
FOLD4CAL_DIR = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
CACHE_DIR = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

METRICS = ['aucroc', 'ap']
MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}
METRIC_COLORS = {'aucroc': '#3498db', 'ap': '#2ecc71', 'f1': '#e74c3c'}
METRIC_MARKERS = {'aucroc': 'o', 'ap': '^', 'f1': 's'}
CT_COLORS = {'A1-A11': '#e74c3c', 'unseen': '#f39c12', 'flu': '#3498db'}
UNIFIED_LAM = VBIAS_BETA_LAM  # 0.05 for prediction
RESIDUAL_LAM = VBIAS_BETA_LAM  # vbias residual fit (curve+β on actual − PAPE)
TEST_SETS = ['A1-A11', 'unseen', 'flu']
MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
# style_config uses xbcr_net as key; map for color/display lookup
_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
              'mint': 'mint', 'rleaai': 'rleaai'}
N_BINS_DEFAULT = 8
MIN_SAMPLES = 20
PW, PH = 3.0, 2.5


def load_fold4cal_data(model):
    """Load model-specific cal + test data from fold4-as-cal pipeline.

    Returns: cal_dict, test_dict
      cal_dict = {'fold4_test': (y, pred, dist)}
      test_dict = {ts_name: {'label': y, 'pred': p, 'distance': d}}
    """
    cal_path = os.path.join(FOLD4CAL_DIR, model, 'cal_predictions.csv')
    if not os.path.exists(cal_path):
        return None, None

    cal = pd.read_csv(cal_path)
    model_dir = os.path.join(FOLD4CAL_DIR, model)
    if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
        cal_d = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
    else:
        cal_d = cal['distance'].values.astype(float)
    cal_dict = {'fold4_test': (cal['rbd'].values.astype(int),
                                cal['pred_prob'].values.astype(float),
                                cal_d)}

    test_dict = {}
    for ts in TEST_SETS:
        ts_path = os.path.join(FOLD4CAL_DIR, model, f'{ts}_predictions.csv')
        if not os.path.exists(ts_path):
            continue
        df = pd.read_csv(ts_path)
        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            ts_d = get_bcr_ct_distance(df, model_dir, ts)
        else:
            ts_d = df['distance'].values.astype(float)
        test_dict[ts] = {
            'label': df['rbd'].values.astype(int),
            'pred': df['pred_prob'].values.astype(float),
            'distance': ts_d,
        }

    return cal_dict, test_dict


def _split_dist_indices(distances, n_sub=N_BINS_DEFAULT):
    si = np.argsort(distances)
    bs = len(si) // n_sub
    if bs < MIN_SAMPLES: return []
    return [si[i*bs:(len(si) if i == n_sub-1 else (i+1)*bs)] for i in range(n_sub)]


# ═══════════════════════════════════════════
# Generate panels for each BCR model
# ═══════════════════════════════════════════
from matplotlib.lines import Line2D

bcr_dataset_pooled = {m: [] for m in METRICS}

for model in MODELS:
    model_display = BCR_MODEL_DISPLAY.get(_STYLE_KEY.get(model, model), model)
    print(f"\n=== {model_display} ===")

    cal_dict, test_dict = load_fold4cal_data(model)
    if cal_dict is None or not test_dict:
        print(f"  SKIP: no fold4cal data")
        continue

    cal_y, cal_p, cal_d = cal_dict['fold4_test']
    cal_auc = safe_metric('aucroc', cal_y, cal_p)
    print(f"  Cal: n={len(cal_y)}, prev={cal_y.mean():.3f}, AUROC={cal_auc:.3f}")
    for ts, td in test_dict.items():
        auc = safe_metric('aucroc', td['label'], td['pred'])
        print(f"  {ts}: n={len(td['label'])}, AUROC={auc:.3f}")

    # ── Data-level prediction: fold4 cal → predict each test set ──
    print(f"\n  Data-level prediction (fold4-as-cal):")
    for metric in METRICS:
        result = s2dd_predict(cal_dict,
                               np.concatenate([td['pred'] for td in test_dict.values()]),
                               np.concatenate([td['distance'] for td in test_dict.values()]),
                               metrics=[metric])
        # But we need per-test-set predictions, not concatenated
        # Run separately per test set
        for ts, td in test_dict.items():
            r = s2dd_predict(cal_dict, td['pred'], td['distance'], metrics=[metric])
            actual = safe_metric(metric, td['label'], td['pred'])
            predicted = r['estimated'][metric]
            err = abs(predicted - actual)
            bcr_dataset_pooled[metric].append((predicted, actual, ts, model))
            print(f"    {ts} {MDISP[metric]}: actual={actual:.3f}, pred={predicted:.3f}, err={err:.3f}")

    # ── Degradation curve panels — BIN-level points ──
    # CURVE = vanilla joint fit on all-test+cal bins (UNCHANGED).
    # POINTS = per-bin (bin_actual − β·bin_mp); subtracting β·mp puts points
    # on the curve a·f(d)+c since vanilla fit is metric=a·f(d)+c+β·mp.
    for metric in ['aucroc', 'ap']:
        fig, ax = plt.subplots(1, 1, figsize=(PW, PH))

        # Bin data from ALL test sets + cal (curve + bin-level scatter)
        all_d, all_mp, all_actual, ts_labels = [], [], [], []
        for ts_name, ts_data in list(test_dict.items()) + [('cal', {'label': cal_y, 'pred': cal_p, 'distance': cal_d})]:
            y_ts, p_ts, d_ts = ts_data['label'], ts_data['pred'], ts_data['distance']
            for idx in _split_dist_indices(d_ts, n_sub=8):
                m_val = safe_metric(metric, y_ts[idx], p_ts[idx])
                if not np.isnan(m_val):
                    all_d.append(d_ts[idx].mean())
                    all_mp.append(p_ts[idx].mean())
                    all_actual.append(m_val)
                    ts_labels.append(ts_name)
        all_d = np.array(all_d); all_mp = np.array(all_mp); all_actual = np.array(all_actual)
        ts_labels = np.array(ts_labels)

        if len(all_d) >= 4:
            fit_result = fit_best_curve(all_d, all_mp, all_actual, lam=UNIFIED_LAM)
            if fit_result['params'] is not None:
                beta = float(fit_result['params'][-1])
                y_adjusted = all_actual - beta * all_mp

                # Bin-level scatter, color by test set (cal in grey)
                cal_color = '#95a5a6'
                for ts_name in dict.fromkeys(ts_labels):
                    mask = ts_labels == ts_name
                    color = cal_color if ts_name == 'cal' else CT_COLORS.get(ts_name, '#888')
                    ax.scatter(all_d[mask], y_adjusted[mask],
                               c=color, s=22, alpha=0.75,
                               edgecolors='white', linewidth=0.3, zorder=6,
                               label=ts_name)

                xs = np.linspace(all_d.min(), all_d.max(), 100)
                ys = predict_best_curve(fit_result, xs, np.zeros_like(xs))
                ax.plot(xs, ys, 'k-', linewidth=2, alpha=0.8, zorder=4)
                se = float(np.std(y_adjusted
                                  - predict_best_curve(fit_result, all_d, np.zeros_like(all_d))))
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
                        transform=ax.transAxes, fontsize=4.5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='#ccc', alpha=0.9, linewidth=0.3))

        ax.set_xlabel('S2DD distance')
        ax.set_ylabel(f'{MDISP[metric]} − vbias')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.set_title(f'{model_display} CT {MDISP[metric]}', fontweight='bold', fontsize=12)
        ax.legend(fontsize=5, loc='upper right', framealpha=0.85)

        out = os.path.join(PANEL_DIR, f'fig3_bcr_ct_vbias_{metric}_{model}')
        fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: ct_vbias_{metric}')

    # ── Per-model dataset-level scatter ──
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    all_p, all_a = [], []
    for m in METRICS:
        pts = [(e[0], e[1]) for e in bcr_dataset_pooled[m] if e[3] == model]
        if pts:
            p_arr, a_arr = zip(*pts)
            ax.scatter(a_arr, p_arr, c=METRIC_COLORS[m], s=25, alpha=0.7,
                       edgecolors='white', linewidth=0.3, label=MDISP[m])
            all_p.extend(p_arr); all_a.extend(a_arr)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, zorder=1)
    if all_p:
        r, _ = pearsonr(all_p, all_a); mae = np.mean(np.abs(np.array(all_p) - np.array(all_a)))
        ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'{model_display} CT prediction', fontweight='bold')
    ax.legend(fontsize=5, loc='lower right')

    out = os.path.join(PANEL_DIR, f'fig3_bcr_ct_scatter_{model}')
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: ct_scatter')


# ═══════════════════════════════════════════
# Pooled BCR dataset-level scatter: color=model, shape=metric
# ═══════════════════════════════════════════
print("\n--- Pooled BCR CT scatter (dataset-level, fold4-as-cal) ---")
fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
all_p, all_a = [], []

plotted_models = set()
plotted_metrics = set()
for m in METRICS:
    for mdl in MODELS:
        pts = [(e[0], e[1]) for e in bcr_dataset_pooled[m] if e[3] == mdl]
        if not pts:
            continue
        p_arr, a_arr = zip(*pts)
        ax.scatter(a_arr, p_arr,
                   c=BCR_MODEL_COLORS[_STYLE_KEY.get(mdl, mdl)], marker=METRIC_MARKERS[m],
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
for mdl in MODELS:
    if mdl in plotted_models:
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=BCR_MODEL_COLORS[_STYLE_KEY.get(mdl, mdl)], markersize=5,
                              label=BCR_MODEL_DISPLAY[_STYLE_KEY.get(mdl, mdl)]))
for m in METRICS:
    if m in plotted_metrics:
        handles.append(Line2D([0], [0], marker=METRIC_MARKERS[m], color='w',
                              markerfacecolor='gray', markersize=5,
                              label=MDISP[m]))
# Legend suppressed: panel f (BCR-CV scatter) carries the shared BCR-model
# legend; g duplicate removed to de-clutter (editor: consolidate legends).
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal', 'box')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # full grid, behind data (Nature style)
ax.tick_params(direction='out')
_n_models_title = locals().get('n_models', len(plotted_models) if 'plotted_models' in locals() else 0)
ax.set_title(f'BCR CT ({_n_models_title} models)', fontweight='bold', fontsize=12)
out = os.path.join(PANEL_DIR, 'fig3_bcr_ct_scatter_pooled')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
_r = locals().get('r', float('nan'))
_mae = locals().get('mae', float('nan'))
_all_p = locals().get('all_p', [])
print(f'  Pooled R={_r:.3f}, MAE={_mae:.3f}, n={len(_all_p)}')


print(f"\n=== Fig 3 BCR CT panels complete (fold4-as-cal design) ===")
n = len([f for f in os.listdir(PANEL_DIR) if f.endswith('.pdf') and 'bcr_ct' in f])
print(f"  BCR CT panels: {n}")
