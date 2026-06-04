#!/usr/bin/env python3
"""Fig 5 panel: Combined TCR + BCR recalibration dumbbell (10 models).

Two subplots side-by-side:
  Left:  5 TCR models (v3+v4 cal → 4 test sets, mean AUROC before/after)
  Right: 5 BCR models (per-variant LOO, pooled AUROC before/after)

Output: fig5_combined_recal_dumbbell.pdf / .png
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
                           BCR_MODEL_COLORS, BCR_MODEL_DISPLAY, DPI)
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE, get_bcr_ct_distance
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')

# ═══════════════════════════════════════════
# TCR: v3+v4 calibration → 4 test sets
# ═══════════════════════════════════════════
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_CAL_SETS = ['v3_combined', 'v4_combined']
TCR_TEST_SETS = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
TCR_ALL_SETS = TCR_CAL_SETS + TCR_TEST_SETS
TCR_STYLE_KEY = {m: m for m in TCR_MODELS}

# BCR: per-variant LOO
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
MIN_SAMPLES = 30


def compute_tcr_recal():
    """Returns dict: model -> (mean_auroc_before, mean_auroc_after)."""
    results = {}
    for model in TCR_MODELS:
        ct_model = {}
        for ts in TCR_ALL_SETS:
            # Predictions: results/{model}/cross_test_logdist/predictions/{ts}_predictions_with_label.csv
            pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                     f'{ts}_predictions_with_label.csv')
            # Distances: results/fig2_cache/{model}_ct_{ts}{DIST_SUFFIX}
            # TCR recalibration uses BLOSUM-sqrt (optimal for recalibration)
            tcr_dist_suffix = DIST_SUFFIX.get('blosum-sqrt', DIST_SUFFIX[DIST_TYPE])
            dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{tcr_dist_suffix}')
            if not os.path.exists(pred_path) or not os.path.exists(dist_path):
                continue
            df = pd.read_csv(pred_path)
            d = np.load(dist_path)
            n = min(len(d), len(df))
            lc = 'binder' if 'binder' in df.columns else 'y_true'
            pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
            ct_model[ts] = {'y': df[lc].values[:n].astype(int),
                            'p': df[pc].values[:n].astype(float),
                            'd': d[:n].astype(float)}

        if 'v3_combined' not in ct_model or 'v4_combined' not in ct_model:
            print(f"  [TCR] {model}: missing cal sets, skipping")
            continue

        cal_data = {s: (ct_model[s]['y'], ct_model[s]['p'], ct_model[s]['d'])
                    for s in TCR_CAL_SETS}
        ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(cal_data)

        aucs_before, aucs_after = [], []
        for ts in TCR_TEST_SETS:
            if ts not in ct_model:
                continue
            test = ct_model[ts]
            cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                        ppv_p, npv_p, pp, pn, prev=cal_prev)
            auc_b = safe_metric('aucroc', test['y'], test['p'])
            auc_a = safe_metric('aucroc', test['y'], cal_s)
            aucs_before.append(auc_b)
            aucs_after.append(auc_a)

        if aucs_before:
            results[model] = (np.mean(aucs_before), np.mean(aucs_after))
            delta = results[model][1] - results[model][0]
            print(f"  [TCR] {MODEL_DISPLAY[model]}: {results[model][0]:.3f}→{results[model][1]:.3f} Δ={delta:+.3f}")

    return results


def compute_bcr_recal():
    """Returns dict: model -> (pooled_auroc_before, pooled_auroc_after)."""
    results = {}
    for model in BCR_MODELS:
        display = BCR_MODEL_DISPLAY.get(BCR_STYLE_KEY[model], model)
        model_dir = os.path.join(BCR_FOLD4CAL, model)
        if not os.path.isdir(model_dir):
            print(f"  [BCR] {display}: no data dir, skipping")
            continue

        # Load and pool
        cal_path = os.path.join(model_dir, "cal_predictions.csv")

        if not os.path.exists(cal_path):

            print(f"  [BCR] {model}: cal_predictions.csv missing; skipping")

            continue

        cal = pd.read_csv(cal_path)
        if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
            cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
        cal['source'] = 'fold4_test'
        parts = [cal]
        for ts in ['A1-A11', 'unseen', 'flu']:
            fp = os.path.join(model_dir, f'{ts}_predictions.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            if BCR_DIST_MODE[DIST_TYPE] == 'npy_sidecar':
                df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
            df['source'] = ts
            if 'data_source' not in df.columns:
                df['data_source'] = 'flu' if ts == 'flu' else 'sars'
            parts.append(df)
        pooled = pd.concat(parts, ignore_index=True)

        # Per-variant LOO within each domain
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
                cal_data = {'cal': (cal_y,
                                    cal_sub['pred_prob'].values.astype(float),
                                    cal_sub['distance'].values.astype(float))}
                test_p = test_sub['pred_prob'].values.astype(float)
                test_d = test_sub['distance'].values.astype(float)
                ppv_p, npv_p, pp, pn, _bcr_cal_prev = fit_recalibration(cal_data)
                # BCR per-variant LOO: use test_prev (default) because
                # per-variant prevalence varies widely (0.0 to 0.5).
                # cal_prev ≈ 0.48 would be wrong for zero-binder variants.
                cal_s = apply_recalibration(test_y, test_p, test_d,
                                            ppv_p, npv_p, pp, pn)
                all_y.extend(test_y.tolist())
                all_raw.extend(test_p.tolist())
                all_cal.extend(cal_s.tolist())

        if all_y:
            orig = safe_metric('aucroc', np.array(all_y), np.array(all_raw))
            recal = safe_metric('aucroc', np.array(all_y), np.array(all_cal))
            results[model] = (orig, recal)
            print(f"  [BCR] {display}: {orig:.3f}→{recal:.3f} Δ={recal-orig:+.3f} (n={len(all_y)})")

    return results


# ═══════════════════════════════════════════
# Run both
# ═══════════════════════════════════════════
print("Computing TCR recalibration...")
tcr_results = compute_tcr_recal()
print("\nComputing BCR recalibration...")
bcr_results = compute_bcr_recal()

# ═══════════════════════════════════════════
# Plot: top-bottom dumbbell (models as rows, AUROC on x-axis)
# ═══════════════════════════════════════════
fig, (ax_tcr, ax_bcr) = plt.subplots(2, 1, figsize=(4.5, 4.5),
                                      gridspec_kw={'hspace': 0.4})

def plot_dumbbell(ax, results, style_key_map, color_map, display_map, title):
    """Plot dumbbell on given axis, sorted by delta (largest on top)."""
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1][1] - x[1][0],
                          reverse=True)
    yp = np.arange(len(sorted_items))[::-1]

    for i, (model, (before, after)) in enumerate(sorted_items):
        sk = style_key_map.get(model, model)
        color = color_map.get(sk, '#888888')
        delta = after - before

        # Dumbbell line
        ax.plot([before, after], [yp[i], yp[i]], color=color, linewidth=2.5,
                solid_capstyle='round', alpha=0.6)
        # Before (open circle)
        ax.scatter(before, yp[i], color='white', edgecolor=color, s=40,
                   zorder=5, linewidth=1.0)
        # After (filled circle)
        ax.scatter(after, yp[i], color=color, s=45, zorder=5,
                   edgecolor='white', linewidth=0.5)
        # Delta label
        label_x = max(before, after) + 0.008
        ax.text(label_x, yp[i], f'{delta:+.3f}', va='center', fontsize=7,
                color=color, fontweight='bold')

    ax.axvline(0.5, color='gray', linewidth=0.4, linestyle=':', alpha=0.5)
    ax.set_yticks(yp)
    ax.set_yticklabels([display_map.get(style_key_map.get(m, m), m)
                        for m, _ in sorted_items], fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=10, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shared x range across both subplots — set later
    return [v for b, a in results.values() for v in (b, a)]


# TCR subplot (top)
tcr_vals = plot_dumbbell(ax_tcr, tcr_results, TCR_STYLE_KEY, MODEL_COLORS,
                         MODEL_DISPLAY, 'TCR–epitope (5 models)')
ax_tcr.set_xticklabels([])  # hide x labels on top subplot

# BCR subplot (bottom)
bcr_vals = plot_dumbbell(ax_bcr, bcr_results, BCR_STYLE_KEY, BCR_MODEL_COLORS,
                         BCR_MODEL_DISPLAY, 'BCR–antigen (5 models)')
ax_bcr.set_xlabel('AUROC', fontsize=10)

# Shared x limits
all_vals = tcr_vals + bcr_vals
x_min = min(all_vals) - 0.03
x_max = max(all_vals) + 0.08
ax_tcr.set_xlim(x_min, x_max)
ax_bcr.set_xlim(x_min, x_max)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
           markeredgecolor='gray', markersize=6, linewidth=0, label='Before'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
           markersize=6, linewidth=0, label='After recalibration'),
]
ax_tcr.legend(handles=legend_elements, loc='lower right', ncol=1,
              fontsize=7, frameon=True, framealpha=0.9)

# Save
out = os.path.join(PANEL_DIR, 'fig5_combined_recal_dumbbell')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}.pdf / .png")
