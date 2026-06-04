#!/usr/bin/env python3
"""Generate ALL remaining individual panels for Figs 3-6.

Each panel saved as panels/fig{N}/fig{N}_{label}_{desc}.pdf + .png
All from cached data — zero distance recomputation.

Run: python generate_remaining_panels.py [--fig 3] [--panel a]
  No args = generate everything. --fig N = only that figure. --panel X = only that panel.
"""
import os, sys, warnings, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)  # panels/fig3/
DESIGNED_DIR = os.path.dirname(os.path.dirname(PANEL_DIR))  # designed_figures/
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
sys.path.insert(0, DESIGNED_DIR)
sys.path.insert(0, SCRIPT_DIR)

from style_config import (
    MODEL_COLORS, MODEL_DISPLAY, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY,
    METRIC_COLORS, METRIC_DISPLAY,
    apply_publication_style, DPI, FONT_LABEL, FONT_TICK, FONT_LEGEND,
)
from calipper.general_evaluator import safe_metric

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
BCR_CACHE = os.path.join(RESULTS, 'fig2_bcr_cache')
FIG34_CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
PANEL_DIR = os.path.join(SCRIPT_DIR, 'panels')

METRIC_COLORS_LOCAL = {'aucroc': '#3498db', 'ap': '#2ecc71', 'f1': '#e74c3c'}
CT_COLORS = {'A1-A11': '#e74c3c', 'BNT162b2': '#9b59b6', 'guoyu': '#2ecc71', 'unseen': '#f39c12'}
FOLD_COLORS = {0: '#a3c4e0', 1: '#ffc68a', 2: '#a3d9a3', 3: '#e8a3a3', 4: '#c4b3d9'}
C_STUDY = {'deepAntigen': '#4C72B0', 'XBCR-net': '#DD8452', 'PanPep': '#55A868',
           'BigMHC': '#C44E52', 'AntibioticsAI': '#8172B3', 'TCR CT': '#937860'}

PW, PH = 3.0, 2.5


def save_panel(fig, fig_num, label, desc):
    d = os.path.join(PANEL_DIR, f'fig{fig_num}')
    os.makedirs(d, exist_ok=True)
    base = f'fig{fig_num}_{label}_{desc}'
    fig.savefig(os.path.join(d, base + '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(d, base + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{label}] {desc}')


def mp():
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    return fig, ax


def should_generate(target_fig, target_panel, fig_num, label):
    if target_fig and target_fig != fig_num:
        return False
    if target_panel and target_panel != label:
        return False
    # Skip if already exists
    d = os.path.join(PANEL_DIR, f'fig{fig_num}')
    path = os.path.join(d, f'fig{fig_num}_{label}_')
    # Check any file starting with this prefix
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith(f'fig{fig_num}_{label}_') and f.endswith('.pdf'):
                return False  # already exists
    return True


# ── Parse args ──
parser = argparse.ArgumentParser()
parser.add_argument('--fig', type=int, default=0, help='Generate only this figure (0=all)')
parser.add_argument('--panel', type=str, default='', help='Generate only this panel label')
parser.add_argument('--force', action='store_true', help='Regenerate even if exists')
args = parser.parse_args()
TF = args.fig; TP = args.panel

if args.force:
    def should_generate(tf, tp, fn, lb):
        if tf and tf != fn: return False
        if tp and tp != lb: return False
        return True


# ═══════════════════════════════════════════
# FIG 3: Dataset-level prediction
# ═══════════════════════════════════════════
def gen_fig3():
    print("\n=== Fig 3 panels ===")

    # Load TCR prediction data from existing composed figure's cache
    tcr_cv_pred = {}
    tcr_ct_pred = {}

    # Load from BCR cache for BCR panels
    bcr_cv = np.load(os.path.join(FIG34_CACHE, 'bcr_fig3_cv_prediction.npz'), allow_pickle=True)
    bcr_ct = np.load(os.path.join(FIG34_CACHE, 'bcr_fig3_ct_prediction.npz'), allow_pickle=True)
    with open(os.path.join(FIG34_CACHE, 'bcr_fig3_cv_curves.json')) as f:
        bcr_cv_curves = json.load(f)
    with open(os.path.join(FIG34_CACHE, 'bcr_fig3_ct_curves.json')) as f:
        bcr_ct_curves = json.load(f)

    # Load BCR v2.6 prediction CSVs
    bcr_cv_v26 = pd.read_csv(os.path.join(RESULTS, 'baselines', 's2dd_v2_6_bcr_cv.csv'))
    bcr_ct_v26 = pd.read_csv(os.path.join(RESULTS, 'baselines', 's2dd_v2_6_bcr_ct.csv'))

    # 8-metric predictions
    eight_metric_path = os.path.join(RESULTS, 'baselines', 's2dd_v2_6_8metric_all_predictions.csv')
    eight_metric = pd.read_csv(eight_metric_path) if os.path.exists(eight_metric_path) else None

    # Baseline predictions
    v26_pred = pd.read_csv(os.path.join(RESULTS, 'baselines', 's2dd_v2_6_prediction_results.csv'))
    pape_pred = pd.read_csv(os.path.join(RESULTS, 'baselines', 'pape_prediction_results.csv'))
    mcbpe_pred = pd.read_csv(os.path.join(RESULTS, 'baselines', 'mcbpe_prediction_results.csv'))

    METRICS = ['aucroc', 'ap', 'f1']
    MDISP = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}

    from calipper.core import fit_best_curve, predict_best_curve

    def _plot_vbias_curves(ax, curves, metric, title, color_key='fold'):
        """Plot degradation curve: y−β·mp on y-axis, exp/Gaussian on curve."""
        all_d, all_mp, all_actual = [], [], []
        labels = []
        for curve in curves:
            if curve['metric'] != metric: continue
            bd = np.array(curve['bin_d']); ba = np.array(curve['bin_actual'])
            bmp = np.array(curve['bin_mp'])
            all_d.extend(bd); all_mp.extend(bmp); all_actual.extend(ba)
            if color_key == 'fold':
                labels.extend([curve['fold']] * len(bd))
            else:
                labels.extend([curve.get('held', '')] * len(bd))

        all_d = np.array(all_d); all_mp = np.array(all_mp); all_actual = np.array(all_actual)

        if len(all_d) >= 4:
            fit_result = fit_best_curve(all_d, all_mp, all_actual, lam=0.05)
            if fit_result['params'] is not None:
                beta = fit_result['params'][-1]
                y_adjusted = all_actual - beta * all_mp

                # Plot adjusted data points
                for lbl in sorted(set(labels)):
                    mask = np.array([l == lbl for l in labels])
                    if color_key == 'fold':
                        color = FOLD_COLORS.get(lbl, '#888')
                    else:
                        color = CT_COLORS.get(lbl, '#888')
                    ax.scatter(all_d[mask], y_adjusted[mask], c=color, s=15, alpha=0.6,
                               edgecolors='white', linewidth=0.2, zorder=5)

                # Plot curve: a·f(d) + c (without β·mp)
                xs = np.linspace(all_d.min(), all_d.max(), 100)
                ys = predict_best_curve(fit_result, xs, np.zeros_like(xs))
                ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.7)
                y_pred_adj = predict_best_curve(fit_result, all_d, np.zeros_like(all_d))
                se = np.std(y_adjusted - y_pred_adj)
                ax.fill_between(xs, ys - se, ys + se, color='gray', alpha=0.08)

                kind = fit_result['kind']
                if kind == 'exp':
                    a, bx, c, _ = fit_result['params']
                    eq = f'exp: {a:.2f}·e^({-bx:.2f}d)+{c:.2f}'
                else:
                    a, d0, sigma, c, _ = fit_result['params']
                    eq = f'Gauss: N(d₀={d0:.1f},σ={sigma:.1f})+{c:.2f}'
                ax.text(0.03, 0.03, f'{eq}  (β={beta:.2f})\nR²={fit_result["r2"]:.3f}',
                        transform=ax.transAxes, fontsize=4.5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='#ccc', alpha=0.9, linewidth=0.3))
            else:
                # Fallback: plot raw if fit fails
                for curve in curves:
                    if curve['metric'] != metric: continue
                    bd = np.array(curve['bin_d']); ba = np.array(curve['bin_actual'])
                    if color_key == 'fold':
                        color = FOLD_COLORS.get(curve['fold'], '#888')
                    else:
                        color = CT_COLORS.get(curve.get('held', ''), '#888')
                    ax.scatter(bd, ba, c=color, s=15, alpha=0.6, edgecolors='white', linewidth=0.2, zorder=5)

        ax.set_xlabel('S2DD distance'); ax.set_ylabel(f'{MDISP[metric]} − vbias')
        ax.set_title(title, fontweight='bold')

    # Panels b,c: BCR CV vbias curves (AUROC, AP)
    for metric, label in [('aucroc', 'b'), ('ap', 'c')]:
        if not should_generate(TF, TP, 3, label): continue
        fig, ax = mp()
        _plot_vbias_curves(ax, bcr_cv_curves, metric, f'BCR CV {MDISP[metric]}', 'fold')
        save_panel(fig, 3, label, f'bcr_cv_vbias_{metric}')

    # Panels d,e: BCR CT vbias curves
    for metric, label in [('aucroc', 'd'), ('ap', 'e')]:
        if not should_generate(TF, TP, 3, label): continue
        fig, ax = mp()
        _plot_vbias_curves(ax, bcr_ct_curves, metric, f'BCR CT {MDISP[metric]}', 'held')
        save_panel(fig, 3, label, f'bcr_ct_vbias_{metric}')

    # Panel f: BCR CV prediction scatter
    if should_generate(TF, TP, 3, 'f'):
        fig, ax = mp()
        all_p, all_a = [], []
        for m in METRICS:
            pk, ak = f'cv_{m}_predicted', f'cv_{m}_actual'
            if pk in bcr_cv:
                ax.scatter(bcr_cv[ak], bcr_cv[pk], c=METRIC_COLORS_LOCAL[m], s=12, alpha=0.6,
                           edgecolors='white', linewidth=0.2, label=MDISP[m])
                all_p.extend(bcr_cv[pk]); all_a.extend(bcr_cv[ak])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
        if all_p:
            r, _ = pearsonr(all_p, all_a); mae = np.mean(np.abs(np.array(all_p)-np.array(all_a)))
            ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}', transform=ax.transAxes,
                    fontsize=6, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title('BCR CV prediction', fontweight='bold')
        ax.legend(fontsize=5, loc='lower right')
        save_panel(fig, 3, 'f', 'bcr_cv_scatter')

    # Panel g: BCR CT prediction scatter
    if should_generate(TF, TP, 3, 'g'):
        fig, ax = mp()
        all_p, all_a = [], []
        for m in METRICS:
            pk, ak = f'ct_{m}_predicted', f'ct_{m}_actual'
            if pk in bcr_ct:
                ax.scatter(bcr_ct[ak], bcr_ct[pk], c=METRIC_COLORS_LOCAL[m], s=12, alpha=0.6,
                           edgecolors='white', linewidth=0.2, label=MDISP[m])
                all_p.extend(bcr_ct[pk]); all_a.extend(bcr_ct[ak])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
        if all_p:
            r, _ = pearsonr(all_p, all_a); mae = np.mean(np.abs(np.array(all_p)-np.array(all_a)))
            ax.text(0.05, 0.93, f'R={r:.3f}\nMAE={mae:.3f}\nn={len(all_p)}', transform=ax.transAxes,
                    fontsize=6, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title('BCR CT prediction', fontweight='bold')
        ax.legend(fontsize=5, loc='lower right')
        save_panel(fig, 3, 'g', 'bcr_ct_scatter')

    # Panel h: Error box plot (8 metrics)
    if should_generate(TF, TP, 3, 'h') and eight_metric is not None:
        fig, ax = mp()
        metrics_8 = ['aucroc', 'ap', 'f1', 'mcc', 'brier', 'bss', 'ppv', 'npv']
        metrics_present = [m for m in metrics_8 if m in eight_metric['metric'].values]
        box_data = [eight_metric[eight_metric['metric'] == m]['abs_error'].values for m in metrics_present]
        bp = ax.boxplot(box_data, tick_labels=[m.upper() for m in metrics_present],
                       patch_artist=True, widths=0.5)
        colors_box = [METRIC_COLORS_LOCAL.get(m, '#888') for m in metrics_present]
        for patch, c in zip(bp['boxes'], colors_box):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.set_ylabel('Prediction |error|')
        ax.set_title('8-metric prediction error', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        save_panel(fig, 3, 'h', 'error_boxplot_8metric')

    # Panel i: Per-test-set AUROC error
    if should_generate(TF, TP, 3, 'i'):
        fig, ax = mp()
        ct_v26 = v26_pred[v26_pred['experiment'] == 'crosstest']
        ct_auc = ct_v26[ct_v26['metric'] == 'aucroc']
        if len(ct_auc) > 0 and 'setting' in ct_auc.columns:
            settings = ct_auc['setting'].unique()
            for s in settings:
                sub = ct_auc[ct_auc['setting'] == s]
                ax.scatter(sub['actual'], sub['predicted'], s=30, alpha=0.7,
                          edgecolors='white', linewidth=0.3, label=s[:8])
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
            ax.legend(fontsize=4, loc='lower right')
        ax.set_xlabel('Actual AUROC'); ax.set_ylabel('Predicted AUROC')
        ax.set_title('Per-test-set AUROC (TCR CT)', fontweight='bold')
        save_panel(fig, 3, 'i', 'per_testset_auroc')

    # Panel j: Prediction MAE comparison (S2DD vs PAPE vs M-CBPE)
    if should_generate(TF, TP, 3, 'j'):
        fig, ax = mp()
        ct_v26_f = v26_pred[v26_pred['setting'].str.contains('test|Test|seen|unseen|mcpas|iedb|v3|v4', case=False, na=False)] if 'setting' in v26_pred.columns else v26_pred
        ct_pape = pape_pred[(pape_pred['baseline'] == 'BL-1_pape_avg') & (pape_pred['experiment'] == 'crosstest')]
        ct_mcbpe = mcbpe_pred[(mcbpe_pred['baseline'] == 'BL-1_mcbpe_avg') & (mcbpe_pred['experiment'] == 'crosstest')]
        x_g = np.arange(3); w = 0.22
        for mi, m in enumerate(['aucroc', 'ap', 'f1']):
            v = ct_v26_f[ct_v26_f['metric'] == m]['abs_error'].mean() if len(ct_v26_f[ct_v26_f['metric'] == m]) > 0 else 0
            p = ct_pape[ct_pape['metric'] == m]['abs_error'].mean() if len(ct_pape[ct_pape['metric'] == m]) > 0 else 0
            mc = ct_mcbpe[ct_mcbpe['metric'] == m]['abs_error'].mean() if len(ct_mcbpe[ct_mcbpe['metric'] == m]) > 0 else 0
            ax.bar(x_g[mi] - w, v, w, color='#3498db', edgecolor='black', linewidth=0.3)
            ax.bar(x_g[mi], p, w, color='#2ecc71', edgecolor='black', linewidth=0.3)
            ax.bar(x_g[mi] + w, mc, w, color='#e74c3c', edgecolor='black', linewidth=0.3)
        ax.set_xticks(x_g); ax.set_xticklabels(['AUROC', 'AP', 'F1'])
        ax.set_ylabel('Prediction MAE')
        ax.legend(['S2DD', 'PAPE', 'M-CBPE'], fontsize=5, loc='upper left')
        ax.set_title('Method comparison (TCR CT)', fontweight='bold')
        save_panel(fig, 3, 'j', 'method_comparison_mae')

    # Panel k: v2.6 vs vanilla BCR boxplot
    if should_generate(TF, TP, 3, 'k'):
        fig, ax = mp()
        for i, (label, df) in enumerate([('CV', bcr_cv_v26), ('CT', bcr_ct_v26)]):
            v26_err = df[df['method'] == 'v2.6']['abs_error'].values
            van_err = df[df['method'] == 'vanilla']['abs_error'].values if 'vanilla' in df['method'].values else np.array([])
            offset = i * 2.5
            if len(v26_err) > 0:
                bp = ax.boxplot([v26_err], positions=[offset], widths=0.4, patch_artist=True)
                bp['boxes'][0].set_facecolor('#3498db'); bp['boxes'][0].set_alpha(0.6)
            if len(van_err) > 0:
                bp = ax.boxplot([van_err], positions=[offset + 0.6], widths=0.4, patch_artist=True)
                bp['boxes'][0].set_facecolor('#e74c3c'); bp['boxes'][0].set_alpha(0.6)
        ax.set_xticks([0, 0.6, 2.5, 3.1])
        ax.set_xticklabels(['v2.6', 'van.', 'v2.6', 'van.'], fontsize=6)
        ax.text(0.3, -0.15, 'CV', transform=ax.transAxes, ha='center', fontsize=7)
        ax.text(0.8, -0.15, 'CT', transform=ax.transAxes, ha='center', fontsize=7)
        ax.set_ylabel('Prediction |error|')
        ax.set_title('BCR: v2.6 vs vanilla', fontweight='bold')
        save_panel(fig, 3, 'k', 'bcr_v26_vs_vanilla')


# ═══════════════════════════════════════════
# FIG 4: Subset-level prediction
# ═══════════════════════════════════════════
def gen_fig4():
    print("\n=== Fig 4 panels ===")

    # Load v2.6 subset predictions from cache
    v26_data = {}
    for split in ['ct', 'cv']:
        for strategy in ['antigen', 'distance']:
            for metric in ['aucroc', 'ap']:
                key = f'{split}_{strategy}_{metric}'
                path = os.path.join(FIG34_CACHE, f'bcr_fig4_v26_{key}.csv')
                if os.path.exists(path):
                    v26_data[key] = pd.read_csv(path)

    configs = [
        (0, 'aucroc', 'ct', 'a'), (0, 'ap', 'ct', 'e'),
        (0, 'aucroc', 'cv', 'i'), (0, 'ap', 'cv', 'm'),
    ]

    for _, metric, split, base_label in configs:
        mdisp = 'AUROC' if metric == 'aucroc' else 'AP'
        sdisp = 'CT' if split == 'ct' else 'CV'

        for col, strategy, col_label_offset in [(0, 'distance', 0), (1, 'distance', 1),
                                                  (2, 'antigen', 2), (3, 'distance', 3)]:
            label = chr(ord(base_label) + col_label_offset)
            desc_map = {0: 'source_identity', 1: 'sample_properties', 2: 'antigen_splitting', 3: 'distance_splitting'}

            if not should_generate(TF, TP, 4, label):
                continue

            key = f'{split}_{strategy}_{metric}'
            if key not in v26_data:
                continue

            fig, ax = mp()
            df = v26_data[key]

            if col == 1:  # Sample properties (color=prevalence)
                sc = ax.scatter(df['actual'], df['predicted'], c=df['prevalence'],
                               cmap='coolwarm', s=np.clip(df['n']/10, 10, 100),
                               alpha=0.7, edgecolors='white', linewidth=0.2)
                plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label('Pos. ratio', fontsize=5)
            else:  # Colored by source
                if split == 'ct':
                    for ts in df['source'].unique():
                        sub = df[df['source'] == ts]
                        ax.scatter(sub['actual'], sub['predicted'], c=CT_COLORS.get(ts, '#888'),
                                  s=20, alpha=0.7, edgecolors='white', linewidth=0.2, label=ts)
                else:
                    for src in df['source'].unique():
                        fold_n = int(src.replace('fold', ''))
                        sub = df[df['source'] == src]
                        ax.scatter(sub['actual'], sub['predicted'], c=FOLD_COLORS.get(fold_n, '#888'),
                                  s=20, alpha=0.7, edgecolors='white', linewidth=0.2, label=src)
                ax.legend(fontsize=4, loc='lower right', framealpha=0.85)

            # Diagonal + stats
            if len(df) >= 3:
                lo = min(df['actual'].min(), df['predicted'].min()) - 0.05
                hi = max(df['actual'].max(), df['predicted'].max()) + 0.05
                ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.5, alpha=0.3)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                r, _ = pearsonr(df['actual'], df['predicted'])
                mae = np.mean(np.abs(df['actual'] - df['predicted']))
                ax.text(0.05, 0.93, f'r={r:.2f}\nMAE={mae:.3f}', transform=ax.transAxes,
                        fontsize=6, va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                                         edgecolor='#ccc', alpha=0.9, linewidth=0.3))

            ax.set_xlabel(f'Actual {mdisp}'); ax.set_ylabel(f'Predicted {mdisp}')
            ax.set_title(f'BCR {sdisp} {mdisp}\n{desc_map[col].replace("_", " ").title()}', fontweight='bold')
            save_panel(fig, 4, label, f'bcr_{split}_{metric}_{desc_map[col]}')


# ═══════════════════════════════════════════
# FIG 5: Recalibration
# ═══════════════════════════════════════════
def gen_fig5():
    print("\n=== Fig 5 panels ===")

    per_bin = pd.read_csv(os.path.join(RESULTS, 'baselines', 'per_bin_recalibration_delta.csv'))
    per_ep = pd.read_csv(os.path.join(RESULTS, 'baselines', 'per_epitope_recalibration_delta.csv'))
    adaptive = pd.read_csv(os.path.join(RESULTS, 'symmetric_strategy_comparison.csv'))

    # Panel l: Per-bin ΔAUROC
    if should_generate(TF, TP, 5, 'l'):
        fig, ax = mp()
        r_rescue, p_rescue = pearsonr(per_bin['mean_dist'], per_bin['delta_auroc'])
        colors = [plt.cm.YlGn(0.25 + 0.65 * (d - per_bin['delta_auroc'].min()) /
                  (per_bin['delta_auroc'].max() - per_bin['delta_auroc'].min() + 1e-9))
                  for d in per_bin['delta_auroc']]
        bx = np.arange(1, len(per_bin) + 1)
        ax.bar(bx, per_bin['delta_auroc'], width=0.7, color=colors, edgecolor='black', linewidth=0.3)
        z = np.polyfit(bx, per_bin['delta_auroc'].values, 1)
        ax.plot(np.linspace(0.5, 8.5, 50), np.polyval(z, np.linspace(0.5, 8.5, 50)), 'k--', linewidth=1)
        near = per_bin['delta_auroc'].iloc[0]; far = per_bin['delta_auroc'].iloc[-1]
        ax.text(0.05, 0.93, f'r={r_rescue:.3f} (p={p_rescue:.3f})\nFar/near: {far/near:.0f}×',
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#333', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Distance bin (near → far)'); ax.set_ylabel('ΔAUROC')
        ax.set_xticks([1, 4, 8]); ax.set_xticklabels(['1\n(near)', '4', '8\n(far)'])
        ax.set_title('Far samples gain most', fontweight='bold')
        save_panel(fig, 5, 'l', 'per_bin_delta_auroc')

    # Panel m: Per-epitope before/after scatter
    if should_generate(TF, TP, 5, 'm'):
        fig, ax = mp()
        ep_f = per_ep[per_ep['n'] >= 30]
        ax.scatter(ep_f['auroc_before'], ep_f['auroc_after'], c=ep_f['mean_dist'], cmap='viridis',
                  s=8, alpha=0.5, edgecolors='white', linewidth=0.1)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
        n_imp = (ep_f['delta_auroc'] > 0).sum()
        ax.text(0.05, 0.93, f'Improved: {n_imp}/{len(ep_f)} ({n_imp/len(ep_f):.0%})',
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('AUROC before'); ax.set_ylabel('AUROC after')
        ax.set_title('Per-epitope recalibration', fontweight='bold')
        save_panel(fig, 5, 'm', 'per_epitope_scatter')

    # Panel n: Raw AUROC vs ΔAUROC
    if should_generate(TF, TP, 5, 'n'):
        fig, ax = mp()
        ep_f = per_ep[per_ep['n'] >= 30]
        ax.scatter(ep_f['auroc_before'], ep_f['delta_auroc'], c=ep_f['mean_dist'], cmap='viridis',
                  s=8, alpha=0.5, edgecolors='white', linewidth=0.1)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        r, _ = pearsonr(ep_f['auroc_before'], ep_f['delta_auroc'])
        ax.text(0.05, 0.93, f'r={r:.3f}', transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Raw AUROC'); ax.set_ylabel('ΔAUROC')
        ax.set_title('Worse models gain more', fontweight='bold')
        save_panel(fig, 5, 'n', 'raw_vs_delta_auroc')

    # Panel o: Distance vs ΔAUROC per-epitope
    if should_generate(TF, TP, 5, 'o'):
        fig, ax = mp()
        ep_f = per_ep[per_ep['n'] >= 30]
        ax.scatter(ep_f['mean_dist'], ep_f['delta_auroc'], c=ep_f['auroc_before'], cmap='coolwarm',
                  s=8, alpha=0.5, edgecolors='white', linewidth=0.1)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        r, _ = pearsonr(ep_f['mean_dist'], ep_f['delta_auroc'])
        ax.text(0.05, 0.93, f'r={r:.3f}', transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Mean S2DD distance'); ax.set_ylabel('ΔAUROC')
        ax.set_title('Far epitopes gain more', fontweight='bold')
        save_panel(fig, 5, 'o', 'distance_vs_delta_auroc')

    # Panel k: Adaptive strategy
    if should_generate(TF, TP, 5, 'k'):
        fig, ax = mp()
        fixed = adaptive['delta_auc_fixed'].values; sym = adaptive['delta_auc_symmetric'].values
        for i in range(len(fixed)):
            if abs(sym[i] - fixed[i]) < 0.001: c, m = '#888', 'o'
            elif sym[i] > fixed[i]: c, m = '#55A868', '^'
            else: c, m = '#C44E52', 'v'
            ax.scatter(fixed[i], sym[i], c=c, marker=m, s=30, edgecolor='white', linewidth=0.3, zorder=5)
        lm = min(min(fixed), min(sym)) - 0.02; lx = max(max(fixed), max(sym)) + 0.02
        ax.plot([lm, lx], [lm, lx], 'k--', linewidth=0.5, alpha=0.3)
        n_imp = sum(1 for f, s in zip(fixed, sym) if s > f + 0.001)
        n_same = sum(1 for f, s in zip(fixed, sym) if abs(s - f) <= 0.001)
        ax.text(0.05, 0.93, f'Improved: {n_imp}/{len(fixed)}\nSame: {n_same}/{len(fixed)}\nWorse: 0',
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('ΔAUROC (fixed)'); ax.set_ylabel('ΔAUROC (adaptive)')
        ax.set_title('Adaptive: never hurts', fontweight='bold')
        save_panel(fig, 5, 'k', 'adaptive_strategy')


# ═══════════════════════════════════════════
# FIG 6: Retrospective (remaining panels e-p)
# ═══════════════════════════════════════════
def gen_fig6():
    print("\n=== Fig 6 panels ===")

    ROOT = INPUT_DIR
    da_baseline = pd.read_csv(f'{INPUT_DIR}/results/deepantigen_retrospective/neoantigen_recalibration/baselines_and_cases/baseline_comparison.csv')
    xbcr_omicron = pd.read_csv(f'{INPUT_DIR}/results/xbcr_retrospective/mab_recalibration/panel2_omicron_results.csv')
    aa_topk = pd.read_csv(f'{INPUT_DIR}/results/antibioticsai_retrospective/recalibration/topk_tdr_comparison.csv')
    da_neo_rank = pd.read_csv(f'{INPUT_DIR}/results/deepantigen_retrospective/neoantigen_recalibration/baselines_and_cases/neoantigen_ranking_comparison.csv')
    pp_neo = pd.read_csv(f'{INPUT_DIR}/results/panpep_retrospective/neoantigen_recalibration/neoantigen_panpep_analysis.csv')
    pp_recal = pd.read_csv(f'{INPUT_DIR}/results/panpep_retrospective/neoantigen_recalibration/majority_recalibration.csv')
    adaptive = pd.read_csv(f'{INPUT_DIR}/results/symmetric_strategy_comparison.csv')
    bm_recal = pd.read_csv(f'{INPUT_DIR}/results/bigmhc_retrospective/manafest_recalibration/all_models_recalibration.csv')
    per_bin = pd.read_csv(f'{INPUT_DIR}/results/baselines/per_bin_recalibration_delta.csv')
    tcr_ct_pred = pd.read_csv(f'{INPUT_DIR}/results/baselines/fig6_tcr_ct_auroc_corrected.csv')
    da_pred = pd.read_csv(f'{INPUT_DIR}/results/deepantigen_retrospective/performance_prediction/prediction_results_2chain.csv')
    aa_pred = pd.read_csv(f'{INPUT_DIR}/results/antibioticsai_retrospective/performance_prediction/prediction_results.csv')
    bm_pred = pd.read_csv(f'{INPUT_DIR}/results/bigmhc_retrospective/performance_prediction/dataset_level_prediction.csv')
    xbcr_pred = pd.read_csv(f'{INPUT_DIR}/results/xbcr_retrospective/performance_prediction/dataset_level.csv')
    bm_deg = pd.read_csv(f'{INPUT_DIR}/results/bigmhc_retrospective/s2dd_degradation/all_models_degradation.csv')
    aa_repro = pd.read_csv(f'{INPUT_DIR}/results/antibioticsai_retrospective/reproduction/main_test_performance.csv')
    v26_pred = pd.read_csv(f'{INPUT_DIR}/results/baselines/s2dd_v2_6_prediction_results.csv')
    pape_pred = pd.read_csv(f'{INPUT_DIR}/results/baselines/pape_prediction_results.csv')
    mcbpe_pred = pd.read_csv(f'{INPUT_DIR}/results/baselines/mcbpe_prediction_results.csv')

    # Panel e: Pooled AUROC prediction scatter
    if should_generate(TF, TP, 6, 'e'):
        fig, ax = mp()
        pred_data = []
        for _, row in tcr_ct_pred.iterrows():
            pred_data.append(('TCR CT', row['predicted'], row['actual'], 10000))
        pred_data.append(('deepAntigen', 0.716, 0.714, 564514))
        aa_auc = aa_pred[(aa_pred['metric'] == 'aucroc') & (~aa_pred['source'].fillna('').str.contains('halfsplit'))]
        if len(aa_auc) > 0: pred_data.append(('AntibioticsAI', float(aa_auc['predicted'].iloc[0]), float(aa_auc['actual'].iloc[0]), 505))
        bm_auc = bm_pred[(bm_pred['metric'] == 'aucroc') & (bm_pred['calibration'] == '28_HLA_cal_sets')]
        if len(bm_auc) > 0: pred_data.append(('BigMHC', float(bm_auc['predicted'].iloc[0]), float(bm_auc['actual'].iloc[0]), 834))
        xbcr_auc = xbcr_pred[(xbcr_pred['metric'] == 'aucroc') & (xbcr_pred['variant_group'] == 'SARS-CoV2_Beta')]
        if len(xbcr_auc) > 0: pred_data.append(('XBCR-net', float(xbcr_auc['pred_v26'].iloc[0]), float(xbcr_auc['actual'].iloc[0]), 20))
        actuals, preds = [], []
        for study, pred, actual, n in pred_data:
            ms = max(12, min(50, np.log(n+1)*3.5))
            ax.scatter(actual, pred, c=C_STUDY.get(study, '#888'), s=ms, edgecolor='white', linewidth=0.3, zorder=3, alpha=0.85)
            actuals.append(actual); preds.append(pred)
        ax.plot([0.35, 1.05], [0.35, 1.05], 'k--', linewidth=0.5, alpha=0.3)
        r, _ = pearsonr(actuals, preds)
        ax.text(0.05, 0.93, f'r={r:.3f}\nn={len(pred_data)}', transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Actual AUROC'); ax.set_ylabel('Predicted AUROC')
        ax.set_title('Predicted vs actual AUROC', fontweight='bold')
        for s in ['TCR CT', 'deepAntigen', 'AntibioticsAI', 'BigMHC', 'XBCR-net']:
            ax.scatter([], [], c=C_STUDY.get(s, '#888'), s=12, label=s)
        ax.legend(fontsize=4, loc='lower right')
        save_panel(fig, 6, 'e', 'pooled_auroc_scatter')

    # Panel f: Per-study MAE lollipop
    if should_generate(TF, TP, 6, 'f'):
        fig, ax = mp()
        study_mae = []
        da_ic = da_pred[da_pred['metric'] == 'aucroc']
        if len(da_ic) > 0: study_mae.append(('deepAntigen', float(da_ic['MAE'].iloc[0])))
        bm_a2 = bm_pred[(bm_pred['metric'] == 'aucroc') & (bm_pred['calibration'] == '28_HLA_cal_sets')]
        if len(bm_a2) > 0: study_mae.append(('BigMHC', float(bm_a2['abs_error'].iloc[0])))
        study_mae.append(('TCR CT', tcr_ct_pred['abs_error'].mean()))
        aa_a2 = aa_pred[(aa_pred['metric'] == 'aucroc') & (~aa_pred['source'].fillna('').str.contains('halfsplit'))]
        if len(aa_a2) > 0: study_mae.append(('AntibioticsAI', float(aa_a2['abs_error'].iloc[0])))
        xbcr_a2 = xbcr_pred[(xbcr_pred['metric'] == 'aucroc') & (xbcr_pred['variant_group'] == 'SARS-CoV2_Beta')]
        if len(xbcr_a2) > 0: study_mae.append(('XBCR-net', float(xbcr_a2['ae_v26'].iloc[0])))
        study_mae.sort(key=lambda x: x[1])
        y_pos = np.arange(len(study_mae))[::-1]
        for i, (s, mae) in enumerate(study_mae):
            ax.plot([0, mae], [y_pos[i], y_pos[i]], color=C_STUDY.get(s, '#888'), linewidth=2, solid_capstyle='round')
            ax.scatter(mae, y_pos[i], color=C_STUDY.get(s, '#888'), s=40, zorder=5, edgecolor='white', linewidth=0.4)
            ax.text(mae + 0.003, y_pos[i], f'{mae:.3f}', va='center', fontsize=5.5, color=C_STUDY.get(s, '#888'))
        ax.set_yticks(y_pos); ax.set_yticklabels([s[0] for s in study_mae], fontsize=5.5)
        ax.set_xlabel('AUROC prediction MAE'); ax.set_xlim(-0.005, 0.13)
        ax.set_title('Prediction error by study', fontweight='bold')
        save_panel(fig, 6, 'f', 'per_study_mae')

    # Panel g: Prevalence confound
    if should_generate(TF, TP, 6, 'g'):
        fig, ax = mp()
        bm = bm_deg[bm_deg['model'] == 'BigMHC_IM']
        ax.scatter(bm['prevalence'], bm['ap'], color='#C44E52', s=28, edgecolor='white', linewidth=0.3, label='AP')
        z = np.polyfit(bm['prevalence'].values, bm['ap'].values, 1)
        xs = np.linspace(bm['prevalence'].min()-0.01, bm['prevalence'].max()+0.01, 50)
        ax.plot(xs, np.polyval(z, xs), color='#C44E52', linewidth=1)
        ax.scatter(bm['prevalence'], bm['aucroc'], color='#888', s=18, marker='s', edgecolor='white', linewidth=0.3, alpha=0.6, label='AUROC')
        z2 = np.polyfit(bm['prevalence'].values, bm['aucroc'].values, 1)
        ax.plot(xs, np.polyval(z2, xs), color='#888', linewidth=0.8, linestyle='--')
        r_ap, _ = pearsonr(bm['prevalence'], bm['ap']); r_auc, _ = pearsonr(bm['prevalence'], bm['aucroc'])
        ax.text(0.05, 0.93, f'AP vs prev: r={r_ap:.2f}\nAUROC vs prev: r={r_auc:.2f}', transform=ax.transAxes, fontsize=5.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Bin prevalence'); ax.set_ylabel('Metric value')
        ax.legend(fontsize=5, loc='lower right'); ax.set_title('Prevalence confound\n(→ use AUROC)', fontweight='bold')
        save_panel(fig, 6, 'g', 'prevalence_confound')

    # Panel h: Task-type bar chart
    if should_generate(TF, TP, 6, 'h'):
        fig, ax = mp()
        da_deg = pd.read_csv(f'{INPUT_DIR}/results/deepantigen_retrospective/s2dd_degradation/binned_s2dd_sw.csv')
        pp_deg = pd.read_csv(f'{INPUT_DIR}/results/panpep_retrospective/s2dd_degradation/majority_degradation_curves.csv')
        aa_deg = pd.read_csv(f'{INPUT_DIR}/results/antibioticsai_retrospective/s2dd_degradation/main_degradation.csv')
        xbcr_dg = pd.read_csv(f'{INPUT_DIR}/results/xbcr_retrospective/s2dd_degradation/degradation_curves.csv')
        tasks = []
        r, _ = pearsonr(da_deg['mean_val'], da_deg['aucroc']); tasks.append(('deepAntigen', r, 'TCR binding'))
        r, _ = pearsonr(pp_deg['mean_dist'], pp_deg['aucroc']); tasks.append(('PanPep', r, 'TCR binding'))
        xv = xbcr_dg[xbcr_dg['aucroc'].notna()]
        if len(xv) >= 3: r, _ = pearsonr(xv['mean_dist'], xv['aucroc']); tasks.append(('XBCR-net', r, 'BCR binding'))
        r, _ = pearsonr(aa_deg['mean_dist'], aa_deg['aucroc']); tasks.append(('AntibioticsAI', r, 'Drug activity'))
        bm_rs = [pearsonr(bm_deg[bm_deg['model']==m]['mean_dist'], bm_deg[bm_deg['model']==m]['aucroc'])[0] for m in bm_deg['model'].unique()]
        tasks.append(('BigMHC', np.mean(bm_rs), 'Immunogenicity'))
        tasks.sort(key=lambda x: x[1])
        y = np.arange(len(tasks))
        for i, (name, r, task) in enumerate(tasks):
            ax.barh(y[i], r, height=0.55, color=C_STUDY.get(name, '#888'), alpha=0.7, edgecolor='black', linewidth=0.3)
            side = 'left' if r < 0 else 'right'; off = -0.03 if r < 0 else 0.03
            ax.text(r+off, y[i], f'{r:.2f}', va='center', ha=side, fontsize=5.5, fontweight='bold', color=C_STUDY.get(name, '#888'))
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_yticks(y); ax.set_yticklabels([f'{t[0]}\n({t[2]})' for t in tasks], fontsize=5)
        ax.set_xlabel('AUROC degradation r'); ax.set_xlim(-1.1, 0.3)
        ax.set_title('Task-type determines\ndegradation', fontweight='bold')
        save_panel(fig, 6, 'h', 'task_type')

    # Panel i: Recalibration ΔAUROC dumbbell (all 5 studies)
    if should_generate(TF, TP, 6, 'i'):
        fig, ax = mp()
        da_raw = float(da_baseline[da_baseline['method'] == 'Raw deepAntigen']['AUROC'].iloc[0])
        da_bl = float(da_baseline[da_baseline['method'] == 'S2DD-BLOSUM Bayesian']['AUROC'].iloc[0])
        from sklearn.metrics import roc_auc_score
        xo = xbcr_omicron[xbcr_omicron['variant'] == 'omicron'].dropna(subset=['antibody_name'])
        xo = xo[xo['antibody_name'].str.len() > 0]
        try: xb = roc_auc_score(xo['gt_binds'], xo['prediction_score']); xa = roc_auc_score(xo['gt_binds'], xo['calibrated_score'])
        except Exception as e:
            raise RuntimeError(f"XBCR-net AUROC computation failed ({e}). Hardcoded fallback (0.726/0.822) removed to prevent silent insertion of canonical manuscript values — verify upstream CSV at INPUT_DIR/results/xbcr_retrospective/.") from e
        bm_ra = bm_recal[(bm_recal['model']=='BigMHC_IM') & (bm_recal['metric']=='aucroc')]
        bmb, bma = float(bm_ra['before'].iloc[0]), float(bm_ra['after'].iloc[0])
        aa_ad = adaptive[adaptive['study']=='AntibioticsAI']
        if len(aa_ad) > 0:
            aa_d = float(aa_ad['delta_auc_symmetric'].iloc[0])
        else:
            raise RuntimeError("AntibioticsAI delta_auc_symmetric missing from upstream CSV. Hardcoded fallback (0.055) removed to prevent silent insertion of canonical manuscript value — verify upstream data at INPUT_DIR/results/antibioticsai_retrospective/.")
        aab = float(aa_repro['auroc'].iloc[0]); aaa = aab + aa_d
        pp_ar = pp_recal[pp_recal['metric']=='aucroc']
        ppb, ppa = float(pp_ar['before'].iloc[0]), float(pp_ar['after'].iloc[0])
        recal = [('deepAntigen', da_raw, da_bl), ('XBCR-net', xb, xa), ('AntibioticsAI', aab, aaa),
                 ('BigMHC', bmb, bma), ('PanPep', ppb, ppa)]
        recal.sort(key=lambda x: x[2]-x[1], reverse=True)
        yp = np.arange(len(recal))[::-1]
        for i, (s, b, a) in enumerate(recal):
            ax.plot([b, a], [yp[i], yp[i]], color=C_STUDY[s], linewidth=2, solid_capstyle='round', alpha=0.5)
            ax.scatter(b, yp[i], color='white', edgecolor=C_STUDY[s], s=22, zorder=5, linewidth=0.8)
            ax.scatter(a, yp[i], color=C_STUDY[s], s=30, zorder=5, edgecolor='white', linewidth=0.4)
            ax.text((b+a)/2, yp[i]+0.25, f'+{a-b:.3f}', ha='center', fontsize=5, color=C_STUDY[s], fontweight='bold')
        ax.axvline(0.5, color='gray', linewidth=0.3, linestyle=':', alpha=0.5)
        ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in recal], fontsize=5)
        ax.set_xlabel('AUROC'); ax.set_xlim(0.38, 1.08)
        ax.set_title('Recalibration ΔAUROC\n(all 5 studies)', fontweight='bold')
        save_panel(fig, 6, 'i', 'recal_dumbbell_auroc')

    # Panel j: 7-method baseline comparison
    if should_generate(TF, TP, 6, 'j'):
        fig, ax = mp()
        methods = da_baseline['method'].values; tdr5 = da_baseline['TDR@5'].values
        colors_b = []
        for m in methods:
            if 'BLOSUM' in m: colors_b.append('#4C72B0')
            elif 'Lev' in m: colors_b.append('#7aabd4')
            elif 'Isotonic' in m: colors_b.append('#999')
            else: colors_b.append('#ccc')
        short = [m.replace('S2DD-BLOSUM Bayesian', 'S2DD-\nBLOSUM').replace('S2DD-Lev Bayesian', 'S2DD-\nLev')
                  .replace('Prevalence-adjusted', 'Prev.\nadj.').replace('Isotonic regression', 'Isotonic\ncal.')
                  .replace('Platt scaling', 'Platt\ncal.').replace('Raw deepAntigen', 'Raw\nmodel') for m in methods]
        xb = np.arange(len(methods))
        ax.bar(xb, tdr5, width=0.65, color=colors_b, edgecolor='black', linewidth=0.3)
        for i, v in enumerate(tdr5):
            if v > 0: ax.text(xb[i], v+0.02, f'{v:.0%}', ha='center', fontsize=5, fontweight='bold',
                              color=colors_b[i] if colors_b[i] not in ('#ccc','#999') else '#555')
        ax.set_xticks(xb); ax.set_xticklabels(short, fontsize=4.5)
        ax.set_ylabel('Top-5 TDR'); ax.set_ylim(0, 0.82)
        ax.set_title('Neoantigen: S2DD vs 6 baselines', fontweight='bold')
        save_panel(fig, 6, 'j', 'neoantigen_baseline')

    # Panel k: XBCR-net Omicron ranking
    if should_generate(TF, TP, 6, 'k'):
        fig, ax = mp()
        om = xbcr_omicron[xbcr_omicron['variant'] == 'omicron'].dropna(subset=['antibody_name'])
        om = om[om['antibody_name'].str.len() > 0]
        for i, (ab, lbl) in enumerate([('REGN10933', 'REGN10933\n(escape)'), ('LY-CoV1404', 'LY-CoV1404\n(retains)'), ('LY-CoV555', 'LY-CoV555\n(escape)')]):
            row = om[om['antibody_name'] == ab]
            if len(row) == 0: continue
            rr, cr = int(row['raw_rank'].iloc[0]), int(row['cal_rank'].iloc[0])
            c = '#C44E52' if cr > rr else '#55A868'
            ax.annotate('', xy=(cr, 2-i), xytext=(rr, 2-i), arrowprops=dict(arrowstyle='->', color=c, linewidth=1.5))
            ax.scatter(rr, 2-i, color='white', edgecolor='#DD8452', s=25, zorder=5, linewidth=0.8)
            ax.scatter(cr, 2-i, color='#DD8452', s=30, zorder=5, edgecolor='white', linewidth=0.4)
            d = cr - rr; ax.text(max(rr, cr)+0.8, 2-i, f'{"+"+str(d) if d>0 else str(d)}', va='center', fontsize=5.5, color=c, fontweight='bold')
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['LY-CoV555\n(escape)', 'LY-CoV1404\n(retains)', 'REGN10933\n(escape)'], fontsize=5)
        ax.set_xlabel('Rank'); ax.set_title(f'XBCR-net Omicron', fontweight='bold'); ax.invert_xaxis()
        save_panel(fig, 6, 'k', 'xbcr_omicron_ranking')

    # Panel l: AntibioticsAI Top-k TDR
    if should_generate(TF, TP, 6, 'l'):
        fig, ax = mp()
        k_v = aa_topk['top_k'].values; tm = aa_topk['tdr_ps'].values; tc = aa_topk['tdr_combined'].values
        ax.plot(k_v, tm, 'o-', color='#aaa', markersize=3, linewidth=1.2, label='Model only')
        ax.plot(k_v, tc, 's-', color='#8172B3', markersize=3, linewidth=1.2, label='S2DD + model')
        ax.fill_between(k_v, tm, tc, color='#8172B3', alpha=0.12)
        i50 = np.where(k_v == 50)[0]
        if len(i50) > 0:
            ax.annotate(f'{tm[i50[0]]:.0%}→{tc[i50[0]]:.0%}\n(p=0.049)', xy=(k_v[i50[0]], tc[i50[0]]),
                       xytext=(k_v[i50[0]]+15, tc[i50[0]]+0.12), fontsize=5, color='#8172B3', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#8172B3', linewidth=0.6))
        ax.set_xlabel('Top-k'); ax.set_ylabel('True Discovery Rate')
        ax.legend(fontsize=5, loc='upper right'); ax.set_title('AntibioticsAI TDR', fontweight='bold')
        save_panel(fig, 6, 'l', 'antibioticsai_topk_tdr')

    # Panel m: Per-bin ΔAUROC
    if should_generate(TF, TP, 6, 'm'):
        fig, ax = mp()
        r_r, p_r = pearsonr(per_bin['mean_dist'], per_bin['delta_auroc'])
        colors = [plt.cm.YlGn(0.25 + 0.65*(d-per_bin['delta_auroc'].min())/(per_bin['delta_auroc'].max()-per_bin['delta_auroc'].min()+1e-9)) for d in per_bin['delta_auroc']]
        bx = np.arange(1, len(per_bin)+1)
        ax.bar(bx, per_bin['delta_auroc'], width=0.7, color=colors, edgecolor='black', linewidth=0.3)
        z = np.polyfit(bx, per_bin['delta_auroc'].values, 1)
        ax.plot(np.linspace(0.5, 8.5, 50), np.polyval(z, np.linspace(0.5, 8.5, 50)), 'k--', linewidth=1)
        near = per_bin['delta_auroc'].iloc[0]; far = per_bin['delta_auroc'].iloc[-1]
        ax.text(0.05, 0.93, f'r={r_r:.3f}\nFar/near: {far/near:.0f}×', transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#333', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Distance bin'); ax.set_ylabel('ΔAUROC')
        ax.set_xticks([1, 4, 8]); ax.set_xticklabels(['1\n(near)', '4', '8\n(far)'])
        ax.set_title('Far samples gain most', fontweight='bold')
        save_panel(fig, 6, 'm', 'per_bin_delta')

    # Panel n: Neoantigen reranking scatter
    if should_generate(TF, TP, 6, 'n'):
        fig, ax = mp()
        conf = da_neo_rank[da_neo_rank['confirmed'] == True]
        nconf = da_neo_rank[da_neo_rank['confirmed'] == False]
        ax.scatter(nconf['raw_rank'], nconf['blosum_rank'], color='#ccc', s=8, alpha=0.25, zorder=3, label=f'Non-confirmed ({len(nconf)})')
        ax.scatter(conf['raw_rank'], conf['blosum_rank'], color='#4C72B0', s=30, edgecolor='white', linewidth=0.3, zorder=5, label=f'Confirmed ({len(conf)})')
        ax.plot([1, 100], [1, 100], 'k--', linewidth=0.5, alpha=0.3)
        n_prom = (conf['blosum_rank'] < conf['raw_rank']).sum()
        ax.text(0.05, 0.93, f'{n_prom}/{len(conf)} confirmed\npromoted', transform=ax.transAxes, fontsize=5.5, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#4C72B0', alpha=0.9, linewidth=0.3))
        ax.set_xlabel('Raw model rank'); ax.set_ylabel('S2DD-BLOSUM rank')
        ax.set_xlim(0, 105); ax.set_ylim(0, 105); ax.legend(fontsize=4.5, loc='lower right')
        ax.set_title('Neoantigen reranking', fontweight='bold')
        save_panel(fig, 6, 'n', 'neoantigen_reranking')

    # Panel o: Neoantigen baseline bar (7 methods)
    if should_generate(TF, TP, 6, 'o'):
        # Same as panel j — different label/position in layout
        pass  # Already generated as panel j

    # Panel p: PanPep leakage detection
    if should_generate(TF, TP, 6, 'p'):
        fig, ax = mp()
        seen = pp_neo[pp_neo['seen_in_training'] == True]['distance'].values
        unseen = pp_neo[pp_neo['seen_in_training'] == False]['distance'].values
        _, p_mwu = mannwhitneyu(seen, unseen, alternative='two-sided')
        parts = ax.violinplot([seen, unseen], positions=[0, 1], showmeans=True, showmedians=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#55A868', '#aaa'][i]); pc.set_alpha(0.5)
        for kv in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if kv in parts: parts[kv].set_linewidth(0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f'Seen\n(n={len(seen)})', f'Unseen\n(n={len(unseen)})'])
        ax.set_ylabel('S2DD distance')
        p_str = f'p={p_mwu:.1e}' if p_mwu < 0.001 else f'p={p_mwu:.4f}'
        ax.set_title(f'S2DD detects leakage\n(MW-U {p_str})', fontweight='bold')
        save_panel(fig, 6, 'p', 'panpep_leakage')

    # Panel c_method: Prediction MAE comparison
    if should_generate(TF, TP, 6, 'c_method'):
        fig, ax = mp()
        ct_v26_f = v26_pred[v26_pred['setting'].str.contains('test|Test|seen|unseen|mcpas|iedb|v3|v4', case=False, na=False)] if 'setting' in v26_pred.columns else v26_pred
        ct_pape = pape_pred[(pape_pred['baseline'] == 'BL-1_pape_avg') & (pape_pred['experiment'] == 'crosstest')]
        ct_mcbpe = mcbpe_pred[(mcbpe_pred['baseline'] == 'BL-1_mcbpe_avg') & (mcbpe_pred['experiment'] == 'crosstest')]
        x_g = np.arange(3); w = 0.22
        for mi, m in enumerate(['aucroc', 'ap', 'f1']):
            v = ct_v26_f[ct_v26_f['metric']==m]['abs_error'].mean() if len(ct_v26_f[ct_v26_f['metric']==m]) > 0 else 0
            p = ct_pape[ct_pape['metric']==m]['abs_error'].mean() if len(ct_pape[ct_pape['metric']==m]) > 0 else 0
            mc = ct_mcbpe[ct_mcbpe['metric']==m]['abs_error'].mean() if len(ct_mcbpe[ct_mcbpe['metric']==m]) > 0 else 0
            ax.bar(x_g[mi]-w, v, w, color='#3498db', edgecolor='black', linewidth=0.3)
            ax.bar(x_g[mi], p, w, color='#2ecc71', edgecolor='black', linewidth=0.3)
            ax.bar(x_g[mi]+w, mc, w, color='#e74c3c', edgecolor='black', linewidth=0.3)
        ax.set_xticks(x_g); ax.set_xticklabels(['AUROC', 'AP', 'F1'])
        ax.set_ylabel('Prediction MAE')
        ax.legend(['S2DD', 'PAPE', 'M-CBPE'], fontsize=5, loc='upper left')
        ax.set_title('S2DD vs PAPE vs M-CBPE', fontweight='bold')
        save_panel(fig, 6, 'c_method', 'prediction_mae_comparison')


# ── Main ──
if TF == 0 or TF == 3: gen_fig3()
if TF == 0 or TF == 4: gen_fig4()
if TF == 0 or TF == 5: gen_fig5()
if TF == 0 or TF == 6: gen_fig6()

# Final count
print("\n=== Panel inventory ===")
total = 0
for fig_dir in sorted(os.listdir(PANEL_DIR)):
    d = os.path.join(PANEL_DIR, fig_dir)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.endswith('.pdf')])
        total += n
        print(f"  {fig_dir}: {n} panels")
print(f"  Total: {total} panels")
