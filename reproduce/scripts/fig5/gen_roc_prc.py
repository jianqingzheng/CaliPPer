#!/usr/bin/env python3
"""Generate ROC and PRC before/after recalibration panels.

Outputs per model + 5-model overlay:
  fig5_roc_{tcr_ct|bcr_ct}_{model}.pdf    (per-model, single test set)
  fig5_prc_{tcr_ct|bcr_ct}_{model}.pdf
  fig5_roc_{tcr_ct|bcr_ct}_5models.pdf    (5-model overlay)
  fig5_prc_{tcr_ct|bcr_ct}_5models.pdf

Uses v2.7 fit_recalibration/apply_recalibration with adaptive defaults.

TCR CT: v3+v4 cal → Unseen test set (best demonstration of recalibration)
BCR CT: per-variant LOO within SARS/flu, fold4 model (pooled results)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

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

# Per-domain distance: BLOSUM for TCR (≤20 AA), Lev for BCR (≥100 AA)
TCR_DIST_SUFFIX = DIST_SUFFIX['blosum-sqrt']
BCR_DIST_MODE_ACTUAL = BCR_DIST_MODE['lev-log']
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_CAL = ['v3_combined', 'v4_combined']
# Use Unseen as the primary test set for ROC/PRC (largest improvement)
TCR_ROC_TEST = 'unseen_fold34'

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
MIN_SAMPLES = 30
PW, PH = 3.2, 3.0


def save(fig, name):
    for subdir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(_FIG_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


def plot_roc(y, p_before, p_after, title, out_name, color='#1f77b4'):
    """Single-model ROC before/after."""
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    auc_b = roc_auc_score(y, p_before)
    auc_a = roc_auc_score(y, p_after)
    fpr_b, tpr_b, _ = roc_curve(y, p_before)
    fpr_a, tpr_a, _ = roc_curve(y, p_after)
    ax.plot(fpr_b, tpr_b, '--', color=color, linewidth=1.0, alpha=0.5,
            label=f'Before ({auc_b:.3f})')
    ax.plot(fpr_a, tpr_a, '-', color=color, linewidth=1.5,
            label=f'After ({auc_a:.3f}, \u0394={auc_a-auc_b:+.3f})')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('False positive rate', fontsize=9)
    ax.set_ylabel('True positive rate', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')
    save(fig, out_name)


def plot_prc(y, p_before, p_after, title, out_name, color='#1f77b4'):
    """Single-model PRC before/after."""
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    ap_b = average_precision_score(y, p_before)
    ap_a = average_precision_score(y, p_after)
    pr_b, re_b, _ = precision_recall_curve(y, p_before)
    pr_a, re_a, _ = precision_recall_curve(y, p_after)
    ax.plot(re_b, pr_b, '--', color=color, linewidth=1.0, alpha=0.5,
            label=f'Before ({ap_b:.3f})')
    ax.plot(re_a, pr_a, '-', color=color, linewidth=1.5,
            label=f'After ({ap_a:.3f}, \u0394={ap_a-ap_b:+.3f})')
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    save(fig, out_name)


# ═══════════════════════════════════════════
# TCR CT: ROC/PRC per model + 5-model overlay
# ═══════════════════════════════════════════
print("=== TCR CT ROC/PRC ===")

tcr_curves = {}  # model -> (y, p_before, p_after)

for model in TCR_MODELS:
    display = MODEL_DISPLAY.get(model, model)
    ct = {}
    for ts in TCR_CAL + [TCR_ROC_TEST]:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{TCR_DIST_SUFFIX}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ct[ts] = {'y': df[lc].values[:n].astype(int),
                  'p': df[pc].values[:n].astype(float),
                  'd': d[:n].astype(float)}

    if TCR_ROC_TEST not in ct or 'v3_combined' not in ct:
        print(f"  {display}: missing data, skipping")
        continue

    cal_data = {s: (ct[s]['y'], ct[s]['p'], ct[s]['d'])
                for s in TCR_CAL if s in ct}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)
    test = ct[TCR_ROC_TEST]
    cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                ppv_p, npv_p, pp, pn, prev=_cal_prev)

    tcr_curves[model] = (test['y'], test['p'], cal_s)
    color = MODEL_COLORS.get(model, '#888')

    # Per-model panels
    plot_roc(test['y'], test['p'], cal_s,
             f'ROC: {display} (Unseen)', f'fig5_roc_tcr_ct_{model}', color)
    plot_prc(test['y'], test['p'], cal_s,
             f'PRC: {display} (Unseen)', f'fig5_prc_tcr_ct_{model}', color)

# 5-model overlay ROC
if tcr_curves:
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    for model in TCR_MODELS:
        if model not in tcr_curves:
            continue
        y, pb, pa = tcr_curves[model]
        color = MODEL_COLORS.get(model, '#888')
        display = MODEL_DISPLAY.get(model, model)
        auc_b = roc_auc_score(y, pb)
        auc_a = roc_auc_score(y, pa)
        fpr_b, tpr_b, _ = roc_curve(y, pb)
        fpr_a, tpr_a, _ = roc_curve(y, pa)
        ax.plot(fpr_b, tpr_b, color=color, linewidth=1.0, alpha=0.5,
                linestyle=(0, (8, 4)), zorder=3)  # sparse dash, above solid
        ax.plot(fpr_a, tpr_a, '-', color=color, linewidth=1.5, zorder=2,
                label=f'{display} ({auc_b:.2f}\u2192{auc_a:.2f})')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('False positive rate', fontsize=9)
    ax.set_ylabel('True positive rate', fontsize=9)
    ax.set_title('ROC: 5 TCR models (Unseen)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=5, loc='lower right')
    save(fig, 'fig5_roc_tcr_ct_5models')

    # 5-model overlay PRC
    fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
    for model in TCR_MODELS:
        if model not in tcr_curves:
            continue
        y, pb, pa = tcr_curves[model]
        color = MODEL_COLORS.get(model, '#888')
        display = MODEL_DISPLAY.get(model, model)
        ap_b = average_precision_score(y, pb)
        ap_a = average_precision_score(y, pa)
        pr_b, re_b, _ = precision_recall_curve(y, pb)
        pr_a, re_a, _ = precision_recall_curve(y, pa)
        ax.plot(re_b, pr_b, color=color, linewidth=1.0, alpha=0.5,
                linestyle=(0, (8, 4)), zorder=3)
        ax.plot(re_a, pr_a, '-', color=color, linewidth=1.5, zorder=2,
                label=f'{display} ({ap_b:.2f}\u2192{ap_a:.2f})')
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    ax.set_title('PRC: 5 TCR models (Unseen)', fontweight='bold', fontsize=9)
    ax.legend(fontsize=5, loc='upper right')
    save(fig, 'fig5_prc_tcr_ct_5models')


# ═══════════════════════════════════════════
# BCR CT: ROC/PRC per model (pooled SARS+flu per-variant LOO)
# ═══════════════════════════════════════════
print("\n=== BCR CT ROC/PRC ===")

bcr_curves = {}

for model in BCR_MODELS:
    sk = BCR_STYLE_KEY[model]
    display = BCR_MODEL_DISPLAY.get(sk, model)
    model_dir = os.path.join(BCR_FOLD4CAL, model)
    if not os.path.isdir(model_dir):
        print(f"  {display}: no data dir, skipping")
        continue

    cal_path = os.path.join(model_dir, "cal_predictions.csv")


    if not os.path.exists(cal_path):


        print(f"  [BCR] {model}: cal_predictions.csv missing; skipping")


        continue


    cal = pd.read_csv(cal_path)
    if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
        cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
    cal['source'] = 'fold4_test'
    parts = [cal]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
            df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
        df['source'] = ts
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)

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
            cal_data_v = {'cal': (cal_y,
                                  cal_sub['pred_prob'].values.astype(float),
                                  cal_sub['distance'].values.astype(float))}
            test_p = test_sub['pred_prob'].values.astype(float)
            test_d = test_sub['distance'].values.astype(float)
            ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_v)
            cal_s = apply_recalibration(test_y, test_p, test_d,
                                        ppv_p, npv_p, pp, pn)
            all_y.extend(test_y.tolist())
            all_raw.extend(test_p.tolist())
            all_cal.extend(cal_s.tolist())

    if all_y:
        ya = np.array(all_y)
        ra = np.array(all_raw)
        ca = np.array(all_cal)
        bcr_curves[model] = (ya, ra, ca)
        color = BCR_MODEL_COLORS.get(sk, '#888')
        plot_roc(ya, ra, ca, f'ROC: {display} (BCR CT)',
                 f'fig5_roc_bcr_ct_{model}', color)
        plot_prc(ya, ra, ca, f'PRC: {display} (BCR CT)',
                 f'fig5_prc_bcr_ct_{model}', color)

# 5-model overlay
if bcr_curves:
    for curve_type in ['roc', 'prc']:
        fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
        for model in BCR_MODELS:
            if model not in bcr_curves:
                continue
            sk = BCR_STYLE_KEY[model]
            y, pb, pa = bcr_curves[model]
            color = BCR_MODEL_COLORS.get(sk, '#888')
            display = BCR_MODEL_DISPLAY.get(sk, model)
            if curve_type == 'roc':
                auc_b = roc_auc_score(y, pb)
                auc_a = roc_auc_score(y, pa)
                c1x, c1y, _ = roc_curve(y, pb)
                c2x, c2y, _ = roc_curve(y, pa)
                ax.plot(c1x, c1y, color=color, linewidth=1.0, alpha=0.5,
                        linestyle=(0, (8, 4)), zorder=3)
                ax.plot(c2x, c2y, '-', color=color, linewidth=1.5, zorder=2,
                        label=f'{display} ({auc_b:.2f}\u2192{auc_a:.2f})')
            else:
                ap_b = average_precision_score(y, pb)
                ap_a = average_precision_score(y, pa)
                pr_b, re_b, _ = precision_recall_curve(y, pb)
                pr_a, re_a, _ = precision_recall_curve(y, pa)
                ax.plot(re_b, pr_b, color=color, linewidth=1.0, alpha=0.5,
                        linestyle=(0, (8, 4)), zorder=3)
                ax.plot(re_a, pr_a, '-', color=color, linewidth=1.5, zorder=2,
                        label=f'{display} ({ap_b:.2f}\u2192{ap_a:.2f})')

        if curve_type == 'roc':
            ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('False positive rate', fontsize=9)
            ax.set_ylabel('True positive rate', fontsize=9)
            ax.set_title('ROC: 5 BCR models (CT)', fontweight='bold', fontsize=9)
        else:
            ax.set_xlabel('Recall', fontsize=9)
            ax.set_ylabel('Precision', fontsize=9)
            ax.set_title('PRC: 5 BCR models (CT)', fontweight='bold', fontsize=9)
        ax.legend(fontsize=5, loc='lower right' if curve_type == 'roc' else 'upper right')
        save(fig, f'fig5_{curve_type}_bcr_ct_5models')


print("\n=== All ROC/PRC panels generated ===")
