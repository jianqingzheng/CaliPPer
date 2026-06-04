#!/usr/bin/env python3
"""Generate scatter-with-marginals panels: raw vs recalibrated predictions.

Generates one panel per model x domain:
  fig5_scatter_marginals_{tcr_ct|tcr_cv|bcr_ct|bcr_cv}_{model}.pdf

Uses v2.7 fit_recalibration/apply_recalibration from General_Eval/s2dd.py.

TCR CT: v3+v4 calibration -> 4 test sets pooled per model
TCR CV: within-fold half-split (5 folds pooled)
BCR CT: per-variant LOO within SARS/flu, fold4 model
BCR CV: within-fold half-split (5 folds pooled, XBCR-net only currently)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

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
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

# Per-domain distance: BLOSUM for TCR (≤20 AA), Lev for BCR (≥100 AA)
TCR_DIST_SUFFIX = DIST_SUFFIX['blosum-sqrt']
BCR_DIST_MODE_ACTUAL = BCR_DIST_MODE['lev-log']

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')

# ---- Constants ----
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_CAL_SETS = ['v3_combined', 'v4_combined']
TCR_TEST_SETS = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
TCR_ALL_SETS = TCR_CAL_SETS + TCR_TEST_SETS

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
BCR_CV_DIR_FMT = os.path.join(RESULTS, '{model}', 'combined_bind_ab_cv')  # per-model
BCR_CV_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
MIN_SAMPLES = 30
MAX_PTS = 8000


def save(fig, name):
    fig.savefig(os.path.join(PANEL_DIR, name + '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PANEL_DIR, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ---- Plotting function (adapted from new_fig5_recalibration.py panel a/b) ----
def plot_scatter_marginals(y, p_before, p_after, title, out_name,
                           color_pos='#CC3300', color_neg='#2E8B57',
                           adaptive_range=False):
    """Scatter of raw (x) vs recalibrated (y) predictions with marginal KDEs."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    rng = np.random.default_rng(42)
    prev = y.mean()

    mask_neg = y == 0
    mask_pos = y == 1
    pb_n, pa_n = p_before[mask_neg], p_after[mask_neg]
    pb_p, pa_p = p_before[mask_pos], p_after[mask_pos]

    if len(pb_n) > MAX_PTS:
        idx = rng.choice(len(pb_n), MAX_PTS, replace=False)
        pb_n, pa_n = pb_n[idx], pa_n[idx]
    if len(pb_p) > MAX_PTS:
        idx = rng.choice(len(pb_p), MAX_PTS, replace=False)
        pb_p, pa_p = pb_p[idx], pa_p[idx]

    ax.scatter(pb_n, pa_n, c=color_neg, s=6, alpha=0.3, edgecolors='none',
              rasterized=True, label='Non-binder')
    ax.scatter(pb_p, pa_p, c=color_pos, s=6, alpha=0.3, edgecolors='none',
              rasterized=True, label='Binder')

    # Determine axis ranges (independent x/y for adaptive)
    if adaptive_range:
        x_lo = max(0, np.percentile(p_before, 0.5) - 0.03)
        x_hi = min(1, np.percentile(p_before, 99.5) + 0.03)
        y_lo = max(0, np.percentile(p_after, 0.5) - 0.03)
        y_hi = min(1, np.percentile(p_after, 99.5) + 0.03)
    else:
        x_lo, x_hi, y_lo, y_hi = 0, 1, 0, 1

    # Diagonal (only within overlapping range)
    diag_lo = max(x_lo, y_lo); diag_hi = min(x_hi, y_hi)
    if diag_lo < diag_hi:
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], 'k--', linewidth=0.5, alpha=0.3)
    ax.axhline(y=prev, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(x=prev, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    # Marginal KDEs
    xg = np.linspace(x_lo, x_hi, 200)
    ax_top = ax.inset_axes([0, 1.0, 1, 0.15], sharex=ax)
    for data, c in [(p_before[mask_neg], color_neg), (p_before[mask_pos], color_pos)]:
        if len(data) > 20:
            try:
                kde = gaussian_kde(data, bw_method=0.05)
                ax_top.fill_between(xg, kde(xg), color=c, alpha=0.4)
            except Exception as _e_kde:
                import sys as _s_kde
                print(f"  ⚠ FALLBACK [scatter-marginals KDE-top]: gaussian_kde failed ({type(_e_kde).__name__}: {_e_kde}); skipping marginal", file=_s_kde.stderr, flush=True)
    ax_top.set_xlim(x_lo, x_hi); ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False)
    for sp in ax_top.spines.values(): sp.set_visible(False)

    yg = np.linspace(y_lo, y_hi, 200)
    ax_right = ax.inset_axes([1.0, 0, 0.15, 1], sharey=ax)
    for data, c in [(p_after[mask_neg], color_neg), (p_after[mask_pos], color_pos)]:
        if len(data) > 20:
            try:
                kde = gaussian_kde(data, bw_method=0.05)
                ax_right.fill_betweenx(yg, kde(yg), color=c, alpha=0.4)
            except Exception as _e_kde2:
                import sys as _s_kde2
                print(f"  ⚠ FALLBACK [scatter-marginals KDE-right]: gaussian_kde failed ({type(_e_kde2).__name__}: {_e_kde2}); skipping marginal", file=_s_kde2.stderr, flush=True)
    ax_right.set_xticks([])
    ax_right.tick_params(labelleft=False)
    for sp in ax_right.spines.values(): sp.set_visible(False)

    auc_b = safe_metric('aucroc', y, p_before)
    auc_a = safe_metric('aucroc', y, p_after)
    rho = spearmanr(p_before, p_after)[0]
    ax.text(0.03, 0.97,
            f'AUROC: {auc_b:.3f}\u2192{auc_a:.3f}\n'
            f'\u0394AUC={auc_a-auc_b:+.3f}, \u03c1={rho:.2f}',
            transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    ax.set_xlim(x_lo - 0.02, x_hi + 0.02); ax.set_ylim(y_lo - 0.02, y_hi + 0.02)
    if not adaptive_range:
        ax.set_box_aspect(1)
    ax.set_xlabel('Before recalibration', fontsize=9)
    ax.set_ylabel('After recalibration', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='lower right', framealpha=0.9)

    save(fig, out_name)


# ═══════════════════════════════════════════
# TCR CT: v3+v4 calibration -> 4 test sets pooled
# ═══════════════════════════════════════════
print("=== TCR CT scatter-with-marginals ===")

for model in TCR_MODELS:
    display = MODEL_DISPLAY.get(model, model)
    ct_model = {}
    for ts in TCR_ALL_SETS:
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
        ct_model[ts] = {'y': df[lc].values[:n].astype(int),
                        'p': df[pc].values[:n].astype(float),
                        'd': d[:n].astype(float)}

    if 'v3_combined' not in ct_model or 'v4_combined' not in ct_model:
        print(f"  {display}: missing cal sets, skipping")
        continue

    cal_data = {s: (ct_model[s]['y'], ct_model[s]['p'], ct_model[s]['d'])
                for s in TCR_CAL_SETS}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

    # Panel C: unseen epitopes only (strongest recalibration signal)
    # Cal still uses v3+v4; test = unseen_fold34 only
    TCR_SCATTER_SETS = ['unseen_fold34']
    all_y, all_raw, all_cal = [], [], []
    for ts in TCR_SCATTER_SETS:
        if ts not in ct_model:
            continue
        test = ct_model[ts]
        cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                    ppv_p, npv_p, pp, pn, prev=_cal_prev)
        all_y.extend(test['y'].tolist())
        all_raw.extend(test['p'].tolist())
        all_cal.extend(cal_s.tolist())

    if all_y:
        plot_scatter_marginals(np.array(all_y), np.array(all_raw), np.array(all_cal),
                               f'TCR-unseen ({display})',
                               f'fig5_scatter_marginals_tcr_ct_{model}',
                               adaptive_range=True)


# ═══════════════════════════════════════════
# BCR CT: per-variant LOO, fold4 model
# ═══════════════════════════════════════════
print("\n=== BCR CT scatter-with-marginals ===")

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

    # Panel D: unseen SARS variants only (strongest recalibration signal)
    # Per-variant LOO: hold out one unseen variant, cal = all other SARS
    unseen_df = pooled[(pooled['data_source'] == 'sars') & (pooled['source'] == 'unseen')]
    sars_pool = pooled[pooled['data_source'] == 'sars']
    all_y, all_raw, all_cal = [], [], []
    variants = unseen_df.groupby('variant_seq').size()
    valid = variants[variants >= MIN_SAMPLES].index.tolist()
    for held_v in valid:
        test_sub = unseen_df[unseen_df['variant_seq'] == held_v]
        cal_sub = sars_pool[sars_pool['variant_seq'] != held_v]
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
        cal_s = apply_recalibration(test_y, test_p, test_d, ppv_p, npv_p, pp, pn)
        all_y.extend(test_y.tolist())
        all_raw.extend(test_p.tolist())
        all_cal.extend(cal_s.tolist())

    if all_y:
        plot_scatter_marginals(np.array(all_y), np.array(all_raw), np.array(all_cal),
                               f'BCR-unseen ({display})',
                               f'fig5_scatter_marginals_bcr_ct_{model}')


# ═══════════════════════════════════════════
# BCR CV: within-fold half-split
# ═══════════════════════════════════════════
print("\n=== BCR CV scatter-with-marginals ===")

for model in BCR_MODELS:
    sk = BCR_STYLE_KEY[model]
    display = BCR_MODEL_DISPLAY.get(sk, model)

    all_y, all_raw, all_cal = [], [], []
    bcr_cv_dir = BCR_CV_DIR_FMT.format(model=model)
    for fold in range(5):
        test_path = os.path.join(bcr_cv_dir, f'fold{fold}', 'test.csv')
        if not os.path.exists(test_path):
            continue
        df = pd.read_csv(test_path)
        if 'pred_prob' not in df.columns or 'distance' not in df.columns or 'rbd' not in df.columns:
            continue

        y = df['rbd'].values.astype(int)
        p = df['pred_prob'].values.astype(float)
        if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
            npy_path = os.path.join(BCR_CV_DIST_CACHE, f'{model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
            if os.path.exists(npy_path):
                d = np.load(npy_path).astype(float)[:len(df)]
            else:
                d = df['distance'].values.astype(float)
        else:
            d = df['distance'].values.astype(float)

        si = np.argsort(d)
        cal_idx = si[::2]
        val_idx = si[1::2]
        cal_y, cal_p, cal_d = y[cal_idx], p[cal_idx], d[cal_idx]
        val_y, val_p, val_d = y[val_idx], p[val_idx], d[val_idx]

        if cal_y.sum() < 3 or (len(cal_y) - cal_y.sum()) < 3:
            continue

        cal_data_f = {'cal': (cal_y, cal_p, cal_d)}
        ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_f)
        cal_s = apply_recalibration(val_y, val_p, val_d, ppv_p, npv_p, pp, pn)
        all_y.extend(val_y.tolist())
        all_raw.extend(val_p.tolist())
        all_cal.extend(cal_s.tolist())

    if all_y:
        plot_scatter_marginals(np.array(all_y), np.array(all_raw), np.array(all_cal),
                               f'BCR CV ({display})',
                               f'fig5_scatter_marginals_bcr_cv_{model}')


# ═══════════════════════════════════════════
# TCR CV: within-fold half-split
# ═══════════════════════════════════════════
print("\n=== TCR CV scatter-with-marginals ===")

for model in TCR_MODELS:
    display = MODEL_DISPLAY.get(model, model)

    all_y, all_raw, all_cal = [], [], []
    for fold in range(5):
        pred_path = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}',
                                 'predictions_with_label.csv')
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'

        if DIST_TYPE == 'blosum-sqrt':
            suffixes = ['_blosumsqrt_dist']
        else:
            suffixes = ['_combined_dist', '_dist']
        dist_path = None
        for sfx in suffixes:
            dp = os.path.join(TCR_DIST_CACHE, f'{model}_cv_fold{fold}{sfx}.npy')
            if os.path.exists(dp):
                dist_path = dp
                break
        if dist_path is None:
            continue
        d = np.load(dist_path)
        n = min(len(d), len(df))
        y = df[lc].values[:n].astype(int)
        p = df[pc].values[:n].astype(float)
        d = d[:n].astype(float)

        si = np.argsort(d)
        cal_idx = si[::2]
        val_idx = si[1::2]
        cal_y, cal_p, cal_d = y[cal_idx], p[cal_idx], d[cal_idx]
        val_y, val_p, val_d = y[val_idx], p[val_idx], d[val_idx]

        if cal_y.sum() < 3 or (len(cal_y) - cal_y.sum()) < 3:
            continue

        cal_data_f = {'cal': (cal_y, cal_p, cal_d)}
        ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_f)
        cal_s = apply_recalibration(val_y, val_p, val_d, ppv_p, npv_p, pp, pn)
        all_y.extend(val_y.tolist())
        all_raw.extend(val_p.tolist())
        all_cal.extend(cal_s.tolist())

    if all_y:
        plot_scatter_marginals(np.array(all_y), np.array(all_raw), np.array(all_cal),
                               f'TCR CV ({display})',
                               f'fig5_scatter_marginals_tcr_cv_{model}')


print("\n=== All scatter-with-marginals panels generated ===")
