#!/usr/bin/env python3
"""Generate new Fig 5 panels for 4x4 recalibration figure.

Panels generated (Row 2-4; Row 1 = existing a/b/c/d):
  (e) Combined 10-model dumbbell (TCR top + BCR bottom)
  (f) Dataset-level AUROC before vs after scatter
  (g) Dataset-level AP before vs after scatter
  (h) PPV/NPV curves (reuse existing — not regenerated here)
  (i) Subset AUROC before vs after (per-epitope TCR + per-variant BCR)
  (j) Subset AP before vs after
  (k) ΔAUROC vs raw AUROC ("worse gain more")
  (l) ΔAP vs raw AP
  (m) Per-bin ΔAUROC vs distance ("far samples gain most")
  (n) ΔAUROC vs mean distance per epitope/variant ("far epitopes gain more")
  (o) BCR SARS vs FLU domain bars
  (p) ΔAUROC heatmap (5 TCR models × 4 test sets)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
_FIG_DIR = os.path.join(FIG_DIR, 'fig5')
os.makedirs(_FIG_DIR, exist_ok=True)
from style_config import (apply_publication_style, MODEL_COLORS, MODEL_DISPLAY,
                           BCR_MODEL_COLORS, BCR_MODEL_DISPLAY, DPI)
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE, get_bcr_ct_distance

apply_publication_style()

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
RESULTS = os.path.join(INPUT_DIR, 'results')
os.makedirs(PANEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
TCR_CAL_SETS = ['v3_combined', 'v4_combined']
TCR_TEST_SETS = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
TCR_ALL_SETS = TCR_CAL_SETS + TCR_TEST_SETS
TCR_TS_DISP = {'seen_test': 'Seen', 'unseen_fold34': 'Unseen',
               'mcpas': 'McPAS', 'iedb_sars': 'IEDB'}

BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
BCR_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
                 'mint': 'mint', 'rleaai': 'rleaai'}
MIN_SAMPLES = 30
N_BINS = 8
TCR_COLOR = '#1f77b4'
BCR_COLOR = '#ff7f0e'


def save(fig, name):
    fig.savefig(os.path.join(PANEL_DIR, name + '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PANEL_DIR, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ═══════════════════════════════════════════
# Load and compute TCR recalibration
# ═══════════════════════════════════════════
print("Loading TCR data...")

# Store per-test-set results AND per-epitope results
tcr_dataset_results = []   # (model, test_set, metric, before, after)
tcr_epitope_results = []   # (model, test_set, epitope, metric, before, after, n, mean_d)
tcr_perbin_results = []    # (model, test_set, bin_idx, d_center, delta_auroc)
tcr_heatmap = {}           # model -> {test_set -> delta_auroc}

for model in TCR_MODELS:
    ct_model = {}
    for ts in TCR_ALL_SETS:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        # TCR recalibration uses BLOSUM-sqrt (optimal for recalibration)
        tcr_dist_suffix = DIST_SUFFIX.get('blosum-sqrt', DIST_SUFFIX[DIST_TYPE])
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{tcr_dist_suffix}')
        # Levenshtein distances for per-bin x-axis (comparability with BCR panel n)
        lev_dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}_dist.npy')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        d_lev = np.load(lev_dist_path) if os.path.exists(lev_dist_path) else d
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
        ct_model[ts] = {'y': df[lc].values[:n].astype(int),
                        'p': df[pc].values[:n].astype(float),
                        'd': d[:n].astype(float),
                        'd_lev': d_lev[:n].astype(float),
                        'ep': df[ep_col].values[:n] if ep_col in df.columns else None}

    if 'v3_combined' not in ct_model or 'v4_combined' not in ct_model:
        print(f"  [TCR] {model}: missing cal sets, skipping")
        continue

    cal_data = {s: (ct_model[s]['y'], ct_model[s]['p'], ct_model[s]['d'])
                for s in TCR_CAL_SETS}
    ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(cal_data)
    tcr_heatmap[model] = {}

    for ts in TCR_TEST_SETS:
        if ts not in ct_model:
            continue
        test = ct_model[ts]
        cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                    ppv_p, npv_p, pp, pn, prev=cal_prev)

        # Dataset-level
        for metric in ['aucroc', 'ap']:
            before = safe_metric(metric, test['y'], test['p'])
            after = safe_metric(metric, test['y'], cal_s)
            tcr_dataset_results.append((model, ts, metric, before, after))

        auc_b = safe_metric('aucroc', test['y'], test['p'])
        auc_a = safe_metric('aucroc', test['y'], cal_s)
        tcr_heatmap[model][ts] = auc_a - auc_b

        # Per-epitope
        if test['ep'] is not None:
            for ep in np.unique(test['ep']):
                mask = test['ep'] == ep
                if mask.sum() < MIN_SAMPLES:
                    continue
                yi = test['y'][mask]
                if yi.sum() == 0 or yi.sum() == len(yi):
                    continue
                for metric in ['aucroc', 'ap']:
                    b = safe_metric(metric, yi, test['p'][mask])
                    a = safe_metric(metric, yi, cal_s[mask])
                    mean_d = test['d'][mask].mean()
                    tcr_epitope_results.append((model, ts, ep, metric, b, a, mask.sum(), mean_d))

        # Per-bin ΔAUROC — Lev recal + Lev bin (panel m only, consistent distance)
        # Separate from BLOSUM recalibration used for panels k/l/i
        cal_data_lev = {s: (ct_model[s]['y'], ct_model[s]['p'], ct_model[s]['d_lev'])
                        for s in TCR_CAL_SETS if s in ct_model}
        ppv_p_lev, npv_p_lev, pp_lev, pn_lev, cp_lev = fit_recalibration(cal_data_lev)
        cal_s_lev = apply_recalibration(test['y'], test['p'], test['d_lev'],
                                        ppv_p_lev, npv_p_lev, pp_lev, pn_lev, prev=cp_lev)
        si = np.argsort(test['d_lev'])
        bs = len(si) // N_BINS
        for bi in range(N_BINS):
            s = bi * bs
            e = len(si) if bi == N_BINS - 1 else (bi + 1) * bs
            idx = si[s:e]
            yi = test['y'][idx]
            if yi.sum() == 0 or yi.sum() == len(yi):
                continue
            ab = safe_metric('aucroc', yi, test['p'][idx])
            aa = safe_metric('aucroc', yi, cal_s_lev[idx])
            tcr_perbin_results.append((model, ts, bi, test['d_lev'][idx].mean(), aa - ab))

    print(f"  [TCR] {MODEL_DISPLAY[model]}: {len([r for r in tcr_dataset_results if r[0]==model])} dataset entries")


# ═══════════════════════════════════════════
# Load and compute BCR recalibration
# ═══════════════════════════════════════════
print("\nLoading BCR data...")

bcr_dataset_results = []   # (model, domain, metric, before, after)
bcr_variant_results = []   # (model, domain, variant, metric, before, after, n, mean_d)

bcr_model_pooled = {}  # model -> (pooled_before, pooled_after) for dumbbell
bcr_domain_deltas = {} # model -> {domain -> delta_auroc}

for model in BCR_MODELS:
    display = BCR_MODEL_DISPLAY.get(BCR_STYLE_KEY[model], model)
    model_dir = os.path.join(BCR_FOLD4CAL, model)
    if not os.path.isdir(model_dir):
        print(f"  [BCR] {display}: no data dir, skipping")
        continue

    cal_path = os.path.join(model_dir, 'cal_predictions.csv')
    if not os.path.exists(cal_path):
        print(f"  [BCR] {display}: cal_predictions.csv missing (deposit gap "
              f"— see retrain_fig3_inputs.sh --model bcr_ct_fold4cal); skipping")
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

    all_y_pool, all_raw_pool, all_cal_pool = [], [], []
    bcr_domain_deltas[model] = {}

    for domain in ['sars', 'flu']:
        domain_df = pooled[pooled['data_source'] == domain]
        variants = domain_df.groupby('variant_seq').size()
        valid = variants[variants >= MIN_SAMPLES].index.tolist()

        dom_y, dom_raw, dom_cal = [], [], []
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

            ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(cal_data_v)
            cal_s = apply_recalibration(test_y, test_p, test_d, ppv_p, npv_p, pp, pn)

            # Per-variant
            for metric in ['aucroc', 'ap']:
                b = safe_metric(metric, test_y, test_p)
                a = safe_metric(metric, test_y, cal_s)
                mean_d = test_d.mean()
                bcr_variant_results.append((model, domain, held_v, metric, b, a,
                                           len(test_y), mean_d))

            dom_y.extend(test_y.tolist())
            dom_raw.extend(test_p.tolist())
            dom_cal.extend(cal_s.tolist())

        if dom_y:
            dom_y_a, dom_raw_a, dom_cal_a = np.array(dom_y), np.array(dom_raw), np.array(dom_cal)
            for metric in ['aucroc', 'ap']:
                b = safe_metric(metric, dom_y_a, dom_raw_a)
                a = safe_metric(metric, dom_y_a, dom_cal_a)
                bcr_dataset_results.append((model, domain, metric, b, a))
            bcr_domain_deltas[model][domain] = (safe_metric('aucroc', dom_y_a, dom_raw_a),
                                                 safe_metric('aucroc', dom_y_a, dom_cal_a))
            all_y_pool.extend(dom_y)
            all_raw_pool.extend(dom_raw)
            all_cal_pool.extend(dom_cal)

    if all_y_pool:
        orig = safe_metric('aucroc', np.array(all_y_pool), np.array(all_raw_pool))
        recal = safe_metric('aucroc', np.array(all_y_pool), np.array(all_cal_pool))
        bcr_model_pooled[model] = (orig, recal)
        print(f"  [BCR] {display}: {orig:.3f}→{recal:.3f} Δ={recal-orig:+.3f}")


# Also get TCR model pooled for dumbbell
tcr_model_pooled = {}
for model in TCR_MODELS:
    rows = [(b, a) for m, ts, met, b, a in tcr_dataset_results
            if m == model and met == 'aucroc']
    if rows:
        tcr_model_pooled[model] = (np.mean([r[0] for r in rows]),
                                   np.mean([r[1] for r in rows]))


# ═══════════════════════════════════════════
# Panel (e): Combined 10-model dumbbell
# ═══════════════════════════════════════════
print("\n=== Generating panels ===")

fig, (ax_tcr, ax_bcr) = plt.subplots(2, 1, figsize=(4.0, 4.0),
                                      gridspec_kw={'hspace': 0.35})

def plot_dumbbell(ax, results, style_key_map, color_map, display_map, title):
    sorted_items = sorted(results.items(), key=lambda x: x[1][1] - x[1][0], reverse=True)
    yp = np.arange(len(sorted_items))[::-1]
    for i, (model, (before, after)) in enumerate(sorted_items):
        sk = style_key_map.get(model, model)
        color = color_map.get(sk, '#888')
        delta = after - before
        ax.plot([before, after], [yp[i], yp[i]], color=color, linewidth=2.5,
                solid_capstyle='round', alpha=0.6)
        ax.scatter(before, yp[i], color='white', edgecolor=color, s=40, zorder=5, linewidth=1.0)
        ax.scatter(after, yp[i], color=color, s=45, zorder=5, edgecolor='white', linewidth=0.5)
        ax.text(max(before, after) + 0.008, yp[i], f'{delta:+.3f}', va='center',
                fontsize=7, color=color, fontweight='bold')
    ax.axvline(0.5, color='gray', linewidth=0.4, linestyle=':', alpha=0.5)
    ax.set_yticks(yp)
    ax.set_yticklabels([display_map.get(style_key_map.get(m, m), m)
                        for m, _ in sorted_items], fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=9, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return [v for b, a in results.values() for v in (b, a)]

tcr_sk = {m: m for m in TCR_MODELS}
tcr_vals = plot_dumbbell(ax_tcr, tcr_model_pooled, tcr_sk, MODEL_COLORS,
                         MODEL_DISPLAY, 'TCR–epitope (5 models)')
ax_tcr.set_xticklabels([])
bcr_vals = plot_dumbbell(ax_bcr, bcr_model_pooled, BCR_STYLE_KEY, BCR_MODEL_COLORS,
                         BCR_MODEL_DISPLAY, 'BCR–antigen (5 models)')
ax_bcr.set_xlabel('AUROC', fontsize=9)
all_vals = tcr_vals + bcr_vals
ax_tcr.set_xlim(min(all_vals) - 0.03, max(all_vals) + 0.08)
ax_bcr.set_xlim(min(all_vals) - 0.03, max(all_vals) + 0.08)
from matplotlib.lines import Line2D
ax_tcr.legend(handles=[
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
           markeredgecolor='gray', markersize=5, linewidth=0, label='Before'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
           markersize=5, linewidth=0, label='After recalib.'),
], loc='lower right', fontsize=6, frameon=True, framealpha=0.9)
save(fig, 'fig5_combined_recal_dumbbell')


# ═══════════════════════════════════════════
# Panel (f)/(g): Dataset-level before vs after scatter (AUROC / AP)
# ═══════════════════════════════════════════
for metric, panel_label in [('aucroc', 'f'), ('ap', 'g')]:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    # TCR points
    tcr_b = [b for m, ts, met, b, a in tcr_dataset_results if met == metric]
    tcr_a = [a for m, ts, met, b, a in tcr_dataset_results if met == metric]
    # BCR points
    bcr_b = [b for m, d, met, b, a in bcr_dataset_results if met == metric]
    bcr_a = [a for m, d, met, b, a in bcr_dataset_results if met == metric]

    ax.scatter(tcr_b, tcr_a, c=TCR_COLOR, s=25, alpha=0.7, label='TCR', edgecolor='white', linewidth=0.3)
    ax.scatter(bcr_b, bcr_a, c=BCR_COLOR, s=25, alpha=0.7, label='BCR', edgecolor='white', linewidth=0.3)

    # Identity line
    lims = [min(tcr_b + bcr_b + tcr_a + bcr_a) - 0.05,
            max(tcr_b + bcr_b + tcr_a + bcr_a) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)

    # Count improved
    all_b = tcr_b + bcr_b
    all_a = tcr_a + bcr_a
    improved = sum(1 for b, a in zip(all_b, all_a) if a > b)
    total = len(all_b)
    ax.text(0.05, 0.95, f'Improved: {improved}/{total} ({100*improved/total:.0f}%)',
            transform=ax.transAxes, fontsize=7, va='top')

    ax.set_xlabel(f'{metric.upper()} before', fontsize=9)
    ax.set_ylabel(f'{metric.upper()} after', fontsize=9)
    ax.set_title(f'Dataset-level {metric.upper()}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, f'fig5_dataset_{metric}_before_after')


# ═══════════════════════════════════════════
# Panel (k)/(l): Dataset-level before vs after scatter WITH CV halfsplit
# CT = solid larger dots, CV = light smaller dots
# ═══════════════════════════════════════════
print("\n=== Computing CV halfsplit recalibration for panels K/L ===")

# TCR CV halfsplit
tcr_cv_results = []  # (model, fold, metric, before, after)
for model in TCR_MODELS:
    for fold in range(5):
        pred_path = os.path.join(RESULTS, model, 'cv_logdist',
                                  f'fold{fold}', 'test_predictions_with_label.csv')
        tcr_dist_suffix = DIST_SUFFIX.get('blosum-sqrt', DIST_SUFFIX[DIST_TYPE])
        dist_path = os.path.join(TCR_DIST_CACHE, f'{model}_cv_fold{fold}{tcr_dist_suffix}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path); d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        y = df[lc].values[:n].astype(int)
        p = df[pc].values[:n].astype(float)
        d = d[:n]
        # Halfsplit: sort by distance, alternating cal/test
        si = np.argsort(d)
        cal_idx = si[0::2]; test_idx = si[1::2]
        if len(cal_idx) < 50 or len(test_idx) < 50:
            continue
        cal_data_cv = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
        ppv_cv, npv_cv, pp_cv, pn_cv, cp_cv = fit_recalibration(cal_data_cv)
        cal_s = apply_recalibration(y[test_idx], p[test_idx], d[test_idx],
                                     ppv_cv, npv_cv, pp_cv, pn_cv, prev=cp_cv)
        for metric in ['aucroc', 'ap']:
            b = safe_metric(metric, y[test_idx], p[test_idx])
            a = safe_metric(metric, y[test_idx], cal_s)
            tcr_cv_results.append((model, fold, metric, b, a))
print(f"  TCR CV: {len(tcr_cv_results)} results")

# BCR CV halfsplit
bcr_cv_results = []  # (model, fold, metric, before, after)
for model in BCR_MODELS:
    for fold in range(5):
        test_path = os.path.join(RESULTS, model if model != 'xbcr' else 'xbcr',
                                  'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
        if not os.path.exists(test_path):
            continue
        df = pd.read_csv(test_path)
        if 'pred_prob' not in df.columns or 'distance' not in df.columns:
            continue
        y = df['rbd'].values.astype(int)
        p = df['pred_prob'].values.astype(float)
        d = df['distance'].values.astype(float)
        si = np.argsort(d)
        cal_idx = si[0::2]; test_idx = si[1::2]
        if len(cal_idx) < 50:
            continue
        cal_data_cv = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
        ppv_cv, npv_cv, pp_cv, pn_cv, cp_cv = fit_recalibration(cal_data_cv)
        cal_s = apply_recalibration(y[test_idx], p[test_idx], d[test_idx],
                                     ppv_cv, npv_cv, pp_cv, pn_cv, prev=cp_cv)
        for metric in ['aucroc', 'ap']:
            b = safe_metric(metric, y[test_idx], p[test_idx])
            a = safe_metric(metric, y[test_idx], cal_s)
            bcr_cv_results.append((model, fold, metric, b, a))
print(f"  BCR CV: {len(bcr_cv_results)} results")

# Plot panels K/L
for metric in ['aucroc', 'ap']:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))

    # CV points (light, smaller) — plot first so CT is on top
    tcr_cv_b = [b for _, _, m, b, a in tcr_cv_results if m == metric]
    tcr_cv_a = [a for _, _, m, b, a in tcr_cv_results if m == metric]
    bcr_cv_b = [b for _, _, m, b, a in bcr_cv_results if m == metric]
    bcr_cv_a = [a for _, _, m, b, a in bcr_cv_results if m == metric]
    ax.scatter(tcr_cv_b, tcr_cv_a, c=TCR_COLOR, s=12, alpha=0.35, edgecolor='none')
    ax.scatter(bcr_cv_b, bcr_cv_a, c=BCR_COLOR, s=14, alpha=0.35, marker='s', edgecolor='none')

    # CT points (solid, larger)
    tcr_ct_b = [b for _, _, m, b, a in tcr_dataset_results if m == metric]
    tcr_ct_a = [a for _, _, m, b, a in tcr_dataset_results if m == metric]
    bcr_ct_b = [b for _, d, m, b, a in bcr_dataset_results if m == metric]
    bcr_ct_a = [a for _, d, m, b, a in bcr_dataset_results if m == metric]
    ax.scatter(tcr_ct_b, tcr_ct_a, c=TCR_COLOR, s=25, alpha=0.8,
               edgecolor='white', linewidth=0.3)
    ax.scatter(bcr_ct_b, bcr_ct_a, c=BCR_COLOR, s=25, alpha=0.8,
               edgecolor='white', linewidth=0.3)

    # Identity line
    all_vals = tcr_cv_b + tcr_cv_a + tcr_ct_b + tcr_ct_a + bcr_cv_b + bcr_cv_a + bcr_ct_b + bcr_ct_a
    lims = [min(all_vals) - 0.05, max(all_vals) + 0.05] if all_vals else [0.3, 1.0]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)

    # Count improved separately for CT and CV
    ct_improved = sum(1 for b, a in zip(tcr_ct_b + bcr_ct_b, tcr_ct_a + bcr_ct_a) if a > b)
    ct_total = len(tcr_ct_b) + len(bcr_ct_b)
    cv_improved = sum(1 for b, a in zip(tcr_cv_b + bcr_cv_b, tcr_cv_a + bcr_cv_a) if a > b)
    cv_total = len(tcr_cv_b) + len(bcr_cv_b)
    # Defensive: when no CV data (TCR CV fold predictions + BCR CV both absent
    # from deposit) format "N/A" instead of ZeroDivisionError. Same for CT.
    ct_pct = f'{100*ct_improved/ct_total:.0f}%' if ct_total > 0 else 'N/A'
    cv_pct = f'{100*cv_improved/cv_total:.0f}%' if cv_total > 0 else 'N/A'
    ax.text(0.03, 0.97,
            f'Improved ↑\n'
            f'CT: {ct_improved}/{ct_total} ({ct_pct})\n'
            f'CV: {cv_improved}/{cv_total} ({cv_pct})',
            transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#ccc'))

    ax.set_xlabel(f'{metric.upper()} before', fontsize=8)
    ax.set_ylabel(f'{metric.upper()} after', fontsize=8)
    ax.set_title(f'Dataset-level {metric.upper()}', fontweight='bold', fontsize=9)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TCR_COLOR,
                    markersize=4, alpha=0.35, label='TCR CV'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=BCR_COLOR,
                    markersize=4, alpha=0.35, label='BCR CV'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TCR_COLOR,
                    markersize=5, alpha=0.8, label='TCR CT'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BCR_COLOR,
                    markersize=5, alpha=0.8, label='BCR CT'),
    ], fontsize=5, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, f'fig5_dataset_{metric}_before_after_with_cv')


# ═══════════════════════════════════════════
# Panel (i)/(j): Subset (epitope/variant) before vs after scatter
# ═══════════════════════════════════════════
SARS_COLOR = '#e74c3c'
FLU_COLOR = '#3498db'

for metric, panel_label in [('aucroc', 'i'), ('ap', 'j')]:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    # TCR per-epitope
    tcr_b = [b for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == metric]
    tcr_a = [a for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == metric]
    # BCR per-variant — split by domain
    bcr_sars_b = [b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'sars']
    bcr_sars_a = [a for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'sars']
    bcr_flu_b = [b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'flu']
    bcr_flu_a = [a for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'flu']

    ax.scatter(tcr_b, tcr_a, c=TCR_COLOR, s=12, alpha=0.4, label=f'TCR (n={len(tcr_b)})',
              edgecolor='none')
    ax.scatter(bcr_sars_b, bcr_sars_a, c=SARS_COLOR, s=15, alpha=0.5, marker='s',
              label=f'BCR SARS (n={len(bcr_sars_b)})', edgecolor='none')
    ax.scatter(bcr_flu_b, bcr_flu_a, c=FLU_COLOR, s=15, alpha=0.5, marker='D',
              label=f'BCR Flu (n={len(bcr_flu_b)})', edgecolor='none')

    all_b = tcr_b + bcr_sars_b + bcr_flu_b
    all_a = tcr_a + bcr_sars_a + bcr_flu_a
    lims = [min(all_b + all_a) - 0.05, max(all_b + all_a) + 0.05]
    lims = [max(lims[0], -0.05), min(lims[1], 1.05)]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)

    improved = sum(1 for b, a in zip(all_b, all_a) if a > b)
    total = len(all_b)
    ax.text(0.05, 0.95, f'Improved: {improved}/{total} ({100*improved/total:.0f}%)',
            transform=ax.transAxes, fontsize=7, va='top')

    ax.set_xlabel(f'{metric.upper()} before', fontsize=9)
    ax.set_ylabel(f'{metric.upper()} after', fontsize=9)
    ax.set_title(f'Per-epitope/variant {metric.upper()}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=5, loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, f'fig5_subset_{metric}_before_after')


# ═══════════════════════════════════════════
# Panel (k)/(l): ΔAUROC/ΔAP vs raw value ("worse gain more")
# ═══════════════════════════════════════════
for metric, panel_label in [('aucroc', 'k'), ('ap', 'l')]:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    # TCR
    tcr_raw = [b for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == metric]
    tcr_delta = [a - b for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == metric]
    # BCR by domain
    bcr_sars_raw = [b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'sars']
    bcr_sars_delta = [a - b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'sars']
    bcr_flu_raw = [b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'flu']
    bcr_flu_delta = [a - b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == metric and dom == 'flu']

    ax.scatter(tcr_raw, tcr_delta, c=TCR_COLOR, s=12, alpha=0.4, label='TCR', edgecolor='none')
    ax.scatter(bcr_sars_raw, bcr_sars_delta, c=SARS_COLOR, s=15, alpha=0.5, marker='s',
              label='BCR SARS', edgecolor='none')
    ax.scatter(bcr_flu_raw, bcr_flu_delta, c=FLU_COLOR, s=15, alpha=0.5, marker='D',
              label='BCR Flu', edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

    all_raw = tcr_raw + bcr_sars_raw + bcr_flu_raw
    all_delta = tcr_delta + bcr_sars_delta + bcr_flu_delta
    if len(all_raw) > 3:
        r, p = pearsonr(all_raw, all_delta)
        ax.text(0.05, 0.95, f'r = {r:.3f} (p = {p:.1e})', transform=ax.transAxes,
                fontsize=7, va='top')

    ax.set_xlabel(f'Raw {metric.upper()}', fontsize=9)
    ax.set_ylabel(f'\u0394{metric.upper()}', fontsize=9)
    ax.set_title(f'Worse models gain more', fontweight='bold', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, f'fig5_delta_vs_raw_{metric}')


# ═══════════════════════════════════════════
# Panel (m): Per-bin ΔAUROC vs distance ("far samples gain most")
# ═══════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.5))
if tcr_perbin_results:
    df_bin = pd.DataFrame(tcr_perbin_results,
                          columns=['model', 'test_set', 'bin', 'd_center', 'delta'])
    bin_mean = df_bin.groupby('bin').agg(d=('d_center', 'mean'), delta=('delta', 'mean')).reset_index()
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(bin_mean)))
    ax.bar(range(1, len(bin_mean) + 1), bin_mean['delta'].values, color=colors, edgecolor='white')

    r, p = pearsonr(bin_mean['d'].values, bin_mean['delta'].values)
    # Trend line
    z = np.polyfit(range(1, len(bin_mean) + 1), bin_mean['delta'].values, 1)
    x_fit = np.linspace(0.5, len(bin_mean) + 0.5, 50)
    ax.plot(x_fit, np.polyval(z, x_fit), 'k--', linewidth=1, alpha=0.7)
    ax.text(0.05, 0.95, f'r = {r:.3f} (p = {p:.3f})', transform=ax.transAxes,
            fontsize=7, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Distance bin (near → far)', fontsize=9)
    xticks = list(range(1, len(bin_mean) + 1))
    ax.set_xticks([1, len(bin_mean)])
    ax.set_xticklabels(['(near)', '(far)'], fontsize=7)
ax.set_ylabel('ΔAUROC', fontsize=9)
ax.set_title('Far samples gain most (TCR)', fontweight='bold', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'fig5_perbin_delta_vs_distance')


# ═══════════════════════════════════════════
# Panel (n): ΔAUROC vs mean distance per epitope/variant
# ═══════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
tcr_d_ep = [d for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == 'aucroc']
tcr_delta_ep = [a - b for m, ts, ep, met, b, a, n, d in tcr_epitope_results if met == 'aucroc']
bcr_d_var = [d for m, dom, v, met, b, a, n, d in bcr_variant_results if met == 'aucroc']
bcr_delta_var = [a - b for m, dom, v, met, b, a, n, d in bcr_variant_results if met == 'aucroc']

ax.scatter(tcr_d_ep, tcr_delta_ep, c=TCR_COLOR, s=12, alpha=0.4, label='TCR', edgecolor='none')
ax.scatter(bcr_d_var, bcr_delta_var, c=BCR_COLOR, s=12, alpha=0.4, label='BCR', edgecolor='none')
ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

all_d = tcr_d_ep + bcr_d_var
all_delta = tcr_delta_ep + bcr_delta_var
if len(all_d) > 3:
    r, p = pearsonr(all_d, all_delta)
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=7, va='top')

ax.set_xlabel('Mean S2DD distance', fontsize=9)
ax.set_ylabel('ΔAUROC', fontsize=9)
ax.set_title('Far epitopes gain more', fontweight='bold', fontsize=9)
ax.legend(fontsize=7, loc='upper right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'fig5_delta_vs_distance_epitope')


# ═══════════════════════════════════════════
# Panel (o): BCR SARS vs FLU domain bars
# ═══════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.5))
DOMAIN_COLORS = {'sars': '#e74c3c', 'flu': '#3498db'}
x = np.arange(len(BCR_MODELS))
w = 0.35
for i, domain in enumerate(['sars', 'flu']):
    deltas = []
    for model in BCR_MODELS:
        if domain in bcr_domain_deltas.get(model, {}):
            b, a = bcr_domain_deltas[model][domain]
            deltas.append(a - b)
        else:
            deltas.append(0)
    offset = -w / 2 + i * w
    ax.bar(x + offset, deltas, w, label=domain.upper(),
           color=DOMAIN_COLORS[domain], alpha=0.8, edgecolor='white')
    for j, d in enumerate(deltas):
        if abs(d) > 0.001:
            ax.text(x[j] + offset, d + (0.003 if d >= 0 else -0.01),
                    f'{d:+.3f}', ha='center', fontsize=5, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([BCR_MODEL_DISPLAY[BCR_STYLE_KEY[m]] for m in BCR_MODELS],
                    fontsize=7, rotation=30, ha='right')
ax.set_ylabel('ΔAUROC', fontsize=8)
ax.set_title('BCR recalibration by domain', fontweight='bold', fontsize=9)
ax.legend(fontsize=6, loc='upper right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'fig5_bcr_domain_bars')


# ═══════════════════════════════════════════
# Panel (p): ΔAUROC heatmap (5 TCR models × 4 test sets)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
models_order = TCR_MODELS
ts_order = TCR_TEST_SETS
matrix = np.full((len(models_order), len(ts_order)), np.nan)
for mi, model in enumerate(models_order):
    for ti, ts in enumerate(ts_order):
        if model in tcr_heatmap and ts in tcr_heatmap[model]:
            matrix[mi, ti] = tcr_heatmap[model][ts]

vmax = max(0.05, np.nanmax(np.abs(matrix)))
im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
for mi in range(len(models_order)):
    for ti in range(len(ts_order)):
        if not np.isnan(matrix[mi, ti]):
            v = matrix[mi, ti]
            color = 'white' if abs(v) > vmax * 0.6 else 'black'
            ax.text(ti, mi, f'{v:+.3f}', ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

ax.set_xticks(range(len(ts_order)))
ax.set_xticklabels([TCR_TS_DISP[ts] for ts in ts_order], fontsize=7)
ax.set_yticks(range(len(models_order)))
ax.set_yticklabels([MODEL_DISPLAY[m] for m in models_order], fontsize=7)
ax.set_title('ΔAUROC (TCR CT recalibration)', fontweight='bold', fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.8, label='ΔAUROC')
save(fig, 'fig5_heatmap_tcr_ct')


# ═══════════════════════════════════════════
# BCR CT heatmap: 5 models × 2 domains
# ═══════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
bcr_models_order = BCR_MODELS
bcr_domains = ['sars', 'flu']
bcr_dom_disp = {'sars': 'SARS', 'flu': 'Flu'}
bcr_matrix = np.full((len(bcr_models_order), len(bcr_domains)), np.nan)
for mi, model in enumerate(bcr_models_order):
    for di, dom in enumerate(bcr_domains):
        if model in bcr_domain_deltas and dom in bcr_domain_deltas[model]:
            b, a = bcr_domain_deltas[model][dom]
            bcr_matrix[mi, di] = a - b

vmax_b = max(0.05, np.nanmax(np.abs(bcr_matrix)))
im_b = ax.imshow(bcr_matrix, cmap='RdBu_r', vmin=-vmax_b, vmax=vmax_b, aspect='auto')
for mi in range(len(bcr_models_order)):
    for di in range(len(bcr_domains)):
        if not np.isnan(bcr_matrix[mi, di]):
            v = bcr_matrix[mi, di]
            color = 'white' if abs(v) > vmax_b * 0.6 else 'black'
            ax.text(di, mi, f'{v:+.3f}', ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

ax.set_xticks(range(len(bcr_domains)))
ax.set_xticklabels([bcr_dom_disp[d] for d in bcr_domains], fontsize=7)
ax.set_yticks(range(len(bcr_models_order)))
ax.set_yticklabels([BCR_MODEL_DISPLAY[BCR_STYLE_KEY[m]] for m in bcr_models_order], fontsize=7)
ax.set_title('ΔAUROC (BCR CT recalibration)', fontweight='bold', fontsize=9)
plt.colorbar(im_b, ax=ax, shrink=0.8, label='ΔAUROC')
save(fig, 'fig5_heatmap_bcr_ct')


# ═══════════════════════════════════════════
# ECE comparison: model-native vs S2DD-derived
# ═══════════════════════════════════════════
# Load from pre-computed CSV if available, otherwise skip
ece_path = os.path.join(RESULTS, 'ppv_npv_confidence', 'ppv_npv_calibration.csv')
if os.path.exists(ece_path):
    ece_df = pd.read_csv(ece_path)
    # Support both column name conventions
    native_col = 'ece_native' if 'ece_native' in ece_df.columns else 'ece_model_prob'
    display_col = 'model_display' if 'model_display' in ece_df.columns else 'model'
    if native_col in ece_df.columns and 'ece_s2dd' in ece_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.8))
        models_ece = ece_df[display_col].values
        ece_native = ece_df[native_col].values
        ece_s2dd = ece_df['ece_s2dd'].values
        y_pos = np.arange(len(models_ece))[::-1]

        ax.scatter(ece_native, y_pos, color='#e74c3c', s=40, zorder=5, label='Model-native')
        ax.scatter(ece_s2dd, y_pos, color='#2ecc71', s=40, zorder=5, marker='D', label='S2DD-derived')
        for i in range(len(models_ece)):
            ax.plot([ece_native[i], ece_s2dd[i]], [y_pos[i], y_pos[i]],
                    color='gray', linewidth=1, alpha=0.5)
            pct = (ece_native[i] - ece_s2dd[i]) / ece_native[i] * 100 if ece_native[i] > 0 else 0
            ax.text(max(ece_native[i], ece_s2dd[i]) + 0.005, y_pos[i],
                    f'-{pct:.0f}%', fontsize=6, color='#e74c3c', va='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models_ece, fontsize=7)
        ax.set_xlabel('Expected Calibration Error (ECE)', fontsize=9)
        ax.set_title('Calibration improvement', fontweight='bold', fontsize=9)
        ax.legend(fontsize=6, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        save(fig, 'fig5_ece_comparison')
        print("  ECE panel generated from CSV")
    else:
        print("  ECE CSV missing required columns, skipping")
else:
    print(f"  ECE CSV not found at {ece_path}, skipping")


print("\n=== All Fig 5 new panels generated ===")
