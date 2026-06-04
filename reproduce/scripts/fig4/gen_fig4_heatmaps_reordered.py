#!/usr/bin/env python3
"""Regenerate fig4 heatmaps with reordered columns: metric→model (not model→metric).

Reads from the cached prediction data, not from the heatmap PNGs.
TCR: BLOSUM-sqrt distances. BCR: Lev-logtransf distances.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
PANEL_DIR = os.path.join(FIG_DIR, 'fig4')
os.makedirs(PANEL_DIR, exist_ok=True)
from style_config import apply_publication_style, MODEL_DISPLAY, DPI
apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig3_fig4_tcr_cache')
BCR_CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_DISPLAY = {'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
               'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT'}
BCR_DISPLAY = {'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
               'mint': 'MINT', 'rleaai': 'RLEAAI'}
METRICS = ['aucroc', 'ap']
METRIC_DISP = {'aucroc': 'AUROC', 'ap': 'AP'}

BCR_MAPPING_PATH = os.path.join(INPUT_DIR, 'fig4_meta', 'bcr_variant_name_mapping.csv')


def build_heatmap(data_df, models, model_display, metrics, metric_disp,
                   subset_col, subset_labels, subset_sizes, subset_colors,
                   title, out_path, legend_labels=None, top_n=25,
                   colorbar='right'):
    """Build heatmap with columns ordered: metric → model.

    Uses pcolormesh for perfect alignment with bar chart (same as original).
    Only change from original: column order is metric→model instead of model→metric.
    legend_labels: list of (color, label) for bar chart legend.
    """
    data_df = data_df.copy()

    # Unique-average: per (subset, model, metric), average predicted/actual first,
    # then compute |error|. This avoids inflating MAE via Jensen's inequality
    # when the same epitope/variant appears in multiple folds.
    data_df = data_df.groupby([subset_col, 'model', 'metric']).agg(
        predicted=('predicted', 'mean'),
        actual=('actual', 'mean'),
        n=('n', 'sum'),
        prevalence=('prevalence', 'mean'),
    ).reset_index()
    data_df['abs_error'] = (data_df['predicted'] - data_df['actual']).abs()

    subset_counts = data_df.groupby(subset_col)['n'].first().sort_values(ascending=False)
    top_subsets = subset_counts.head(top_n).index.tolist()

    n_subsets = len(top_subsets)
    n_models = len(models)
    n_metrics = len(metrics)
    n_cols = n_models * n_metrics

    # Column order: metric → model
    col_labels = []
    col_keys = []
    for metric in metrics:
        for model in models:
            col_labels.append(f'{model_display[model]}\n{metric_disp[metric]}')
            col_keys.append((model, metric))

    # Build MAE matrix (subsets + Mean row) × cols
    mae_with_mean = np.full((n_subsets + 1, n_cols), np.nan)
    for ci, (model, metric) in enumerate(col_keys):
        for si, subset in enumerate(top_subsets):
            sub = data_df[(data_df[subset_col] == subset) &
                          (data_df['model'] == model) &
                          (data_df['metric'] == metric)]
            if len(sub) > 0:
                mae_with_mean[si, ci] = sub['abs_error'].mean()
        col_vals = mae_with_mean[:n_subsets, ci]
        valid = ~np.isnan(col_vals)
        if valid.any():
            mae_with_mean[n_subsets, ci] = col_vals[valid].mean()

    # Plot — use fixed per-row height so cells match between TCR and BCR
    row_h = 0.28  # fixed height per row
    heatmap_h = max(6, n_subsets * row_h)
    # width_ratios: [2, 3] gives the bar/label sidebar ~40% of panel width (was 25%)
    # — accommodates long BCR antigen names (e.g. SARS-CoV2-Omicron-BA1.1) without truncation
    _WR = [2, 3]
    if colorbar == 'bottom':
        # Use gridspec with dedicated colorbar row so heatmap area is unchanged
        import matplotlib.gridspec as mgs
        fig = plt.figure(figsize=(6.5, heatmap_h + 0.6))
        gs_outer = mgs.GridSpec(2, 1, figure=fig, height_ratios=[heatmap_h, 0.3],
                                 hspace=0.25)
        gs_inner = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0],
                                                width_ratios=_WR, wspace=0.02)
        ax_bar = fig.add_subplot(gs_inner[0])
        ax_heat = fig.add_subplot(gs_inner[1])
        # Colorbar: narrow horizontal bar, centered under the heatmap portion
        gs_cbar = mgs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1],
                                               width_ratios=[1, 2, 1])
        cbar_ax = fig.add_subplot(gs_cbar[1])
    else:
        fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(6.5, heatmap_h),
                                                gridspec_kw={'width_ratios': _WR, 'wspace': 0.02})
        cbar_ax = None

    # Bar chart (identical to original)
    bar_y = np.arange(n_subsets)
    counts = [subset_counts.get(s, 0) for s in top_subsets]
    colors = [subset_colors.get(s, '#888') for s in top_subsets]
    ax_bar.barh(bar_y, counts, color=colors, edgecolor='none', height=0.8)

    for i, s in enumerate(top_subsets):
        label = subset_labels.get(s, s)
        # Truncation lifted 18→40 chars to fit full BCR antigen names (e.g.
        # "SARS-CoV2-Omicron-BA.1.1") without losing identifiable suffixes
        disp = label if len(label) <= 40 else label[:37] + '...'
        ax_bar.text(2, i, disp, va='center', ha='right', fontsize=5, color='white',
                    fontweight='bold', path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])

    ax_bar.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.set_xlabel('Sample count', fontsize=6)
    ax_bar.set_ylim(n_subsets + 0.5, -0.5)
    ax_bar.invert_xaxis()
    ax_bar.grid(False)

    # Legend for bar colors
    if legend_labels:
        ax_bar.legend(handles=[Patch(facecolor=c, label=l) for c, l in legend_labels],
                      loc='lower left', fontsize=5, framealpha=0.9)

    # Heatmap — pcolormesh (keeps perfect row alignment with bars)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('#e0e0e0')
    vmax = 0.15
    masked = np.ma.masked_invalid(mae_with_mean)
    im = ax_heat.pcolormesh(np.arange(n_cols + 1) - 0.5,
                             np.arange(n_subsets + 2) - 0.5,
                             masked, cmap=cmap, vmin=0, vmax=vmax,
                             edgecolors='white', linewidth=0.5)

    ax_heat.set_xticks(np.arange(n_cols))
    ax_heat.set_xticklabels(col_labels, fontsize=4.5, rotation=45, ha='right')
    ax_heat.set_yticks([n_subsets])
    ax_heat.set_yticklabels(['Mean'], fontsize=6, fontweight='bold')
    ax_heat.set_title(title, fontsize=8, fontweight='bold')
    ax_heat.set_ylim(n_subsets + 0.5, -0.5)
    ax_heat.grid(False)

    # Separator before Mean row (white gap)
    ax_heat.axhline(y=n_subsets - 0.5, color='white', linewidth=3.0)

    # Metric group separator (between AUROC and AP blocks)
    ax_heat.axvline(x=n_models - 0.5, color='white', linewidth=3.0)

    if colorbar == 'right':
        cbar = plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02, extend='max')
        cbar.set_label('MAE', fontsize=6)
        cbar.ax.tick_params(labelsize=5)
    elif colorbar == 'bottom' and cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='max')
        cbar.set_label('MAE', fontsize=6)
        cbar.ax.tick_params(labelsize=5)
    # colorbar == 'none': skip

    fig.patch.set_facecolor('white')
    fig.savefig(out_path + '.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path + '.pdf', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════
# TCR epitope heatmap (BLOSUM-sqrt)
# ═══════════════════════════════════════════
print("=== TCR epitope heatmap (metric→model order) ===")

tcr_parts = []
for split in ['cv', 'ct']:
    for metric in METRICS:
        f = os.path.join(TCR_CACHE, f'tcr_fig4_blosum-sqrt_{split}_epitope_{metric}.csv')
        if os.path.exists(f):
            tcr_parts.append(pd.read_csv(f))
if tcr_parts:
    tcr_df = pd.concat(tcr_parts, ignore_index=True)
    # Get epitope seen/unseen from training
    train_path = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'splits', 'train.csv')
    train_eps = set()
    if os.path.exists(train_path):
        train_eps = set(pd.read_csv(train_path)['peptide'].unique())

    ep_labels = {s: s for s in tcr_df['subset'].unique()}
    ep_colors = {s: '#3498db' if s in train_eps else '#e67e22' for s in tcr_df['subset'].unique()}

    out = os.path.join(PANEL_DIR, 'blosum-sqrt', 'fig4_heatmap_epitope_mae')
    build_heatmap(tcr_df, TCR_MODELS, TCR_DISPLAY, METRICS, METRIC_DISP,
                   'subset', ep_labels, None, ep_colors,
                   'CaliPPer prediction MAE per epitope', out,
                   legend_labels=[('#3498db', 'Seen in training'), ('#e67e22', 'Unseen')],
                   colorbar='none')

# ═══════════════════════════════════════════
# BCR antigen heatmap (Lev-logtransf)
# ═══════════════════════════════════════════
print("\n=== BCR antigen heatmap (metric→model order, Lev) ===")

bcr_parts = []
for split in ['cv', 'ct']:
    for metric in METRICS:
        f = os.path.join(BCR_CACHE, f'bcr_fig4_fold4cal_{split}_antigen_{metric}.csv')
        if os.path.exists(f):
            df_part = pd.read_csv(f)
            # Fix CT placeholder: subset='variant' → use source as key
            if 'source' in df_part.columns and df_part['subset'].nunique() == 1:
                df_part['subset'] = df_part['source']
            bcr_parts.append(df_part)
if bcr_parts:
    bcr_df = pd.concat(bcr_parts, ignore_index=True)

    # Variant name mapping
    import hashlib
    bcr_mapping = pd.read_csv(BCR_MAPPING_PATH) if os.path.exists(BCR_MAPPING_PATH) else None
    var_labels = {}
    var_colors = {}
    if bcr_mapping is not None:
        for _, row in bcr_mapping.iterrows():
            vhash = hashlib.md5(row['variant_seq'].encode()).hexdigest()[:12]
            name = row['heatmap_label'] if pd.notna(row.get('heatmap_label')) else row.get('variant_name', vhash)
            var_labels[vhash] = name if pd.notna(name) else vhash
            var_colors[vhash] = '#3498db' if row.get('data_source') != 'flu' else '#e67e22'

    # Also handle CT source-prefixed keys (e.g., 'sars_2c5dba60a448')
    for subset_key in bcr_df['subset'].unique():
        if subset_key not in var_labels and '_' in subset_key:
            prefix, hash_part = subset_key.split('_', 1)
            var_labels[subset_key] = var_labels.get(hash_part, subset_key)
            var_colors[subset_key] = '#e67e22' if prefix == 'flu' else '#3498db'

    # Output to BOTH folders so panel_path finds it
    for dist_dir in ['blosum-sqrt/BCR_panels', 'lev-logtransf/BCR_panels']:
        out_dir = os.path.join(PANEL_DIR, dist_dir)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, 'fig4_bcr_heatmap_antigen_mae')
        build_heatmap(bcr_df, BCR_MODELS, BCR_DISPLAY, METRICS, METRIC_DISP,
                       'subset', var_labels, None, var_colors,
                       'CaliPPer prediction MAE per antigen', out,
                       legend_labels=[('#3498db', 'SARS-CoV-2'), ('#e67e22', 'Influenza')],
                       colorbar='bottom')

print("\nDone.")
