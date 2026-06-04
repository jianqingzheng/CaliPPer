#!/usr/bin/env python3
"""Assemble 4×4 composite figures from individual panels (Fig 2-5).

Uses panels from blosum-sqrt/ subfolders. Creates grey placeholder for missing panels.
Output: Manuscript/designed_figures/assembled/fig{N}_assembled.png

Usage:
    python assemble_figures.py              # all figures
    python assemble_figures.py --fig 2      # single figure
"""
import os, sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANELS_DIR = os.path.join(SCRIPT_DIR, 'panels')
OUT_DIR = os.path.join(SCRIPT_DIR, 'assembled')
os.makedirs(OUT_DIR, exist_ok=True)

DIST_DEFAULT = 'blosum-sqrt'

# Per-figure distance override: fig2 uses Lev (degradation is where Lev wins)
DIST_PER_FIG = {
    2: 'lev-logtransf',
    3: 'lev-logtransf',  # BLOSUM worse for BCR CT (see feedback_blosum_vs_lev_limitation.md)
    4: 'blosum-sqrt',
    5: 'blosum-sqrt',
    6: 'blosum-sqrt',
}


def panel_path(fig_num, filename):
    """Resolve panel path from distance-specific subfolder."""
    dist = DIST_PER_FIG.get(fig_num, DIST_DEFAULT)
    candidates = [
        os.path.join(PANELS_DIR, f'fig{fig_num}', dist, filename),
        os.path.join(PANELS_DIR, f'fig{fig_num}', dist, 'TCR_panels', filename),
        os.path.join(PANELS_DIR, f'fig{fig_num}', dist, 'BCR_panels', filename),
        # Fallback: root panel dir (for non-distance-specific panels)
        os.path.join(PANELS_DIR, f'fig{fig_num}', filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def make_placeholder(label, w=400, h=300):
    """Create grey placeholder image with label text."""
    img = np.ones((h, w, 3), dtype=np.float32) * 0.92
    return img, label


def assemble_4x4(fig_num, grid, title, nrows=4, ncols=4, span_map=None):
    """Assemble nrows×ncols grid of panels into one figure.

    grid: list of (label, filename_or_None) for each slot, row-major order.
    span_map: dict of {label: span_spec} where span_spec is either:
              - int: colspan (e.g., 2 means panel spans 2 columns)
              - tuple (colspan, rowspan): e.g., (2, 2) means 2×2 span
              Spanned slots should be ('_skip', None) in the grid.
    """
    import matplotlib.gridspec as gridspec

    if span_map is None:
        span_map = {}

    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.0))

    # Build gridspec
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.02, hspace=0.02)

    skip_slots = set()
    idx = 0
    for label, filename in grid:
        if label == '_skip' or idx in skip_slots:
            idx += 1
            continue

        r, c = idx // ncols, idx % ncols
        span = span_map.get(label, 1)
        if isinstance(span, tuple):
            colspan, rowspan = span
        else:
            colspan, rowspan = span, 1

        if colspan > 1 or rowspan > 1:
            ax = fig.add_subplot(gs[r:r + rowspan, c:c + colspan])
            # Mark all spanned slots as skip
            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        continue
                    skip_slots.add((r + dr) * ncols + (c + dc))
        else:
            ax = fig.add_subplot(gs[r, c])

        path = panel_path(fig_num, filename) if filename else None
        if path:
            try:
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.text(0.02, 0.98, f'{label}', transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.8))
            except Exception as e:
                import sys as _s_lbl
                print(f"  ⚠ FALLBACK [assemble_figures label={label}]: panel-label render failed ({type(e).__name__}: {e}); using Error placeholder", file=_s_lbl.stderr, flush=True)
                ax.text(0.5, 0.5, f'{label}\nError',
                        ha='center', va='center', fontsize=9, color='red',
                        transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        else:
            desc = filename if filename else 'TBD'
            ax.text(0.02, 0.98, f'{label}', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')
            ax.text(0.5, 0.5, f'{desc}\n[missing]',
                    ha='center', va='center', fontsize=7, color='#aaaaaa',
                    transform=ax.transAxes, style='italic')
            ax.set_facecolor('#fafafa')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        idx += 1

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    out = os.path.join(OUT_DIR, f'fig{fig_num}_assembled.png')
    out_pdf = os.path.join(OUT_DIR, f'fig{fig_num}_assembled.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    fig.savefig(out_pdf, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')

    # Count missing
    n_missing = sum(1 for l, f in grid if l != '_skip' and f and not panel_path(fig_num, f))
    n_tbd = sum(1 for l, f in grid if l != '_skip' and f is None)
    n_ok = len([1 for l, f in grid if l != '_skip']) - n_missing - n_tbd
    print(f'  {n_ok} panels OK, {n_missing} missing, {n_tbd} TBD')


# ═══════════════════════════════════════════
# Figure definitions
# ═══════════════════════════════════════════

def fig2():
    """Fig 2: Universal Performance Degradation (4×4, Lev-log)

    Panel changes (2026-04-30):
    - a-b: Sankey (spans 2 cols) — data flow epitopes→datasets→CDR3β
    - c: TCR CV AP (was a) — 5 models overlaid
    - d: BCR CV AP (was c) — 5 models overlaid
    - e-f: TCR-BERT CT (was ATM-TCR) — combined scope, 6 test sets
    - g-h: MambaAAI CT (was XBCR-net) — better flu performance
    - Dropped: TCR CV AUROC (b) and BCR CV AUROC (d) — AP is more informative
    """
    grid = [
        # Row 1: TCR Sankey + BCR Sankey + TCR CV AP + BCR CV AP
        ('a', 'fig2_sankey_tcr.png'),       # TCR data flow
        ('b', 'fig2_sankey_bcr.png'),       # BCR data flow
        ('c', 'fig2_tcr_cv_ap.png'),        # TCR CV AP (5 models overlaid)
        ('d', 'fig2_bcr_cv_ap.png'),        # BCR CV AP (5 models overlaid)
        # Row 2: ATM-TCR CT + MambaAAI CT (combined scope, 6/4 test sets)
        ('e', 'fig2_tcr_ct_ap_combined_atm_tcr.png'),
        ('f', 'fig2_tcr_ct_auroc_combined_atm_tcr.png'),
        ('g', 'fig2_bcr_ct_ap_mambaaai.png'),
        ('h', 'fig2_bcr_ct_auroc_mambaaai.png'),
        # Row 3: ATM-TCR unseen + CV heatmaps
        ('i', 'fig2_tcr_ct_ap_unseen_atm_tcr.png'),       # ATM-TCR unseen AP
        ('j', 'fig2_tcr_ct_auroc_unseen_atm_tcr.png'),    # ATM-TCR unseen AUROC
        ('k', 'fig2_tcr_heatmap_ap.png'),                 # TCR CV r/rho heatmap
        ('l', 'fig2_bcr_heatmap_ap.png'),                 # BCR CV r/rho heatmap
        # Row 4: Distance comparison + training size + CT distance comparison
        ('m', 'fig2_distance_comparison_tcr_cv.png'),      # TCR CV Lev vs BLOSUM
        ('n', 'fig2_distance_comparison_bcr_cv.png'),      # BCR CV Lev vs BLOSUM
        ('o', 'fig2_cross_model_consistency_ap.png'),       # Cross-model consistency (CV, AP)
        ('p', 'fig2_cross_model_consistency_auroc.png'),   # Cross-model consistency (CV, AUROC)
    ]
    assemble_4x4(2, grid, 'Fig 2: Universal Performance Degradation')


def fig3():
    """Fig 3: Dataset-Level Performance Prediction (4×4 with spanning panels)

    Layout:
      Row 1: (a) placeholder, (b) placeholder, (c) ATM-TCR vbias, (d) BCR MINT vbias
      Row 2: (e) TCR CV scatter, (f) TCR CT scatter, (g) BCR CV scatter, (h) BCR CT scatter
      Row 3: (i) TCR CT pred_error [2×2 span], (k) BCR CT pred_error [1×2 span], (l) TCR boxplot
      Row 4: [span from i], [span from k], (p) BCR boxplot

    All panels from lev-logtransf/ (NOT blosum-sqrt).
    """
    # Contiguous a–k (concept is ONE panel that merely spans 2 cols).
    grid = [
        # Row 1: concept diagram (single panel a, spans 2 cols) + b/c vbias curves
        ('a', 'fig3_concept_ab.png'),  # single concept panel, spans 2 cols
        ('_skip', None),  # grid cell spanned by a
        ('b', 'fig3_tcr_ct_vbias_aucroc_atm_tcr.png'),
        ('c', 'fig3_bcr_ct_vbias_ap_mint.png'),
        # Row 2: pooled scatter (CV + CT for TCR and BCR)
        ('d', 'fig3_tcr_cv_scatter_pooled.png'),
        ('e', 'fig3_tcr_ct_scatter_pooled.png'),
        ('f', 'fig3_bcr_cv_scatter_pooled.png'),
        ('g', 'fig3_bcr_ct_scatter_pooled.png'),
        # Row 3: pred_error heatmaps (spanning) + method boxplots
        ('h', 'fig3_tcr_ct_pred_error_aucroc.png'),  # spans 2×2 grid cells
        ('_skip', None),  # grid cell spanned by h
        ('i', 'fig3_bcr_ct_pred_error_aucroc.png'),  # spans 1×2 grid cells
        ('j', 'fig3_method_comparison_tcr.png'),
        # Row 4: continuations of spans + BCR boxplot
        ('_skip', None),  # grid cell spanned by h
        ('_skip', None),  # grid cell spanned by h
        ('_skip', None),  # grid cell spanned by i
        ('k', 'fig3_method_comparison_bcr.png'),
    ]
    span_map = {
        'a': (2, 1),  # concept: 2 cols × 1 row (single panel)
        'h': (2, 2),  # TCR CT pred-error: 2 cols × 2 rows
        'i': (1, 2),  # BCR CT pred-error: 1 col × 2 rows
    }
    assemble_4x4(3, grid, 'Fig 3: Dataset-Level Performance Prediction',
                 span_map=span_map)


def fig4():
    """Fig 4: Subset-Level Performance Prediction

    Rows 1-2: 4 columns (a-h)
    Rows 3-4: 3 equal columns — i (TCR heatmap, 2-row), k (BCR heatmap, 2-row),
              right col: l (top) + p (bottom)

    Uses 12-column gridspec (LCM of 4 and 3) for mixed column widths.
    """
    import matplotlib.gridspec as gridspec

    # Panel definitions: (label, filename, row, col_start, col_span, row_span)
    # 8-col grid (per 2026-05-21 user direction — i/k share 3 panel-cols total, l/p=2):
    #   rows 0-1 panels = 2 cols each (4 panels × 2 = 8)
    #   rows 2-3: i (3) + k (3) + l/p (2 stacked) = 8 — no gap between k and l
    # Sequence-label widening for i, k is handled INTERNALLY in gen_fig4_heatmaps_reordered.py
    # (width_ratios + truncation limit), not by changing the panel width.
    panels = [
        # Row 1 (4 panels, each 2 cols wide)
        ('a', 'fig4_property_epitope_ap_blosum_rf_unique.png',      0, 0, 2, 1),
        ('b', 'fig4_property_epitope_aucroc_blosum_rf_unique.png',  0, 2, 2, 1),
        ('c', 'fig4_bcr_property_antigen_ap_rleaai_unique.png',     0, 4, 2, 1),
        ('d', 'fig4_bcr_property_antigen_aucroc_rleaai_unique.png', 0, 6, 2, 1),
        # Row 2 (4 panels, each 2 cols wide)
        ('e', 'fig4_tcr_epitope_error_scatter_unique.png',  1, 0, 2, 1),
        ('f', 'fig4_bcr_variant_error_scatter_unique.png',  1, 2, 2, 1),
        ('g', 'fig4_subset_mae_boxplot.png',                1, 4, 2, 1),
        ('h', 'fig4_subset_correlation_boxplot.png',        1, 6, 2, 1),
        # Rows 3-4: i (3 cols), j (3 cols), k (top) and l (bottom) each 2 cols — total 8 ✓
        # Reindexed 2026-05-21: no letter gaps (was i, k, l, p; now i, j, k, l)
        ('i', 'fig4_heatmap_epitope_mae.png',                    2, 0, 3, 2),
        ('j', 'fig4_bcr_heatmap_antigen_mae.png',                2, 3, 3, 2),
        ('k', 'fig4_boxplot_correlation_subset_type_ap.png',     2, 6, 2, 1),
        ('l', 'fig4_boxplot_correlation_subset_type_aucroc.png', 3, 6, 2, 1),
    ]

    fig = plt.figure(figsize=(14, 12))  # same physical dims as original 12-col layout
    gs = gridspec.GridSpec(4, 8, figure=fig, wspace=0.02, hspace=0.02)

    for label, filename, r, c, cs, rs in panels:
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        path = panel_path(4, filename)
        # Top-align panels in rows 2-3 so heatmaps and boxplots align at top
        if r >= 2:
            ax.set_anchor('N')
        if path:
            try:
                img = mpimg.imread(path)
                # Only use aspect='auto' for bottom spanning rows to prevent squeezing
                ax.imshow(img, aspect='auto' if r >= 2 else 'equal')
                ax.text(0.02, 0.98, f'{label}', transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.8))
            except Exception as _e_img:
                import sys as _s_img
                print(f"  ⚠ FALLBACK [assemble_figures]: panel image rendering failed for label={label} ({type(_e_img).__name__}: {_e_img}); rendering 'Error' placeholder", file=_s_img.stderr, flush=True)
                ax.text(0.5, 0.5, f'{label}\nError',
                        ha='center', va='center', fontsize=9, color='red',
                        transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        else:
            ax.text(0.02, 0.98, f'{label}', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')
            ax.text(0.5, 0.5, f'{filename}\n[missing]',
                    ha='center', va='center', fontsize=7, color='#aaaaaa',
                    transform=ax.transAxes, style='italic')
            ax.set_facecolor('#fafafa')
        ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    out = os.path.join(OUT_DIR, 'fig4_assembled.png')
    out_pdf = os.path.join(OUT_DIR, 'fig4_assembled.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    fig.savefig(out_pdf, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')

    n_ok = sum(1 for l, f, *_ in panels if panel_path(4, f))
    n_missing = sum(1 for l, f, *_ in panels if not panel_path(4, f))
    print(f'  {n_ok} panels OK, {n_missing} missing')


def fig5():
    """Fig 5: Bayesian Recalibration (4×4)

    Distance rule: use lev-logtransf for all panels (May 2026).
    Scripts have BLOSUM-sqrt TCR override built in (dumbbell, scatter k/l, perbin m).
    BCR uses Levenshtein (optimal for long conserved sequences).
    All lev-logtransf panels regenerated May 7 with trimmed CDR3 + cal_prev fix.
    blosum-sqrt panels are stale (pre-fix, May 6).
    """
    # (label, filename, dist_override)
    # dist_override: None = use DIST_PER_FIG default, or explicit folder
    # Row 1 has a special layout: concept diagram spans cols 0-1, then c and d in cols 2-3
    # Rows 2-4 are regular 4-column grids
    regular_panels = [
        # Row 2: ROC 5-model (TCR + BCR) + dataset-level boxplots
        ('d', 'fig5_roc_tcr_ct_5models.png', 'lev-logtransf', 1, 0),
        ('e', 'fig5_roc_bcr_ct_5models.png', 'lev-logtransf', 1, 1),
        ('f', 'fig5_recal_paired_boxplot_dataset_aucroc.png', 'lev-logtransf', 1, 2),
        ('g', 'fig5_recal_paired_boxplot_dataset_ap.png', 'lev-logtransf', 1, 3),
        # Row 3: Dumbbell (AUROC + AP) + dataset before/after scatter
        ('h', 'fig5_combined_recal_dumbbell.png', 'lev-logtransf', 2, 0),
        ('i', 'fig5_combined_recal_ap_dumbbell.png', 'lev-logtransf', 2, 1),
        ('j', 'fig5_dataset_aucroc_before_after_with_cv.png', 'lev-logtransf', 2, 2),
        ('k', 'fig5_dataset_ap_before_after_with_cv.png', 'lev-logtransf', 2, 3),
        # Row 4: Far-sample-gain-most + ΔAUROC vs PPV/NPV MAE
        ('l', 'fig5_perbin_delta_vs_distance.png', 'lev-logtransf', 3, 0),
        ('m', 'fig5_perbin_delta_bcr.png', 'lev-logtransf', 3, 1),
        ('n', 'fig5_lev_vs_blosum_recal.png', 'lev-logtransf', 3, 2),
        ('o', 'fig5_subset_recal_scatter.png', 'lev-logtransf', 3, 3),
    ]

    fig = plt.figure(figsize=(4 * 3.5, 4 * 3.0))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.02, hspace=0.02)

    # --- Row 1: concept (a,b spanning 2 cols) + c + d ---
    # Concept diagram spans columns 0-1
    ax_ab = fig.add_subplot(gs[0, 0:2])
    concept_path = os.path.join(PANELS_DIR, 'fig5', 'lev-logtransf', 'fig5_concept_recalibration.png')
    if not os.path.exists(concept_path):
        concept_path = os.path.join(PANELS_DIR, 'fig5', 'concept', 'recalibration_concept.png')
    if os.path.exists(concept_path):
        img = mpimg.imread(concept_path)
        ax_ab.imshow(img)
        ax_ab.text(0.01, 0.98, 'a', transform=ax_ab.transAxes,
                   fontsize=14, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='none', alpha=0.8))
    else:
        ax_ab.text(0.5, 0.5, 'concept\n[missing]', ha='center', va='center',
                   fontsize=9, color='#aaa', transform=ax_ab.transAxes)
        ax_ab.set_facecolor('#fafafa')
    ax_ab.set_xticks([]); ax_ab.set_yticks([]); ax_ab.axis('off')

    # Panels b, c in columns 2-3
    for label, filename, dist, col in [
        ('b', 'fig5_scatter_marginals_tcr_ct_nettcr.png', 'lev-logtransf', 2),
        ('c', 'fig5_scatter_marginals_bcr_ct_deepaai.png', 'lev-logtransf', 3),
    ]:
        ax = fig.add_subplot(gs[0, col])
        candidates = [
            os.path.join(PANELS_DIR, 'fig5', dist, filename),
            os.path.join(PANELS_DIR, 'fig5', filename),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path:
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.text(0.02, 0.98, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')

    # --- Rows 2-4: regular panels ---
    for label, filename, dist, row, col in regular_panels:
        ax = fig.add_subplot(gs[row, col])
        candidates = [
            os.path.join(PANELS_DIR, 'fig5', dist, filename),
            os.path.join(PANELS_DIR, 'fig5', dist, 'TCR_panels', filename),
            os.path.join(PANELS_DIR, 'fig5', dist, 'BCR_panels', filename),
            os.path.join(PANELS_DIR, 'fig5', filename),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path:
            try:
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.text(0.02, 0.98, label, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.8))
            except Exception as _e_img:
                import sys as _s_img
                print(f"  ⚠ FALLBACK [assemble_figures]: panel image rendering failed for label={label} ({type(_e_img).__name__}: {_e_img}); rendering 'Error' placeholder", file=_s_img.stderr, flush=True)
                ax.text(0.5, 0.5, f'{label}\nError',
                        ha='center', va='center', fontsize=9, color='red',
                        transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        else:
            ax.text(0.02, 0.98, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')
            ax.text(0.5, 0.5, f'{filename}\n[missing]',
                    ha='center', va='center', fontsize=7, color='#aaaaaa',
                    transform=ax.transAxes, style='italic')
            ax.set_facecolor('#fafafa')
        ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    out = os.path.join(OUT_DIR, 'fig5_assembled.png')
    out_pdf = os.path.join(OUT_DIR, 'fig5_assembled.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    fig.savefig(out_pdf, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out}')

    n_found = sum(1 for _, f, d, *_ in regular_panels
                  if any(os.path.exists(os.path.join(PANELS_DIR, 'fig5', d, sub, f))
                         for sub in ['', 'TCR_panels', 'BCR_panels'])
                  or os.path.exists(os.path.join(PANELS_DIR, 'fig5', f)))
    n_found += 3  # concept (a,b span) + c + d
    print(f'  {n_found} panels OK, {15 - n_found} missing')


def fig6():
    """Fig 6: Independent Retrospective Validation (4×4) — Redesigned 2026-05-02

    Row 1: Conceptual + prediction + baselines
    Row 2: Recalibration overview + PanPep
    Row 3: deepAntigen + XBCR-net
    Row 4: BigMHC + AntibioticsAI
    """
    grid = [
        # Row 1: Overview (concept spans a+b)
        ('a', 'fig6_concept_ab.png'),  # spans 2 cols (a+b)
        ('_skip', None),  # b — spanned by a
        ('c', 'fig6_c_prediction_scatter.png'),
        ('d', 'fig6_d_baseline_boxplot.png'),
        # Row 2: Recalibration overview + deepAntigen
        ('e', 'fig6_e_recal_dumbbell_auroc_ap.png'),
        ('f', 'fig6_f_roc_deepantigen.png'),
        ('g', 'fig6_g_neoantigen_reranking.png'),
        ('h', 'fig6_h_tdr_deepantigen.png'),
        # Row 3: PanPep + XBCR-net
        ('i', 'fig6_i_roc_panpep.png'),
        ('j', 'fig6_j_tdr_panpep.png'),
        ('k', 'fig6_k_roc_xbcrnet.png'),
        ('l', 'fig6_l_xbcr_omicron_reranking.png'),
        # Row 4: BigMHC + AntibioticsAI
        ('m', 'fig6_m_roc_bigmhc.png'),
        ('n', 'fig6_n_tdr_bigmhc.png'),
        ('o', 'fig6_o_roc_antibioticsai.png'),
        ('p', 'fig6_p_tdr_antibioticsai.png'),
    ]
    span_map = {
        'a': (2, 1),  # 2 cols × 1 row (concept spans a+b)
    }
    assemble_4x4(6, grid, 'Fig 6: Independent Retrospective Validation',
                 span_map=span_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=int, help='Generate specific figure (2-6)')
    args = parser.parse_args()

    figs = {2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

    if args.fig:
        if args.fig in figs:
            figs[args.fig]()
        else:
            print(f'Unknown figure: {args.fig}')
    else:
        for n, fn in sorted(figs.items()):
            fn()

    print('\nDone.')
