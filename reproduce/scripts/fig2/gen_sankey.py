#!/usr/bin/env python3
"""Generate Sankey panels matching the reference design (new_fig2_from_4x4 panel a).

Left/right columns are GROUPED by split membership (Train/Val/Test-only),
NOT individual sequences. Middle column shows individual datasets.

TCR: [Train|Train+Val|Val|Test-only] → [trainF0..trainF2|validF3/F4|Seen|Unseen|v3|v4|McPAS|IEDB] → [Train|Train+Val|Val|Test-only]
BCR: [SARS2|Influenza] → [trainF0..F2|validF3/F4|A1-A11|unseen|flu] → [Train|Test-only]

Counts and percentages are shown for each group.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style
apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')

# Colors
SPLIT_COLORS = {
    'Train': '#3498db', 'Train+Val': '#1abc9c', 'Train+Test': '#8e44ad',
    'Val': '#f39c12', 'Test-only': '#e74c3c',
    'SARS2': '#3498db', 'Influenza': '#2ecc71',
}
FOLD_COLORS = {0: '#3498db', 1: '#e67e22', 2: '#2ecc71', 3: '#e74c3c', 4: '#9b59b6'}
CT_COLORS = {'Seen': '#2ecc71', 'Unseen': '#e74c3c', 'v3': '#3498db', 'v4': '#f39c12',
             'McPAS': '#9b59b6', 'IEDB': '#e67e22',
             'A1-A11': '#e74c3c', 'unseen': '#9b59b6', 'flu': '#2ecc71'}


def _draw_bezier_flow(ax, x0, y0s, y0e, x1, y1s, y1e, color, alpha=0.30):
    cx = (x0 + x1) / 2.0
    verts = [(x0,y0s),(cx,y0s),(cx,y1s),(x1,y1s),(x1,y1e),(cx,y1e),(cx,y0e),(x0,y0e),(x0,y0s)]
    codes = [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
             MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts,codes), facecolor=color, edgecolor='none',
                                     alpha=alpha, linewidth=0))


def classify_split(seq, train_seqs, val_seqs):
    in_train = seq in train_seqs
    in_val = seq in val_seqs
    if in_train and in_val: return 'Train+Val'
    if in_train: return 'Train'
    if in_val: return 'Val'
    return 'Test-only'


def draw_grouped_sankey(ax, df, left_group_col, mid_col, right_group_col,
                         mid_colors, left_label, mid_label, right_label,
                         left_order=None, right_order=None,
                         left_colors=None, right_colors=None,
                         x_positions=(0.0, 0.42, 0.84),
                         mid_nw=0.045):
    """Sankey with LEFT and RIGHT as grouped split categories, MID as individual datasets.

    Shows count (n) and percentage for each group.
    mid_nw: half-width of the middle bars (wider = more room for text).
    """
    # Aggregate flows
    left_mid = df.groupby([left_group_col, mid_col]).size().reset_index(name='count')
    mid_right = df.groupby([mid_col, right_group_col]).size().reset_index(name='count')

    # Orders
    if left_order is None:
        left_order = ['Train', 'Train+Val', 'Val', 'Test-only']
    left_order = [l for l in left_order if l in df[left_group_col].unique()]

    mid_order_cv = [m for m in df[mid_col].unique() if m.startswith('train') or m.startswith('valid')]
    mid_order_ct_all = ['Seen', 'Unseen', 'v3', 'v4', 'McPAS', 'IEDB',
                        'A1-A11', 'unseen', 'flu']
    # Preserve train/valid order, then CT order
    train_folds = sorted([m for m in mid_order_cv if m.startswith('train')])
    valid_folds = sorted([m for m in mid_order_cv if m.startswith('valid')])
    mid_order_ct = [m for m in mid_order_ct_all if m in df[mid_col].unique()]
    mid_order = train_folds + valid_folds + mid_order_ct

    if right_order is None:
        right_order = ['Train', 'Train+Val', 'Val', 'Test-only']
    right_order = [r for r in right_order if r in df[right_group_col].unique()]

    left_totals = df.groupby(left_group_col).size().to_dict()
    mid_totals = df.groupby(mid_col).size().to_dict()
    right_totals = df.groupby(right_group_col).size().to_dict()
    grand_total = len(df)

    if left_colors is None:
        left_colors = SPLIT_COLORS
    if right_colors is None:
        right_colors = SPLIT_COLORS

    x_left, x_mid, x_right = x_positions
    side_nw = 0.022  # half-width of side bars
    total_h = 1.0

    def layout(order, totals, gap=0.02, min_h=0.015):
        total = sum(totals.get(n,0) for n in order)
        usable = total_h - gap * max(len(order)-1, 0)
        pos = {}; y = total_h
        for name in order:
            h = max(min_h, usable * totals.get(name,1) / total)
            pos[name] = (y-h, y, h); y -= h + gap
        return pos

    def layout_mid(order, totals):
        """Layout middle column with gap between train/valid/CT groups."""
        trains = [m for m in order if m.startswith('train')]
        valids = [m for m in order if m.startswith('valid')]
        cts = [m for m in order if not m.startswith('train') and not m.startswith('valid')]
        groups = [(trains, 'train'), (valids, 'valid'), (cts, 'ct')]
        groups = [(g, label) for g, label in groups if g]

        group_totals = [sum(totals.get(m,0) for m in g) for g, _ in groups]
        grand = sum(group_totals)
        if grand == 0: return {}

        group_gap = 0.035
        n_gaps = max(len(groups)-1, 0)
        avail = total_h - group_gap * n_gaps
        ng = 0.006  # gap within group

        pos = {}; y = total_h
        for gi, (group, label) in enumerate(groups):
            g_total = sum(totals.get(m,0) for m in group)
            g_h = avail * g_total / grand
            g_usable = g_h - ng * max(len(group)-1, 0)
            for m in group:
                h = max(0.018, g_usable * totals.get(m,1) / g_total) if g_total > 0 else 0.018
                pos[m] = (y-h, y, h); y -= h + ng
            y = y + ng - group_gap  # remove last ng, add group_gap
        return pos

    l_pos = layout(left_order, left_totals, gap=0.025, min_h=0.02)
    m_pos = layout_mid(mid_order, mid_totals)
    r_pos = layout(right_order, right_totals, gap=0.025, min_h=0.02)

    # Flows: left → mid
    lc = {n:0 for n in left_order}; mc_l = {n:0 for n in mid_order}
    for _, row in left_mid.sort_values('count', ascending=False).iterrows():
        ln, mn, cnt = row[left_group_col], row[mid_col], row['count']
        if ln not in l_pos or mn not in m_pos: continue
        lb,lt,lh = l_pos[ln]; mb,mt,mh = m_pos[mn]
        fl = lh * cnt / left_totals[ln]; fr = mh * cnt / mid_totals[mn]
        y0s=lt-lc[ln]; y0e=y0s-fl; y1s=mt-mc_l[mn]; y1e=y1s-fr
        _draw_bezier_flow(ax, x_left+side_nw, y0s, y0e, x_mid-mid_nw, y1s, y1e,
                          mid_colors.get(mn, '#999'))
        lc[ln]+=fl; mc_l[mn]+=fr

    # Flows: mid → right
    mc_r = {n:0 for n in mid_order}; rc = {n:0 for n in right_order}
    for _, row in mid_right.sort_values('count', ascending=False).iterrows():
        mn, rn, cnt = row[mid_col], row[right_group_col], row['count']
        if mn not in m_pos or rn not in r_pos: continue
        mb,mt,mh = m_pos[mn]; rb,rt,rh = r_pos[rn]
        fl = mh * cnt / mid_totals[mn]; fr = rh * cnt / right_totals[rn]
        y0s=mt-mc_r[mn]; y0e=y0s-fl; y1s=rt-rc[rn]; y1e=y1s-fr
        _draw_bezier_flow(ax, x_mid+mid_nw, y0s, y0e, x_right-side_nw, y1s, y1e,
                          mid_colors.get(mn, '#999'), alpha=0.25)
        mc_r[mn]+=fl; rc[rn]+=fr

    # Draw nodes
    for pos, x, colors_map, is_mid, hw in [(l_pos, x_left, left_colors, False, side_nw),
                                             (m_pos, x_mid, mid_colors, True, mid_nw),
                                             (r_pos, x_right, right_colors, False, side_nw)]:
        for name, (yb,yt,h) in pos.items():
            if is_mid:
                c = mid_colors.get(name, '#aaa')
            else:
                c = colors_map.get(name, '#888')
            ax.add_patch(mpatches.FancyBboxPatch((x-hw,yb), hw*2, h,
                         boxstyle="round,pad=0.003", facecolor=c, edgecolor='white', linewidth=0.5))

    # Labels
    fs_side = 7; fs_mid = 6; fs_count = 6.5

    def fmt_count(cnt):
        if cnt >= 10000:
            return f'{cnt/1000:.1f}k'
        return f'{cnt:,}'

    # Left side labels
    for name, (yb,yt,h) in l_pos.items():
        cnt = left_totals.get(name, 0)
        pct = 100.0 * cnt / grand_total if grand_total > 0 else 0
        label_text = f'{name}\n({fmt_count(cnt)}, {pct:.1f}%)'
        ax.text(x_left-side_nw-0.005, (yb+yt)/2, label_text, ha='right', va='center',
                fontsize=fs_side, fontweight='bold', color=left_colors.get(name,'#333'),
                linespacing=1.15)

    # Middle labels: dataset name inside bar (or left of bar if too thin), count to the right
    for name, (yb,yt,h) in m_pos.items():
        cnt = mid_totals.get(name, 0)
        pct = 100.0 * cnt / grand_total if grand_total > 0 else 0
        if h >= 0.025:
            # Name inside bar
            ax.text(x_mid, (yb+yt)/2, name, ha='center', va='center',
                    fontsize=fs_mid, fontweight='bold', color='white', zorder=10)
        else:
            # Node too thin — place name to the left of bar
            ax.text(x_mid - mid_nw - 0.006, (yb+yt)/2, name, ha='right', va='center',
                    fontsize=fs_mid, fontweight='bold', color='#333', zorder=10)
        # Count/percentage to the right of bar
        ax.text(x_mid + mid_nw + 0.006, (yb+yt)/2, f'{fmt_count(cnt)} ({pct:.0f}%)',
                ha='left', va='center', fontsize=fs_count, color='#444', zorder=10,
                fontweight='medium')

    # Right side labels
    for name, (yb,yt,h) in r_pos.items():
        cnt = right_totals.get(name, 0)
        pct = 100.0 * cnt / grand_total if grand_total > 0 else 0
        label_text = f'{name}\n({fmt_count(cnt)}, {pct:.1f}%)'
        ax.text(x_right+side_nw+0.005, (yb+yt)/2, label_text, ha='left', va='center',
                fontsize=fs_side, fontweight='bold', color=right_colors.get(name,'#333'),
                linespacing=1.15)

    # Column headers
    ax.text(x_left, total_h+0.04, left_label, ha='center', fontsize=8, fontweight='bold')
    ax.text(x_mid, total_h+0.04, mid_label, ha='center', fontsize=8, fontweight='bold')
    ax.text(x_right, total_h+0.04, right_label, ha='center', fontsize=8, fontweight='bold')

    ax.set_xlim(x_left-0.17, x_right+0.19)
    ax.set_ylim(-0.06, total_h+0.07)
    ax.axis('off')


def save(fig, name):
    out_dir = os.path.join(PANEL_DIR, 'lev-logtransf')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name+'.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, name+'.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


# ═══════════════════════════════════════════
# TCR Sankey
# ═══════════════════════════════════════════
print("=== TCR Sankey ===")
train_tcr = pd.read_csv(os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'splits', 'train.csv'))
train_eps = set(train_tcr['peptide'].unique())
train_cdr3 = set(train_tcr['CDR3b'].unique())
val_eps, val_cdr3 = set(), set()
for ts in ['v3_combined', 'v4_combined']:
    fp = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'predictions', f'{ts}_predictions_with_label.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp); val_eps.update(df['peptide'].unique()); val_cdr3.update(df['CDR3b'].unique())

records = []
# F0-F2 → trainF0, trainF1, trainF2; F3-F4 → validF3, validF4
for fold in range(5):
    fp = os.path.join(RESULTS, 'nettcr', 'cv_logdist', f'fold{fold}', 'test_predictions_with_label.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)[['peptide','CDR3b']]
        prefix = 'train' if fold <= 2 else 'valid'
        df['dataset'] = f'{prefix}F{fold}'
        records.append(df)
for ts, display in [('seen_test','Seen'),('unseen_fold34','Unseen'),('v3_combined','v3'),
                     ('v4_combined','v4'),('mcpas','McPAS'),('iedb_sars','IEDB')]:
    fp = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'predictions', f'{ts}_predictions_with_label.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)[['peptide','CDR3b']]; df['dataset'] = display; records.append(df)

tcr_df = pd.concat(records, ignore_index=True)
tcr_df['ep_split'] = tcr_df['peptide'].apply(lambda x: classify_split(x, train_eps, val_eps))
tcr_df['cdr_split'] = tcr_df['CDR3b'].apply(lambda x: classify_split(x, train_cdr3, val_cdr3))

# Mid colors for renamed folds
mid_colors = {}
for fold in range(5):
    prefix = 'train' if fold <= 2 else 'valid'
    mid_colors[f'{prefix}F{fold}'] = FOLD_COLORS[fold]
mid_colors.update(CT_COLORS)

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
draw_grouped_sankey(ax, tcr_df, 'ep_split', 'dataset', 'cdr_split',
                     mid_colors, left_label='Epitope', mid_label='Dataset', right_label='CDR3\u03b2',
                     left_order=['Train', 'Train+Val', 'Val', 'Test-only'],
                     right_order=['Train', 'Train+Val', 'Val', 'Test-only'],
                     x_positions=(0.0, 0.44, 0.88))
ax.set_title('TCR data flow',
             fontweight='bold', fontsize=10, pad=10)
save(fig, 'fig2_sankey_tcr')

# ═══════════════════════════════════════════
# BCR Sankey — antigen split into SARS2 / Influenza subgroups
#              antibody split into Train / Train+Val / Train+Test / Test-only
#              F0-F3 = trainF0..trainF3, F4 = validF4
# ═══════════════════════════════════════════
print("\n=== BCR Sankey ===")
bcr_records = []

# Collect variant-to-pathogen mapping from training source column
variant_pathogen = {}  # variant_seq -> 'SARS2' or 'Influenza'

# Collect Heavy chain sets for split classification
# "Train" = Heavy seen in any fold's train.csv
# "Val" = Heavy seen in fold4's test.csv (calibration set)
# "Test" = Heavy seen in CT test sets (A1-A11, unseen, flu)
bcr_all_train_heavy = set()
bcr_fold4_test_heavy = set()
bcr_ct_test_heavy = set()

for fold in range(5):
    train_fp = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'train.csv')
    test_fp = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
    if os.path.exists(train_fp):
        tr = pd.read_csv(train_fp)
        bcr_all_train_heavy.update(tr['Heavy'].unique())
        for _, row in tr[['variant_seq', 'source']].drop_duplicates().iterrows():
            pathogen = 'Influenza' if row['source'] == 'flu' else 'SARS2'
            variant_pathogen[row['variant_seq']] = pathogen
    if os.path.exists(test_fp):
        te = pd.read_csv(test_fp)
        # F0-F3 = train, F4 = valid
        prefix = 'train' if fold <= 3 else 'valid'
        te['dataset'] = f'{prefix}F{fold}'
        bcr_records.append(te[['variant_seq', 'Heavy', 'dataset', 'source']])
        for _, row in te[['variant_seq', 'source']].drop_duplicates().iterrows():
            pathogen = 'Influenza' if row['source'] == 'flu' else 'SARS2'
            variant_pathogen[row['variant_seq']] = pathogen
        if fold == 4:
            bcr_fold4_test_heavy = set(te['Heavy'].unique())

# Cross-test sets
for ts in ['A1-A11', 'unseen', 'flu']:
    fp = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal', 'xbcr', f'{ts}_predictions.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df['dataset'] = ts
        df['source'] = 'flu' if ts == 'flu' else ts
        bcr_records.append(df[['variant_seq', 'Heavy', 'dataset', 'source']])
        pathogen = 'Influenza' if ts == 'flu' else 'SARS2'
        for v in df['variant_seq'].unique():
            variant_pathogen[v] = pathogen
        bcr_ct_test_heavy.update(df['Heavy'].unique())


def classify_bcr_antibody(h, train_set, val_set, test_set):
    """Classify antibody Heavy chain into Train/Train+Val/Train+Test/Val/Test-only."""
    in_train = h in train_set
    in_val = h in val_set
    in_test = h in test_set
    if in_train and in_val:
        return 'Train+Val'
    if in_train and in_test:
        return 'Train+Test'
    if in_train:
        return 'Train'
    if in_val:
        return 'Val'
    return 'Test-only'


if bcr_records:
    bcr_df = pd.concat(bcr_records, ignore_index=True)
    bcr_df['ag_split'] = bcr_df['variant_seq'].apply(
        lambda x: variant_pathogen.get(x, 'SARS2'))
    bcr_df['ab_split'] = bcr_df['Heavy'].apply(
        lambda x: classify_bcr_antibody(x, bcr_all_train_heavy, bcr_fold4_test_heavy, bcr_ct_test_heavy))

    bcr_mid_colors = {}
    for fold in range(5):
        prefix = 'train' if fold <= 3 else 'valid'
        bcr_mid_colors[f'{prefix}F{fold}'] = FOLD_COLORS[fold]
    bcr_mid_colors.update(CT_COLORS)

    bcr_left_colors = {
        'SARS2': '#3498db',
        'Influenza': '#2ecc71',
    }
    bcr_right_colors = {
        'Train': '#3498db',
        'Train+Val': '#1abc9c',
        'Train+Test': '#8e44ad',
        'Val': '#f39c12',
        'Test-only': '#e74c3c',
    }

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
    draw_grouped_sankey(ax, bcr_df, 'ag_split', 'dataset', 'ab_split',
                         bcr_mid_colors, left_label='Antigen', mid_label='Dataset', right_label='Antibody',
                         left_order=['SARS2', 'Influenza'],
                         right_order=['Train', 'Train+Val', 'Train+Test', 'Val', 'Test-only'],
                         left_colors=bcr_left_colors,
                         right_colors=bcr_right_colors,
                         x_positions=(0.0, 0.44, 0.88))
    ax.set_title('BCR data flow',
                 fontweight='bold', fontsize=10, pad=10)
    save(fig, 'fig2_sankey_bcr')

print("\nDone.")
