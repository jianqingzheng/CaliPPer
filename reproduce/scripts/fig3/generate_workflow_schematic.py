#!/usr/bin/env python3
"""Generate Fig 3 performance prediction workflow schematic panel.

Size: 7.0" × 2.0" — 2-column span for top-of-figure workflow diagram.
Content: 4-step workflow (New sequences → S2DD → Predict → Go/No-go)
  with colored boxes, arrows, and italic subtitles.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ → fig3/ → panels/ → designed_figures/ → Manuscript/ → INPUT_DIR
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..'))
DESIGNED_FIG_DIR = os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures')
sys.path.insert(0, DESIGNED_FIG_DIR)

from style_config import apply_publication_style
apply_publication_style()

PANEL_DIR = os.path.dirname(SCRIPT_DIR)  # panels/fig3/

fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')

# 4 workflow steps with wide spacing
steps = [
    (0.10, 0.55, 'New test\nsequences'),
    (0.36, 0.55, 'Compute\nS2DD distance'),
    (0.64, 0.55, 'Predict AUROC\n/ AP / F1'),
    (0.90, 0.55, 'Go / No-go\ndecision'),
]
colors = ['#d5e8d4', '#dae8fc', '#fff2cc', '#f8cecc']
edge_colors = ['#82b366', '#6c8ebf', '#d6b656', '#b85450']

# Draw boxes
for i, (x, y, txt) in enumerate(steps):
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[i],
                      edgecolor=edge_colors[i], linewidth=1.3))

# Arrows (explicit start/end to leave space for boxes)
arrow_pairs = [(0.16, 0.30), (0.43, 0.57), (0.71, 0.85)]
for x_start, x_end in arrow_pairs:
    ax.annotate('', xy=(x_end, 0.55), xytext=(x_start, 0.55),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', linewidth=1.8))

# Italic subtitles under each box
sub_labels = [
    'e.g., neoantigen\ncandidates',
    'vs. training data',
    'v2.6 PAPE+vbias\n(no labels needed)',
    'wet-lab validation\nor trust model',
]
for (x, _, _), sub in zip(steps, sub_labels):
    ax.text(x, 0.15, sub, ha='center', va='center',
            fontsize=6.5, style='italic', color='#555')

# Title
ax.text(0.5, 0.93, 'Performance prediction workflow',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Remove old versions + save
for f in os.listdir(PANEL_DIR):
    if ('workflow' in f.lower() or 'schematic' in f.lower()) and f.endswith(('.pdf', '.png')):
        os.remove(os.path.join(PANEL_DIR, f))

out = os.path.join(PANEL_DIR, 'fig3_prediction_workflow_schematic')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"Saved: fig3_prediction_workflow_schematic.pdf/png")
print(f"Size: 7.0\" × 2.0\" (2-col wide, 1 row tall)")
