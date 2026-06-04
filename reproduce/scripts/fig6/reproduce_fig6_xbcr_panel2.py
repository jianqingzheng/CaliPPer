"""Regeneration helper for panel2_omicron_results_3chain_clean.csv.

This script is an OPTIONAL regenerator. The canonical file ships in
``reproduce/data/input/results/xbcr_retrospective/mab_recalibration/``
via ``download_data.sh``. This script writes a regenerated copy to
``reproduce/data/output/xbcr_retrospective/mab_recalibration/`` so users
can diff their regeneration against the staged canonical version.

Schema matches what eval_xbcr_panel2_omicron.py would produce.

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/fig6/reproduce_fig6_xbcr_panel2.py
"""
import os, sys
import numpy as np
import pandas as pd

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR  # also adds CaliPPer/ to sys.path

from calipper.combine_first_helpers import compute_chain_weights, compute_combine_first_distances

CHAIN_COLS = ['Heavy', 'Light', 'variant_seq']
k, b, K = 0.1, 0.03, 30

OMICRON_RBD = {
    'omicron': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
    'BA2': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLQSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
    'BA4': "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
}
VARIANT_MAP = {'omicron': 'omicron', 'omicron ': 'omicron', 'BA2': 'BA2', 'BA4': 'BA4'}

# Load
train = pd.read_csv(os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                                  'extracted_panels', 'panel1_training.csv'))
train['Light'] = train['Light'].fillna('')
train['variant_seq'] = train['variant_seq'].fillna('')

panel2 = pd.read_csv(os.path.join(INPUT_DIR, 'Data', 'retrospective_xbcr',
                                   'extracted_panels', 'panel2_therapeutic_mab.csv'))
panel2_valid = panel2[panel2['Heavy'].notna() & panel2['Light'].notna()].copy()
panel2_valid['variant_clean'] = panel2_valid['variant'].map(VARIANT_MAP)
panel2_valid['variant_seq'] = panel2_valid['variant_clean'].map(OMICRON_RBD)
panel2_with_seq = panel2_valid[panel2_valid['variant_seq'].notna()].copy()

# Distances (3-chain sigma_C Lev)
weights, _ = compute_chain_weights(train, CHAIN_COLS, k, b, K, formula='sigma_C')
p2_dist = compute_combine_first_distances(panel2_with_seq, train, CHAIN_COLS, weights, k, b, K)

# Add s2dd_distance column
panel2_with_seq = panel2_with_seq.copy()
panel2_with_seq['s2dd_distance'] = p2_dist

# Schema matches eval_xbcr_panel2_omicron.py:167 output
# `panel2_with_seq[['variant', 'antibody_name', 'prediction_score', 'gt_binds']]`
# Plus s2dd_distance which is what compute_fig6 reads.
out = panel2_with_seq[['variant', 'antibody_name', 'prediction_score', 'gt_binds',
                       'Heavy', 'Light', 'variant_seq', 's2dd_distance']].copy()

# Write to OUTPUT_DIR (regenerated copy; canonical staged version lives in INPUT_DIR)
out_dir = os.path.join(OUTPUT_DIR, 'xbcr_retrospective', 'mab_recalibration')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'panel2_omicron_results_3chain_clean.csv')
out.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")
print(f"Rows: {len(out)}, columns: {list(out.columns)}")
print(f"Antibodies: {sorted(set(out['antibody_name'].dropna()))}")
print(f"Distance range: [{p2_dist.min():.3f}, {p2_dist.max():.3f}]")
