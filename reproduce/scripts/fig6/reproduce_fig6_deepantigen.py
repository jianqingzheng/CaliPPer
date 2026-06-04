"""Wrapper to reproduce compute_fig6_recal_data.py's deepAntigen output ONLY,
writing samples to the gitignored CACHE_DIR instead of OUTPUT_DIR/recal_data/.

This script does NOT modify any committed file. It reads
compute_fig6_recal_data.py as a string, substitutes OUT_DIR to CACHE_DIR,
then exec()s the result with __file__ set so SCRIPT_DIR resolves correctly.

Stops at the first non-deepAntigen save_study() call (raises sentinel exception)
so we don't trigger PanPep/XBCR/BigMHC/AntibioticsAI which have missing inputs.

Usage:
    cd <published_repo>/CaliPPer
    python reproduce/scripts/fig6/reproduce_fig6_deepantigen.py
"""
import os, sys, re

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import CACHE_DIR  # also adds CaliPPer/ + scripts/ to sys.path

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compute_fig6_recal_data.py')
OUT = os.path.join(CACHE_DIR, 'recal_data_reproduce')
os.makedirs(OUT, exist_ok=True)

with open(SCRIPT) as f:
    src = f.read()

# Override OUT_DIR -- match the canonical form after P4b.6 patches
new_src = re.sub(
    r"^OUT_DIR\s*=\s*os\.path\.join\(OUTPUT_DIR,\s*'recal_data'\)\s*$",
    f"OUT_DIR = {OUT!r}",
    src,
    count=1,
    flags=re.M,
)
assert new_src != src, "OUT_DIR substitution failed (expected new canonical form: OUTPUT_DIR/recal_data)"

# Insert sentinel: after the first save_study call (for deepAntigen),
# raise SystemExit to skip the other 4 studies whose inputs are missing.
new_src = new_src.replace(
    "save_study('deepAntigen', y_neo, p_neo, cs, d_neo)",
    "save_study('deepAntigen', y_neo, p_neo, cs, d_neo)\n"
    "print('=== deepAntigen done; stopping wrapper here ===')\n"
    "raise SystemExit(0)",
    1,
)

g = {'__name__': '__main__', '__file__': SCRIPT}
try:
    exec(compile(new_src, SCRIPT, 'exec'), g)
except SystemExit:
    pass

print(f'\n>>> Output written to: {OUT}')
