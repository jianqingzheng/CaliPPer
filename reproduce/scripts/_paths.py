"""Self-contained path anchors for reproduce/ scripts.

Resolves all data/output/cache/figure paths to locations inside
``published_repo/CaliPPer/``. Also adds ``CaliPPer/`` to ``sys.path`` so
that ``from calipper.*``, ``from PAPE.*``, and ``from MCBPE.*`` imports
work regardless of how the script is invoked (direct python, reproduce.sh,
or pytest).

See BUILD_PLAN.md sections 1 and 5.2 for the self-containment rule and
the violation taxonomy this helper exists to prevent.

Usage from a script under ``reproduce/scripts/figN/``::

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _paths import PACKAGE_ROOT, INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR

After import, the canonical destinations are:

  - Read inputs       -> INPUT_DIR  (gitignored, committed to reproduce/data/input/)
  - Intermediate CSVs -> OUTPUT_DIR (gitignored, populated at runtime)
  - Scratch / cache   -> CACHE_DIR  (gitignored, populated at runtime; replaces /tmp/)
  - Reference values  -> REFERENCE_DIR (committed, the verify.sh target)
  - Figures           -> FIG_DIR   (gitignored, populated at runtime)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))               # CaliPPer/reproduce/scripts/
PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))  # CaliPPer/

REPRODUCE_DIR = os.path.join(PACKAGE_ROOT, 'reproduce')
INPUT_DIR     = os.path.join(REPRODUCE_DIR, 'data', 'input')
OUTPUT_DIR    = os.path.join(REPRODUCE_DIR, 'data', 'output')
CACHE_DIR     = os.path.join(REPRODUCE_DIR, 'data', 'cache')
REFERENCE_DIR = os.path.join(REPRODUCE_DIR, 'data', 'reference')
FIG_DIR       = os.path.join(REPRODUCE_DIR, 'figures', 'output')

# Make ``from calipper.* import ...``, ``from PAPE.* import ...``, and
# ``from MCBPE.* import ...`` work regardless of cwd or PYTHONPATH.
# Inserted at the FRONT of sys.path so editable installs (pip install -e .)
# are preferred when present.
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
