#!/usr/bin/env python3
"""Recovery attempt for pre-retrain ATM-TCR predictions (Fig 2 Tier 1 gap).

**Status: NEGATIVE FINDING — recovery is IMPOSSIBLE via this path.**

Verified 2026-05-30. Documented as evidence so future sessions don't
re-attempt this avenue.

## Context

The Fig 2 Tier 1 reproducibility gap (per BUILD_PROGRESS.md
"REPRODUCIBILITY RULE — TWO-TIER MODEL") is that the cached ATM-TCR
prediction CSVs in `INPUT_DIR/results/atm_tcr/...` were regenerated
on 2026-05-20 by a different agent. These post-retrain CSVs give
near-chance AUROC (0.50-0.60 on OOD test sets) vs the manuscript
canonical panels which show strong degradation curves implying AUROC
~0.85+ pre-retrain. This causes 5/12 ATM-TCR fig2 cells to diverge
(sign flips on v3_combined, McPAS, etc.).

Option (a) in BUILD_PROGRESS.md was: "Recover pre-retrain prediction
CSVs from any local backup". This script TESTS that option.

## What was checked

1. **CSV backups**: searched for any prediction CSVs dated before
   2026-05-20 anywhere in the research repo. NONE found.
2. **Model checkpoint**: `Model/ATM_TCR/models/atm_cross_test.ckpt`
   has modification date 2026-03-12 04:11 (pre-2026-05-20 retrain
   incident). This was the strongest candidate for recovery.
3. **Re-inference**: ran the canonical ATM-TCR `main.py --mode test`
   with the Mar-12 checkpoint on the staged test data, verified the
   resulting AUROC matches the May-20 post-retrain values exactly:
        seen_test AUROC = 0.7053 (May-20 CSV: 0.7055 — MATCH)
        mcpas     AUROC = 0.5994 (May-20 CSV: 0.5992 — MATCH)

## Conclusion

The Mar-12 ATM-TCR checkpoint **IS** the source of the current May-20
degraded predictions. There is no pre-retrain checkpoint anywhere on
disk. The "pre-retrain" predictions that produced the manuscript Fig 2
ATM-TCR panel canonical |r| values came from a DIFFERENT (older)
checkpoint that was overwritten and never backed up.

Therefore Option (a) of BUILD_PROGRESS.md's resolution options is
IMPOSSIBLE. The remaining options are:
  - (b) Re-render manuscript Fig 2 panels from current cached predictions
  - (c) Generous Tier 1 tolerance for ATM-TCR specifically
  - (d) Documented manuscript footnote acknowledging the gap

## How to re-verify (10 minutes, ~250MB peak)

This script does not run inference itself — it serves as documentation +
provides the exact commands a future session can run if they want to
re-verify the negative finding. The staged ATM-TCR setup is already
inside CaliPPer at `reproduce/data/input/Model/ATM_TCR/`:

    cd reproduce/data/input/Model/ATM_TCR/
    # Inference on seen_test (smallest):
    conda run -n torch python main.py \\
        --infile data/atm_train.csv \\
        --indepfile data/atm_seen_test.csv \\
        --mode test \\
        --model_name atm_cross_test.ckpt

    # Expected output (matches May 20 post-retrain CSV):
    #     auc  0.7053
    # If the AUROC were 0.85+, recovery would have worked. It does not.

Files staged inside CaliPPer for this attempt:
  - reproduce/data/input/Model/ATM_TCR/*.py         (ATM-TCR source)
  - reproduce/data/input/Model/ATM_TCR/models/atm_cross_test.ckpt
                                                    (Mar 12 checkpoint)
  - reproduce/data/input/Model/ATM_TCR/data/atm_train.csv
                                                    (Training data for vocab)
  - reproduce/data/input/Model/ATM_TCR/data/atm_seen_test.csv
                                                    (Test data, seen split)
  - reproduce/data/input/Model/ATM_TCR/data/atm_mcpas.csv
                                                    (Test data, McPAS)
  - reproduce/data/input/Model/ATM_TCR/data/blosum/ (Embedding files)

Patches applied to staged ATM-TCR code (so it runs without the original
ATM_TCR conda env):
  - utils.py: tensorboardX import made optional (only needed for training)
"""
from __future__ import annotations

import sys

EXPECTED_AUROC_SEEN_TEST = 0.7053
EXPECTED_AUROC_MCPAS = 0.5994

PRE_RETRAIN_AUROC_THRESHOLD = 0.80  # if observed AUROC > this, recovery worked

NEGATIVE_FINDING_MESSAGE = """
NEGATIVE FINDING — Option (a) recovery IMPOSSIBLE:

  Mar-12 checkpoint (only pre-2026-05-20 ATM-TCR ckpt available)
  produces AUROC = 0.7053 on seen_test, matching the current
  post-retrain CSV (0.7055). It is NOT a pre-retrain checkpoint
  in the sense of giving the manuscript canonical |r| values.

  The pre-retrain ATM-TCR ckpt (whatever produced the manuscript
  canonical 0.85+ AUROC) does not exist anywhere on disk.

  See module docstring for the experiment details.
  Resolution options remaining: (b), (c), or (d) — user choice.
"""


def main() -> int:
    print(__doc__)
    print(NEGATIVE_FINDING_MESSAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
