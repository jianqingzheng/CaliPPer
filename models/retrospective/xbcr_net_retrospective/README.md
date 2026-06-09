# XBCR-net (Retrospective Validation)

Uses the same architecture and code as the primary XBCR-net benchmark
(`../../xbcr_net/`). The retrospective protocol differs only in the
calibration/test split:

- **Primary**: antibody-stratified 5-fold CV
- **Retrospective**: Panel 1 wild-type binders (cal, n=1,293) → Panel 2
  Omicron-era candidates (test, n=21). See manuscript Methods + Fig 6k,l.

**Reproduction (no CaliPPer Zenodo):**
- Tier 1 (cached predictions): predictions for both protocols are committed at
  `reproduce/data/cached_predictions/xbcr/` (Fig 6 retrospective) and
  `reproduce/data/input/results/xbcr/` (Fig 3-5 primary).
- Tier 2 (retrain): use BCR training data committed at
  `reproduce/data/input/Data/bcr_seq/` (~125 MB) with the authors' published
  model code from `jianqingzheng/XBCR-net`. `reproduce/scripts/data_prep/
  prep_fig6_xbcr.py` auto-fetches the authors' GitHub at run time.
