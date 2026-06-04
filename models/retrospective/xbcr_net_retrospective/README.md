# XBCR-net (Retrospective Validation)

Uses the same architecture and code as the primary XBCR-net benchmark
(`../../xbcr_net/`). The retrospective protocol differs only in the
calibration/test split:

- **Primary**: antibody-stratified 5-fold CV
- **Retrospective**: Panel 1 wild-type binders (cal, n=1,293) → Panel 2
  Omicron-era candidates (test, n=21). See manuscript Methods + Fig 6k,l.

Model weights for both protocols are hosted on Zenodo Record 1
(`calipper-weights-v1.0`). See `reproduce/download_data.sh`.
