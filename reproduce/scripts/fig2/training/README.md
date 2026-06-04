# Fig 2 — Training/Inference Scripts (optional Tier-2 path)

This directory contains the per-model training + cross-test inference scripts
that originally produced the cached prediction CSVs in
`reproduce/data/input/results/{model}/...`.

## When to use these scripts

**You do NOT need to run these to reproduce Fig 2 numerical results.** The
default `bash reproduce/reproduce_fig2.sh` uses the cached predictions
(Tier 1, bit-exact). These training scripts are provided for:

1. **Tier-2 verification**: reviewers who want to confirm that retraining
   the models with the same data produces predictions within a tolerable
   range that covers the cached files (per the two-tier reproducibility
   rule in `BUILD_PROGRESS.md`).

2. **Methodology transparency**: documenting the exact training pipeline
   that produced the published predictions.

3. **Adapting to new models**: if a reviewer wants to add a new TCR model
   to the comparison, these scripts serve as templates.

**Important**: Retraining is non-deterministic (random init + GPU + library
version drift). The resulting predictions WILL differ from the cached
predictions to some degree. The Tier-2 acceptance criterion is that the
spread of retrained outputs covers the cached file as a valid realization.
See `BUILD_PROGRESS.md` "REPRODUCIBILITY RULE — TWO-TIER MODEL" for the
authoritative specification.

## Per-model training/inference scripts

### TCR models (5 models × 2 splits = 10 scripts)

| Script | Model | Split | Framework | Expected runtime (GPU) |
|---|---|---|---|---|
| `eval_cv_folds_logdist.py` | NetTCR | 5-fold CV | TF/Keras (NetTCR env) | ~30 min |
| `eval_cross_test_logdist.py` | NetTCR | Cross-test | TF/Keras (NetTCR env) | ~30 min |
| `eval_atm_tcr_cv_logdist.py` | ATM-TCR | 5-fold CV | PyTorch (torch env) | ~2-3 h |
| `eval_atm_tcr_cross_test_logdist.py` | ATM-TCR | Cross-test | PyTorch (torch env) | ~30 min |
| `eval_blosum_rf_cv_logdist.py` | BLOSUM-RF | 5-fold CV | sklearn (CPU only) | ~10 min |
| `eval_blosum_rf_cross_test_logdist.py` | BLOSUM-RF | Cross-test | sklearn (CPU only) | ~5 min |
| `eval_ergo2_cv_logdist.py` | ERGO-II | 5-fold CV | PyTorch (torch env) | ~60 min |
| `eval_ergo2_cross_test_logdist.py` | ERGO-II | Cross-test | PyTorch (torch env) | ~10 min |
| `eval_tcrbert_cv_logdist.py` | TCR-BERT | 5-fold CV | PyTorch + transformers | ~20 min |
| `eval_tcrbert_cross_test_logdist.py` | TCR-BERT | Cross-test | PyTorch + transformers | ~5 min |

### BCR models (2 canonical scripts — 2-pathogen binding only)

Per `PANEL_MANIFEST.md` "BCR model": **600 AA XBCR-net (retrained 2026-04-27 for flu HA inclusion)** + memory `feedback_bcr_cv_3pathogen_vs_2pathogen.md`: **2-pathogen binding** (SARS-CoV-2 RBD + flu HA pooled). NOT neutralization, NOT SARS-only, NOT 3-pathogen.

| Script | Purpose | Framework | Used by Fig 3 panels |
|---|---|---|---|
| `eval_bcr_bind_ct_fold4cal.py` | 2-pathogen binding CT (fold4-as-cal, 5 models: xbcr/deepaai/mambaaai/mint/rleaai) | TF/Keras + PyTorch | c, g, i (BCR CT vbias/scatter/heatmap) |
| `eval_bcr_combined_ab_stratified.py` | 2-pathogen combined SARS+flu binding 5-fold CV | TF/Keras (TF 2.4.1) | f (BCR CV pooled scatter) |

**Helper library (NOT a retraining target)**:
- `eval_bcr_bind_ab_stratified.py` — SARS-only (1-pathogen) script, archived per the 2-pathogen rule. **Restored 2026-06-02 as an importable helper library**: the canonical 2-pathogen scripts (`eval_bcr_bind_ct_fold4cal.py` + `eval_bcr_combined_ab_stratified.py`) import shared helper functions (`prepare_xbcrnet_data`, `train_xbcrnet_fold`, `infer_xbcrnet_fold`, `collect_predictions`) from this file. Module docstring has a prominent ⚠ banner clarifying it is NOT a Fig 2-5 retraining target; do NOT invoke directly. Not in `retrain_fig3_inputs.sh` MODELS registry.

**Removed from staging (NOT canonical for Fig 3, not imported by canonical scripts)**:
- `eval_bcr_neu_ab_stratified.py` — neutralization (wrong target; Fig 3 is binding)

Cross-model BCR scripts (DeepAAI, MambaAAI, MINT, RLEAAI) live in the
respective model directories under `Model/{model}/` in the research repo —
they were not copied here because each model has its own conda env + custom
preprocessing requirements (each runs into 1000-2000 lines of model-specific
code that doesn't fit a unified "drop-in inference" pattern).

## ⚠ Known hardcoded path in `eval_bcr_bind_ab_stratified.py`

Line 40 of `eval_bcr_bind_ab_stratified.py` contains a hardcoded path
to the author's local Python interpreter:

```python
TF_PYTHON = '/home/jzheng/anaconda3/envs/tf/bin/python'
```

If a reviewer runs this script as-is, it will fail with `FileNotFoundError`
on this exact path. To use the script, replace `TF_PYTHON` with your own
TF 2.4.1-compatible Python interpreter path (e.g., the output of
`which python` inside your XBCR-net conda env). The other 12 training
scripts use the active Python interpreter and do not hardcode any
environment-specific paths.

## Important: scripts as-deposited, not patched

These scripts are **unmodified copies** of the originals in the research
repo. They have not been patched for CaliPPer-local paths because:

1. The user instruction (2026-05-30) was "no need to retrain the models",
   so we don't need to make them runnable in the CaliPPer/-only sandbox.
2. Path-patching for retraining would require staging the full training
   data (~600 MB to 1 GB) which is not part of the default Zenodo bundle.
3. Reviewers who want to retrain should set up the research repo (per
   `INSTALL.md` of the original `general_eval/` repo) and run these scripts
   from there.

If a future need arises to run these from within CaliPPer/ (e.g., a fully
self-contained retraining demo), apply the `_paths.py` bootstrap pattern
used by `reproduce/scripts/fig6/` and stage the training inputs into
`reproduce/data/input/`.

## Typical use (from the research repo root, not from CaliPPer/)

```bash
cd <research_repo_root>

# TCR models
conda activate NetTCR    && python eval_cv_folds_logdist.py             --K 50 --k 0.1 --b 0.1 --bin-num 8 --n-folds 5
conda activate NetTCR    && python eval_cross_test_logdist.py
conda activate ATM_TCR   && python eval_atm_tcr_cv_logdist.py           --epochs 200 --n-folds 5
conda activate ATM_TCR   && python eval_atm_tcr_cross_test_logdist.py   --epochs 200
                            python eval_blosum_rf_cv_logdist.py         --n-folds 5
                            python eval_blosum_rf_cross_test_logdist.py
conda activate ERGO_II   && python eval_ergo2_cv_logdist.py             --epochs 50 --n-folds 5
conda activate ERGO_II   && python eval_ergo2_cross_test_logdist.py     --epochs 50
conda activate TCR_BERT  && python eval_tcrbert_cv_logdist.py           --n-folds 5
conda activate TCR_BERT  && python eval_tcrbert_cross_test_logdist.py

# BCR models (XBCR-net)
conda activate XBCR-net  && python eval_bcr_bind_ab_stratified.py       --no-pretrain
conda activate XBCR-net  && python eval_bcr_neu_ab_stratified.py
conda activate XBCR-net  && python eval_bcr_combined_ab_stratified.py
```

After retraining, copy the resulting prediction CSVs into:

```
reproduce/data/input/results/{nettcr,atm_tcr,blosum_rf,ergo_ii,tcrbert}/{cv_logdist,cross_test_logdist}/predictions/
reproduce/data/input/results/xbcr/{neu_ab_cv_reproduced,bind_ab_cv_v2_nopretrain,combined_bind_ab_cv}/
```

Then re-run `bash reproduce/reproduce_fig2.sh` to compute the per-model |r|
values from the freshly trained models. Per the two-tier rule, retrained
|r| values will differ from cached but should fall within a tolerance band
that covers the cached values (the cached file is one valid realization
within run-to-run variation).
