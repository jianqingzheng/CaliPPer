<div align="center">
<h1>
CaliPPer: Calibration and Prediction of Performance <br />
<small>Quantifying, predicting and improving AI model performance for binding prediction</small>
</h1>

[![DOI](https://img.shields.io/badge/DOI-pending-darkyellow)](#)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
</div>

Code for the paper **"Quantifying, predicting and improving AI model performance for binding prediction"**.

> This repo provides an implementation of distance-aware Bayesian recalibration and performance prediction for BCR–antigen, TCR–epitope, MHC–peptide, and small-molecule binding-prediction models.

---
### Contents ###
- [0. Brief Introduction](#0-brief-introduction)
- [1. Installation](#1-installation)
- [2. Reproducing Fig 6 — retrospective validation (~25 min, no training)](#2-reproducing-fig-6--retrospective-validation-25-min-no-training)
- [3. Reproducing Fig 2-5 — systematic evaluation](#3-reproducing-fig-2-5--systematic-evaluation)
  - [3.1. Tier 1: from cached predictions (~5 min, no training)](#31-tier-1-from-cached-predictions-5-min-no-training)
  - [3.2. Tier 2: from-scratch retraining (~11–15 GPU-hours)](#32-tier-2-from-scratch-retraining-1115-gpu-hours)
- [4. Quick API](#4-quick-api)
- [5. Citing this work](#5-citing-this-work)

---

## 0. Brief Introduction

CaliPPer combines a **Sample-to-Domain Distance (S2DD)** with **distance-aware Bayesian recalibration** to:

1. Quantify how a model's accuracy degrades as test data drifts from training (per-distance degradation curve).
2. Predict aggregate performance on unlabelled cohorts without test labels.
3. Reweight per-sample confidence scores to improve top-k true discovery rates.

The method is post-hoc (no retraining), modular across base distance metrics (Levenshtein, BLOSUM, ESM-2 embedding, Morgan fingerprint), and validated across 10 primary models + 5 retrospective published studies (XBCR-net, deepAntigen, AntibioticsAI, BigMHC, PanPep).

---

## 1. Installation

Clone code from GitHub:
```shell
git clone https://github.com/jianqingzheng/CaliPPer.git
cd CaliPPer/
```

Install dependencies:

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-lightblue)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-lightblue)](https://pandas.pydata.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-lightblue)](https://scipy.org)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3%2B-lightblue)](https://scikit-learn.org)
[![parasail](https://img.shields.io/badge/parasail-1.3%2B-lightblue)](https://github.com/jeffdaily/parasail-python)

```shell
make install            # creates conda env + installs calipper editable
```

> Tier 2 retraining additionally requires per-model conda envs (TensorFlow 2.4.1 for XBCR-net/NetTCR, PyTorch for ATM-TCR/ERGO-II/TCR-BERT/PanPep, sklearn for BLOSUM-RF, RDKit for AntibioticsAI). See `INSTALL.md`.

---

## 2. Reproducing Fig 6 — retrospective validation (~25 min, no training)

Fig 6 is the **★ recommended starting point**. It exercises the full S2DD + Bayesian recalibration pipeline end-to-end on 5 published binding-prediction studies (XBCR-net, deepAntigen, AntibioticsAI, BigMHC, PanPep) and reproduces bit-exact from a fresh clone in **two commands**, no GPU and no Zenodo bundle required.

```shell
bash reproduce/prepare_fig6_data.sh    # fetch raw author data from DOIs + stage 11 cached predictions (~3 min)
bash reproduce/reproduce_fig6.sh       # Stage 0 distance regen + Panel E verify (~22 min)
```

**Why bit-exact-from-scratch is possible here:** Fig 6 uses authors' published pretrained weights (we don't train any model), so the only non-determinism is the S2DD + recalibration computation — purely deterministic linear algebra → max \|Δ\| = 9.99e-16 against committed reference.

`prepare_fig6_data.sh` does two things:

1. **Stage 0**: copies 11 committed author prediction CSVs (5.8 MB total, at `reproduce/data/cached_predictions/`) into `INPUT_DIR/results/{study}_retrospective/...` (where Stage 1 of reproduce reads them).
2. **Per-study fetch**: downloads raw sequence/structure data from each author's DOI.

<div align="center">

| Study | Source | Auto-download? |
|---|---|---|
| **AntibioticsAI** | Nature 2024 supplementary + Zenodo `working_example.zip` | ✓ fully automatic |
| **PanPep** | Zenodo `PanPep-v1.0.0.zip` | ✓ fully automatic |
| **deepAntigen** | GitHub `JiangBioLab/deepAntigen` raw CSVs + Lowery 2022 + ImmuneCODE | ✓ GitHub auto; Lowery + ImmuneCODE manual |
| **XBCR-net** | GitHub `jianqingzheng/XBCR-net` + Mendeley supplementary | ✓ GitHub auto; Mendeley best-effort + manual fallback |
| **BigMHC** | GitHub `KarchinLab/bigmhc` + Mendeley v4 ZIP | ✓ GitHub auto; Mendeley best-effort + manual fallback |

</div>

> Manual fallback prints the exact source URL + target path; place the file and re-run.

**Expected Fig 6 Panel E values** (bit-exact target):

<div align="center">

| Study | n | ΔAUROC | ΔAP |
|---|---:|---:|---:|
| XBCR-net (BCR–antigen) | 21 | +0.163 | +0.112 |
| deepAntigen (TCR–epitope) | 100 | +0.131 | +0.160 |
| AntibioticsAI (small-molecule) | 142 | +0.065 | +0.036 |
| BigMHC (MHC–peptide) | 834 | +0.034 | +0.031 |
| PanPep (TCR–epitope) | 882 | +0.016 | +0.025 |

</div>

Optional flags:

```shell
bash reproduce/prepare_fig6_data.sh --study panpep        # one study only
bash reproduce/prepare_fig6_data.sh --include-inference   # also re-run author inference (PanPep + deepAntigen + AntibioticsAI auto; BigMHC + XBCR-net need --record 1 weights)
bash reproduce/reproduce_fig6.sh --skip-regen             # use cached distances (faster re-runs)
bash reproduce/reproduce_fig6.sh --no-verify              # regenerate, skip verify gate
```

See [`docs/data_provenance/fig6/PROVENANCE.md`](docs/data_provenance/fig6/PROVENANCE.md) for per-study DOIs, conversion-chain details, and manual-step recipes.

---

## 3. Reproducing Fig 2-5 — systematic evaluation

Fig 2-5 evaluate the S2DD + recalibration framework across **10 primary models** (5 TCR + 5 BCR) on cross-test and cross-validation splits. Two reproduction tiers:

### 3.1. Tier 1: from cached predictions (~5 min, no training)

Reproduces manuscript values bit-exact from the prediction CSVs committed to CaliPPer's Zenodo Record 2 (~9 GB).

```shell
bash reproduce/download_data.sh        # fetch Zenodo Record 2 cached predictions (~9 GB, one-time)
bash reproduce/reproduce_fig4.sh       # ~30s   → 12/12 panels (highest deposit coverage)
bash reproduce/reproduce_fig5.sh       # ~230s  → 12/15 panels (3 BCR panels need Tier 2)
bash reproduce/reproduce_fig2.sh       # ~4s    → 45/60 |r| cells bit-exact (5 ATM-TCR cells affected by 2026-05-20 retrain)
bash reproduce/reproduce_fig3.sh       # ~18min → 5/10 panels (5 panels need Tier 2)
```

<div align="center">

| Bash file | Runtime | Deposit-only coverage |
|---|---:|---|
| `reproduce_fig4.sh` | ~30s | **12/12 panels** (audit CSVs intact) |
| `reproduce_fig5.sh` | ~230s | **12/15 panels** (3 BCR panels need Tier 2) |
| `reproduce_fig2.sh` | ~4s | **45/60 |r| cells** (5 ATM-TCR cells affected by 2026-05-20 retrain) |
| `reproduce_fig3.sh` | ~18min | **5/10 panels** (5 panels need Tier 2) |

</div>

### 3.2. Tier 2: from-scratch retraining (~11–15 GPU-hours)

For panels that the deposit cannot cover (3 BCR panels in Fig 5; 5 panels in Fig 3; 5 ATM-TCR cells in Fig 2), retrain the underlying 10 models. **One shared wrapper covers all of Fig 2/3/4/5** — there is no per-figure training cost.

```shell
bash reproduce/retrain_fig3_inputs.sh --list                    # show 12 training targets
bash reproduce/retrain_fig3_inputs.sh --validate                # CPU smoke test (BLOSUM-RF, ~5 min)
bash reproduce/retrain_fig3_inputs.sh --model <name>            # retrain one
bash reproduce/retrain_fig3_inputs.sh --all --promote           # retrain ALL + auto-copy outputs
bash reproduce/reproduce_fig{2,3,5}.sh                          # re-run after retraining
```

<div align="center">

| Target | Model | Framework | Time (GPU) |
|---|---|---|---:|
| `nettcr_{cv,ct}` | NetTCR | TF/Keras | ~1 h |
| `atm_tcr_{cv,ct}` | ATM-TCR | PyTorch | ~3 h |
| `blosum_rf_{cv,ct}` | BLOSUM-RF | sklearn (CPU) | ~15 min |
| `ergo2_{cv,ct}` | ERGO-II | PyTorch | ~70 min |
| `tcrbert_{cv,ct}` | TCR-BERT | PyTorch + transformers | ~25 min |
| `bcr_ct_fold4cal` | XBCR-net + 4 BCR | TF 2.4.1 + PyTorch | ~2–3 h |
| `bcr_cv_combined` | XBCR-net + 4 BCR | TF 2.4.1 + PyTorch | ~4–6 h |
| **Total** | 5 TCR + 5 BCR | 5 conda envs | **~11–15 GPU-hours** |

</div>

> **Tier 2 is non-deterministic** (random init + GPU drift + library version drift). Retrained \|r\| values will differ from cached within a tolerance band that covers the cached values; the deposit is the canonical artifact.

---

## 4. Quick API

```python
from calipper.core import (
    compute_s2dd_distances,
    predict_metric,
    fit_recalibration,
    apply_recalibration,
)

# 1. distance from each test sample to training distribution
d = compute_s2dd_distances(test_df, train_df,
                           chain_cols=["peptide", "CDR3a", "CDR3b"])

# 2. predict performance on unlabelled cohort (AUROC, AP, F1, ...)
result = predict_metric(cal_data, test_p, d, metrics=["aucroc", "ap"])

# 3. Bayesian recalibration → per-sample confidence scores
ppv, npv, p_pos, p_neg, prev = fit_recalibration(cal_data)
cal_scores = apply_recalibration(test_y, test_p, d,
                                  ppv, npv, p_pos, p_neg, prev=prev)
```

---

## 5. Citing this work

Any publication using this source code or the released model weights should cite:

```bibtex
@article{zheng2026calipper,
  title={Quantifying, predicting and improving AI model performance for binding prediction},
  author={Zheng, Jianqing and ...},
  journal={},
  year={2026},
  doi={[]}
}
```

---

## Documentation

- [`docs/data_provenance/fig6/PROVENANCE.md`](docs/data_provenance/fig6/PROVENANCE.md) — Fig 6 retrospective studies provenance + 2-command flow
- [`CHANGELOG.md`](CHANGELOG.md) — version history
