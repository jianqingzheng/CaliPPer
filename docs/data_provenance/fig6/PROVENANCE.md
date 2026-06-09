# Fig 6 — Scratch-Ready Data Provenance

This document records the full chain from authors' originally-published data → CaliPPer's "scratch-ready" input files → Stage 0 distance regeneration. It exists so a reviewer can answer "where did this CSV come from?" for every file Stage 0 reads.

Per-study inventories are in `reproduce/data/input/Data/retrospective_{study}/data_inventory.md`. This file is the cross-study index.

---

## Reproducing Fig 6 — two commands

After committing the 11 author prediction CSVs (~5.8 MB) into `reproduce/data/cached_predictions/`, the chain is:

```bash
bash reproduce/prepare_fig6_data.sh    # raw author data + staged predictions
bash reproduce/reproduce_fig6.sh        # Stage 0 distance regen + Panel E verify
```

End-to-end Panel E reproduces bit-exactly (max |Δ| = 9.99e-16 against canonical reference).

### What `prepare_fig6_data.sh` does

It populates `INPUT_DIR/` with two complementary categories of files, both of which `reproduce_fig6.sh` Stage 0/1 needs:

| Category | Provider | Examples |
|----------|----------|----------|
| **Author model predictions** (~5.8 MB, 11 files) | Committed at `reproduce/data/cached_predictions/` → staged into INPUT_DIR/results/ by Stage 0 of prep | `xbcr/test_predictions_original.csv`, `panpep/{majority,zeroshot}_test_predictions.csv`, `bigmhc/{bigmhc_train,bigmhc_val}_pred_im.csv`, `deepantigen/{zero_shot_with_distances,neoantigen_s2dd_confidence,immunecode_with_distances}.csv`, etc. |
| **Raw sequence / small-molecule data** (~120 MB direct from authors) | Fetched from authors' DOIs by per-study `prep_fig6_*.py` scripts | `Data/retrospective_xbcr/data_s1_training.xlsx`, deepAntigen `train.csv`, AntibioticsAI `MOESM4_ESM.xlsx`, PanPep `Data/{meta,base,zero}_dataset.csv`, BigMHC Mendeley CSVs |

`reproduce_fig6.sh` then regenerates DISTANCES (Stage 0a-0f) from the raw data and combines them with the staged author predictions (read in Stage 1) to produce Panel E.

The author predictions are committed (rather than auto-generated from the inference scripts) because re-running inference requires model weights from authors' GitHub/Zenodo repositories. The committed prediction footprint is only 5.8 MB.

### Why not wire inference into `prepare_fig6_data.sh` by default?

Authors' model predictions (`*_predictions.csv`) require running each author's model on each test set. The inference scripts ARE in CaliPPer (`reproduce_fig6_xbcr.py`, `eval_deepantigen_*.py`, `reproduce_fig6_bigmhc.py`, `eval_panpep_retrospective.py`), and the weights come from each author's own GitHub/Zenodo deposit (not from CaliPPer):

- **PanPep**: weights are 1.5 MB, bundled inside the authors' `PanPep-v1.0.0.zip` (Zenodo DOI 10.5281/zenodo.7544387) — auto-fetched by `prep_fig6_panpep.py`.
- **deepAntigen**: weights ~50 MB, distributed via authors' GitHub `JiangBioLab/deepAntigen` — auto-fetched by `prep_fig6_deepantigen.py`.
- **AntibioticsAI**: no inference needed (predictions are in the authors' Nature supplementary Excel) — already self-contained.
- **BigMHC**: weights ~5 GB BiLSTM ensemble; fetched from authors' Zenodo (DOI 10.5281/zenodo.7611232) via `prep_fig6_bigmhc.py` — large; consider committed predictions instead.
- **XBCR-net**: weights ~50 MB TF/Keras format; fetched from authors' GitHub `jianqingzheng/XBCR-net` via `prep_fig6_xbcr.py`.

`prepare_fig6_data.sh` accepts an `--include-inference` flag that runs inference for studies where weights are small enough to auto-fetch (PanPep + deepAntigen + AntibioticsAI). For BigMHC/XBCR-net, the script prints instructions to fetch their (large) weights from the authors' published deposits if the reviewer wants to re-run inference rather than use the committed predictions.

### Smoke-tested 2026-06-04

`prepare_fig6_data.sh` (without `--include-inference`):
- **AntibioticsAI**: Nature supplementary (2.6 MB) + Zenodo working_example.zip (440 KB, unzipped to train/test/hit CSVs) — verified, no manual step
- **PanPep**: Zenodo PanPep-v1.0.0.zip (810 KB, unzipped to Data + Requirements) — verified, no manual step
- **deepAntigen**: GitHub raw CSVs (`JiangBioLab/deepAntigen`) + ImmuneCODE (registration required) + Lowery 2022 supplementary (manual download)
- **XBCR-net**: GitHub example data (direct) + Mendeley supplementary (best-effort API; manual fallback documented)
- **BigMHC**: GitHub pseudoseqs (direct) + Mendeley v4 ZIP (best-effort API; manual fallback documented)

Two genuine manual steps remain (cannot be auto-scraped):
1. **Mendeley datasets** (XBCR + BigMHC): some Mendeley versions require browser login. Script tries direct API URL first; on failure prints exact source page + target path.
2. **ImmuneCODE-MIRA** (deepAntigen Panel C/D only — Panel E reproduces without it): Adaptive Biotechnologies portal requires registration.

> **No Zenodo deposit from CaliPPer** (retired 2026-06-10). All files needed for reproduction are either committed to this repo (~360 MB total: scripts + cached predictions + Tier-2 training data) or auto-fetched from authors' original DOIs by `prepare_fig6_data.sh`.

---

## Summary

| Study | Original source | Manual extraction step? | Inference script in CaliPPer? | Distance regen in `reproduce_fig6.sh`? |
|---|---|---|---|---|
| **XBCR-net** | Lou et al., *Cell Research* 2022 (DOI 10.1038/s41422-022-00727-6) — Mendeley supplementary | YES (Excel → CSV; documented below) | `reproduce_panel1_fresh_predictions.py` (XBCR-net inference) | Stage 0a (`compute_xbcr_panel1_distances.py`) |
| **deepAntigen** | Zhou et al. (deepAntigen GitHub) + ImmuneCODE portal + Lowery 2022 ELISPOT | NO (verbatim CSVs); ImmuneCODE pre-filtered for SARS | `reproduce_fig6_deepantigen.py` + `eval_deepantigen_bayesian_recalibration.py` | Stages 0c, 0d, 0f |
| **AntibioticsAI** | Wong et al., *Nature* 2024 (DOI 10.1038/s41586-023-06887-8) — supplementary Excel | NO (Excel read directly by Python) | `regen_antibioticsai_distances.py` (single script does inference + distance) | Stage 0e |
| **BigMHC** | Albert et al., *Nature Machine Intelligence* 2023 (Mendeley DOI 10.17632/dvmz6pkzvb) | YES (unzip ZIP; documented below) | `reproduce_fig6_bigmhc.py` (BigMHC inference) | Stage 0b (BLOSUM-sqrt distance) |
| **PanPep** | Gao et al., *Nature Machine Intelligence* 2023 (Zenodo DOI 10.5281/zenodo.7544387) | NO (verbatim CSVs) | `eval_panpep_retrospective.py` (PanPep inference, staged 2026-06-04) | Stage 0b (BLOSUM-sqrt distance) |

---

## Documented Manual Steps

These are the **two** human-in-the-loop steps in the Fig 6 chain that are not scripted in CaliPPer. Both are documented here so a reviewer can audit / reproduce them independently.

### 1. XBCR-net panel extraction from Mendeley Excel

**Original file:** `data_s1_training.xlsx` (Mendeley supplementary, 57 MB) and `source_data.xlsx` (Nature, 14 KB)

**Manual step:** the multi-sheet Excel is loaded into pandas via `pd.read_excel(..., sheet_name='Sheet1')` (or equivalent), columns renamed to CaliPPer convention (`heavy_chain`, `light_chain`, `variant_seq`, `rbd`), and split into Panel 1 train/test based on the author's published train/test partition (column `panel == 'panel1' & split == 'train'`). The resulting `panel1_training.csv` and `panel1_test.csv` are committed to `INPUT_DIR/Data/retrospective_xbcr/extracted_panels/`.

**One-time reviewer reproduction:**
```python
import pandas as pd
src = pd.read_excel('Data/retrospective_xbcr/data_s1_training.xlsx', sheet_name=0)
train = src[(src['panel'] == 'panel1') & (src['split'] == 'train')]
test  = src[(src['panel'] == 'panel1') & (src['split'] == 'test')]
train.to_csv('extracted_panels/panel1_training.csv', index=False)
test.to_csv('extracted_panels/panel1_test.csv', index=False)
```

**Panel 2 (Omicron therapeutic mAbs)** is assembled from CoV-AbDab literature (15 commercial/clinical mAbs) plus author's Panel 2 from `data_s1_training.xlsx`. The compiled `panel2_therapeutic_mab.csv` is committed.

### 2. BigMHC Mendeley ZIP extraction

**Original file:** `BigMHC Training and Evaluation Data.zip` (Mendeley DOI 10.17632/dvmz6pkzvb, version 4)

**Manual step:** `unzip` the ZIP into `Data/retrospective_bigmhc/mendeley_data/extracted/`. The unzipped files (`manafest.csv`, `im_train.csv`, `im_val.csv`, `im_test.csv`, `el_train.csv`, etc.) are committed to `INPUT_DIR/Data/retrospective_bigmhc/mendeley_data/extracted/`.

**One-time reviewer reproduction:**
```bash
cd Data/retrospective_bigmhc/mendeley_data/
unzip "BigMHC Training and Evaluation Data.zip" -d extracted/
```

---

## What ships pre-computed (with regeneration script available)

| File | Generation script in CaliPPer | Wired in `reproduce_fig6.sh`? |
|------|-------------------------------|-------------------------------|
| `results/xbcr_retrospective/distance_cache_panel1.npz` | `compute_xbcr_panel1_distances.py` | YES (Stage 0a) — verified bit-exact regen |
| `results/panpep_retrospective/blosum_sqrt/*.npy` | `compute_panpep_bigmhc_blosum_v2.py` | YES (Stage 0b) |
| `results/bigmhc_retrospective/blosum_sqrt/manafest_blosumsqrt_dist.npy` | `compute_panpep_bigmhc_blosum_v2.py` | YES (Stage 0b) |
| `results/deepantigen_retrospective/s2dd_degradation/zero_shot_sw_topk_distances.csv` | `regen_deepantigen_sw_distances.py` | YES (Stage 0d) |
| `results/deepantigen_retrospective/neoantigen_recalibration/neoantigen_recalibrated.csv` | `regen_deepantigen_distances.py` | YES (Stage 0c) |
| `results/antibioticsai_retrospective/reproduction/main_test_with_distances.csv` | `regen_antibioticsai_distances.py` | YES (Stage 0e) |
| `results/deepantigen_retrospective/s2dd_degradation/immunecode_sw_topk_distances.csv` | `eval_deepantigen_bayesian_recalibration.py` | YES (Stage 0f) |
| `results/deepantigen_retrospective/reproduction/zero_shot_predictions.csv` | `reproduce_fig6_deepantigen.py` (deepAntigen inference) | NO — pre-computed in Zenodo bundle; reviewer can re-run optionally |
| `results/deepantigen_retrospective/reproduction/immunecode_predictions.csv` | `eval_deepantigen_bayesian_recalibration.py` (deepAntigen inference) | NO — same |
| `results/deepantigen_retrospective/neoantigen_confidence/neoantigen_s2dd_confidence.csv` | `eval_deepantigen_neoantigen_confidence.py` (staged 2026-06-04) | NO — optional regen for advanced reviewers |
| `results/bigmhc_retrospective/bigmhc_train_pred_im.csv` | `reproduce_fig6_bigmhc.py` (BigMHC inference) | NO — pre-computed in Zenodo bundle |
| `results/bigmhc_retrospective/bigmhc_val_pred_im.csv` | `reproduce_fig6_bigmhc.py` (BigMHC inference) | NO — same |
| `results/panpep_retrospective/reproduction/{majority,zeroshot}_test_predictions.csv` | `eval_panpep_retrospective.py` (PanPep inference, staged 2026-06-04) | NO — optional regen |
| `results/panpep_retrospective/reproduction/zeroshot_neg_predictions.csv` | `eval_panpep_retrospective.py` | NO — same |
| `results/xbcr_retrospective/reproduction/test_predictions_original.csv` | `reproduce_panel1_fresh_predictions.py` | NO — pre-computed via XBCR-net `main_infer.py` |

**Why some inference outputs are not in the main Stage 0 pipeline**: running model inference (deepAntigen, BigMHC, PanPep, XBCR-net) requires the original authors' deep-learning frameworks (TF 2.4 for XBCR-net, PyTorch + meta-learning for PanPep, etc.), each with conflicting CUDA / library versions. Bundling them all into a single `reproduce_fig6.sh` would force every reviewer to install 4 separate conda environments. The compromise: **distance regeneration is in the critical path (Stage 0a–0f), inference regeneration is opt-in** via the standalone scripts listed above. A reviewer who wants TRUE end-to-end-from-author-data reproduction runs the inference scripts first; the default `reproduce_fig6.sh` path validates everything downstream of inference.

---

## Cross-reference: per-study `data_inventory.md`

- `Data/retrospective_xbcr/data_inventory.md`
- `Data/retrospective_deepantigen/data_inventory.md`
- `Data/retrospective_antibioticsai/data_inventory.md`
- `Data/retrospective_bigmhc/data_inventory.md`
- `Data/retrospective_panpep/data_inventory.md`
