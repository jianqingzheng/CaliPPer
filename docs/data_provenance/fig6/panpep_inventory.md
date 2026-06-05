# PanPep Retrospective Study — Data Inventory

**Date:** 2026-04-19
**Source:** Gao et al., Nature Machine Intelligence 2023 (DOI: 10.1038/s42256-023-00619-3)
**Repo:** https://github.com/bm2-lab/PanPep (cloned to `Model/PanPep/`)
**Zenodo:** DOI: 10.5281/zenodo.7544387

---

## Model

| File | Path | Description |
|------|------|-------------|
| model.pt | `Model/PanPep/Requirements/model.pt` | Pre-trained meta-learner CNN (1,450 params) |
| Content_memory.pkl | `Model/PanPep/Requirements/Content_memory.pkl` | Episodic memory from 208 meta-learning tasks |
| Query.pkl | `Model/PanPep/Requirements/Query.pkl` | Memory read-head projection weights |

**Architecture:** Meta-learning with neural Turing memory. Input: peptide + CDR3β → binding score.
**Training:** meta_dataset.csv (208 peptide tasks, 31K positive pairs).
**Inference modes:** zero-shot, few-shot, majority-shot.

---

## Data Files (in `Model/PanPep/Data/`)

| File | Rows | Peptides | Labels | Purpose |
|------|------|----------|--------|---------|
| meta_dataset.csv | 31,223 | 208 | All positive | Meta-learning training (the actual training set) |
| base_dataset.csv | 32,080 | 699 | All positive | Base model initialization (superset of meta) |
| zero_dataset.csv | 857 | 491 | All positive | Zero-shot test (unseen peptides) |
| majority_training_dataset.csv | 23,232 | 25 | 50/50 balanced | Majority-shot fine-tuning |
| majority_testing_dataset.csv | 5,230 | 25 | 50/50 balanced | Majority-shot evaluation |
| Clone_expansion_result.xlsx | 406K+225K | 45 viral | Binding labels | Clone expansion validation (2 donors) |

---

## Training Data Audit (2026-04-19)

| Check | Result |
|-------|--------|
| Zero-shot peptides unseen in meta training | ✅ 0/491 overlap with 208 meta peptides |
| Zero-shot peptides vs base_dataset | ⚠ 491 zero-shot peptides appear in base (699 total) |
| Majority train/test peptide overlap | ⚠ SAME 25 peptides in both — TCR-level split only |
| meta ⊂ base | ✅ All 208 meta peptides contained in base (expected) |
| Model corresponds to training data | ✅ model.pt trained on meta_dataset.csv |

---

## External Clinical Data

### deepAntigen Supp Fig 18: Neoantigen-TCR test set
- **Source:** `Model/deepAntigen/.../source_data.xlsx` sheet "Supplementary Fig.18"
- **Rows:** 384 peptide-TCR pairs (ALL positive, label=1)
- **Peptides:** 104 unique neoantigens
- **TCRs:** 357 unique CDR3β
- **PanPep scores:** Already computed (mean=0.409, range 0.07–0.86)
- **Training overlap:** 19/104 peptides (18.3%) overlap PanPep training → split into seen/unseen
- **Other models:** deepAntigen, ImRex, TEIM, ERGO2 predictions also available

### Lowery Science 2022 (Table S7)
- **Path:** `Data/retrospective_panpep/lowery_science2022/supplementary/tables/`
- **Content:** 65 archival + 37 prospective neoantigen-reactive TCR clonotypes (CDR3A, CDR3B)
- **Issue:** Has gene+mutation identifiers but NO peptide amino acid sequences
- **Status:** ❌ Cannot use without neoantigen prediction pipeline

### Zheng Cancer Cell 2022 (Table S4)
- **Path:** `Data/retrospective_panpep/zheng_cancercell2022/supplementary/mmc4.xlsx`
- **Content:** 2,850 T cells from 5 GI patients with reactivity labels (111 reactive, 451 non-reactive)
- **Issue:** Has CDR3α/β + reactivity but NO peptide amino acid sequences
- **Status:** ❌ Cannot use without neoantigen prediction pipeline

### DapPep "TCR Sort" benchmark (AUROC=0.684)
- **Source:** DapPep paper (arXiv 2411.17798) evaluated PanPep on GI cancer dataset
- **Issue:** DapPep code/data not publicly available; they reconstructed peptides from mutations
- **Status:** ❌ Cannot reproduce without DapPep's processed data

---

## S2DD v2.6 Retrospective Plan

### Test Scenario 1: Majority-shot evaluation
- **Test:** majority_testing_dataset.csv (5,230 pairs, 25 peptides, balanced labels)
- **Reference:** majority_training_dataset.csv (23,232 pairs, same 25 peptides)
- **S2DD distance:** Levenshtein for CDR3β, BLOSUM for peptide (per §Req 6)
- **Chains:** 2-chain (peptide + CDR3β) — PanPep has no CDR3α
- **Metrics:** AUROC, AP, F1 per peptide group
- **Analysis:** S2DD degradation, LOO performance prediction across 25 peptides, Bayesian recalibration

### Test Scenario 2: Zero-shot with generated negatives
- **Test:** zero_dataset.csv (857 positive pairs, 491 unseen peptides)
- **Reference:** meta_dataset.csv (31,223 pairs, 208 peptides)
- **Negative generation:** Random CDR3-peptide shuffling (standard TCR benchmark protocol)
- **Analysis:** S2DD degradation on truly unseen peptides

### Test Scenario 3: Cross-model neoantigen comparison
- **Test:** deepAntigen Supp Fig 18 (384 pairs, 104 peptides)
- **Reference:** meta_dataset.csv (31,223 pairs, 208 peptides)
- **Split:** 85 unseen vs 19 seen peptides
- **Analysis:** Compare PanPep vs deepAntigen vs ERGO2 vs TEIM vs ImRex
- **Note:** All positive labels — use prediction score ranking, not AUROC

### v2.6 Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chains | peptide, CDR3β (binding_TCR) | PanPep uses 2-chain only |
| k, b, K | 0.1, 0.1, 50 | TCR defaults |
| Weight formula | sigma_C | Standard |
| Distance | BLOSUM for peptide, Levenshtein for CDR3β | Per §Req 6 |
| Recalibration θ | 0.5 | v2.6 default |
| Recalibration λ | 0.0 | Standard for ≥16 cal bins |
