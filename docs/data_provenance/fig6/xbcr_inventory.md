> **⚠ VALUES UPDATED 2026-05-01:** Some ΔAUROC values below are superseded.
> Definitive: XBCR +0.135, deepAntigen +0.120, AntibioticsAI +0.106, PanPep +0.080, BigMHC +0.055.
> Source: `results/retrospective_recalibration_definitive.csv`

# XBCR-net Retrospective Study — Data Inventory

**Date:** 2026-04-19
**Source:** Cell Research 2022 (DOI: 10.1038/s41422-022-00727-6)
**Repo:** https://github.com/jianqingzheng/XBCR-net (cloned to `Data/retrospective_xbcr/xbcr_original_repo/`)
**Supplementary:** Downloaded from Nature static content

---

## Data Sources

| File | Origin | Size |
|------|--------|------|
| `xbcr_original_repo/data/binding/exper/example-experimental_data.xlsx` | GitHub repo | 3.5 MB |
| `xbcr_original_repo/data/binding/nonexp/example-negative_data.xlsx` | GitHub repo | 2.1 MB |
| `data_s1_training.xlsx` | Supplementary Data S1 (Mendeley) | 57 MB |
| `data_s2_scbcrseq.xlsx` | Supplementary Data S2 (Mendeley) | 9.2 MB |
| `source_data.xlsx` | Source Data (Nature) | 14 KB |

---

## Extracted Panels (in `extracted_panels/`)

### Panel 1: Original Train/Test Split
- **Files:** `panel1_training.csv`, `panel1_test.csv`, `panel1_all_experimental.csv`
- **Training:** 1943 samples, 14 variants (SARS-CoV-2 WT, SARS-CoV-1, OC43, MERS, NL63, 229E, Alpha, Beta, Delta, Gamma, SHC014, RaTG13, etc.)
- **Test:** 1483 samples, 5 variants (mostly WT=1443, Alpha=18, Beta=20, Delta=1, Gamma=1)
- **Columns:** Heavy, Light, variant_seq, variant_name, rbd, Name
- **Label:** `rbd` (1=binds RBD by ELISA, 0=doesn't bind)
- **Predictions:** 1293/1483 matched from inference (`results/xbcr_retrospective/reproduction/test_predictions_original.csv`)
- **Status:** ✅ Ready

### Panel 2: Therapeutic mAb Omicron Panel
- **File:** `panel2_therapeutic_mab.csv`
- **Samples:** 26 rows (7 antibodies × 3-4 Omicron variants)
- **Antibodies:** LY-CoV1404 (Bebtelovimab), LY-CoV016 (Etesevimab), LY-CoV555 (Bamlanivimab), REGN10933 (Casirivimab), AZD-1061 (Cilgavimab), ADG-2, S309 (Sotrovimab)
- **Variants tested:** Omicron BA.1, BA.2, BA.4
- **Columns:** variant, prediction_score, antibody_name, binds_to, not_binds_to, neutralizes, not_neutralizes, Heavy, Light, gt_binds
- **Key feature:** These are therapeutic mAbs tested against Omicron variants NOT in training → genuine zero-shot scenario
- **With sequences:** 21/26 rows have Heavy+Light sequences
- **Ground truth:** Per-variant binding + neutralization from published literature (CoV-AbDab)
- **Status:** ✅ Ready for S2DD analysis

### Panel 3: Cross-Reactive mAbs ELISA (Supp Fig S3)
- **File:** `panel3_crossreactive_elisa.csv`
- **Samples:** 16 measurements (8 XBN mAbs × 2 antigens: SARS-CoV-2, SARS-CoV-1)
- **Antibodies:** XBN-6, XBN-10, XBN-11, XBN-12, XBN-13, XBN-15, XBN-19, XBN-22
- **Data:** OD450 ELISA values, ELISA-positive threshold (OD > 3× negative control)
- **Result:** 7/8 bind SARS-CoV-2, 6/8 bind SARS-CoV-1 (XBN-10 and XBN-15 fail SARS-CoV-1)
- **Missing:** Heavy/Light sequences — XBN antibodies were selected from scBCR-seq predictions and cloned internally. Their sequences are NOT in the public repo or supplementary data. The XBN→clonotype mapping would need to be obtained from the authors.
- **Status:** ❌ Cannot proceed without sequences (deprioritize)

### Panel 4: scBCR-seq Validation (Supp Fig S5)
- **Files:** `panel4_scbcrseq_raw.csv`, `panel4_scbcrseq_paired.csv`
- **Raw data:** 14753 IMGT-annotated sequences from GEO GSE171703 (single-cell BCR-seq)
- **Paired:** 2342 Heavy+Light paired clonotypes (from 5 patient samples: 4, 8, 9, 11, 12)
- **Pairing method:** Matched by clonotype ID (e.g., `clonotype2162_consensus_1/2`); chain type determined from V gene (IGH=Heavy, IGK/IGL=Light)
- **Validation subset:** 89 randomly cloned mAbs tested by ELISA against SARS-CoV-2 RBD
- **Fig S5 results:** ACNN precision ~0.85, recall ~0.75, accuracy ~0.75
- **Remaining:** (1) Run XBCR-net inference on 2342 paired sequences, (2) Identify which 89 clonotypes were experimentally validated (mapping from Fig S5a table, not in downloadable data)
- **Status:** ⚠ Paired sequences ready; needs inference + 89-mAb identity matching

### Panel 5: Full Training Data (Data S1)
- **File:** `panel5_full_training.csv`
- **Samples:** 19501 rows
  - coronavirus: 6938 (same as Panel 1 all_experimental)
  - negative_to_RBD: 1110 (anti-CoV antibodies that don't bind RBD)
  - negative_HCV-Ab: 6376 (anti-HCV, assumed no CoV cross-reactivity)
  - negative_HIV-Ab: 5077 (anti-HIV, assumed no CoV cross-reactivity)
  - Note: Ebola (38K) and Flu (19K) sheets lack Heavy/Light columns
- **Purpose:** Complete S2DD reference distribution for distance computation
- **Status:** ✅ Ready

---

## Key Observations

1. **Training composition:** 3265 positive RBD binders + 2071 non-RBD CoV antibodies + 11453 HIV/HCV negatives = ~16789 effective training samples
2. **Omicron is zero-shot:** No Omicron variants in training → Panel 2 is a genuine out-of-distribution test
3. **Cross-reactive antibodies (Panel 3):** XBCR-net predicted them as pan-SARS binders; 6/8 confirmed by ELISA for SARS-CoV-1 cross-reactivity
4. **scBCR-seq (Panel 4):** Independent validation on patient-derived BCRs, not from CoV-AbDab

---

## Processing Status (2026-04-19)

- [x] Panel 1: Extracted + inference complete (1293/1483 matched)
- [x] Panel 2: Extracted with sequences + predictions + ground truth
- [ ] Panel 3: ❌ Blocked — XBN sequences not in public data
- [x] Panel 4: Heavy/Light paired (2342 clonotypes) — needs inference + 89-mAb matching
- [x] Panel 5: Full training data extracted (19501 samples)
- [x] S2DD v2.6 analysis on Panel 2 (Omicron zero-shot) — AUROC +0.096, AP +0.056
- [x] S2DD v2.6 analysis on Panel 1 — AUROC r=-0.21 (weak, 97% WT), performance prediction Beta err=0.060
- [x] XBCR-net inference on Panel 4 paired sequences — 2341 predictions generated (20.9% predicted binders)
- [ ] S2DD v2.6 analysis on Panel 4 — blocked on 89-mAb identity matching

---

## For S2DD v2.6 Retrospective Analysis

| Task | Panel | S2DD Reference | Test Data | Distance Type |
|------|-------|---------------|-----------|---------------|
| Performance prediction | 1 (test split) | Panel 5 (full training) | Panel 1 test | S2DD-Lev (3-chain) |
| Omicron binding prediction | 2 (therapeutic mAbs) | Panel 5 (full training) | 21 mAb-variant pairs | S2DD-BLOSUM (variant_seq) |
| Cross-reactive recalibration | 3 (XBN ELISA) | Panel 5 (full training) | 8 XBN mAbs | Pending sequence matching |
| scBCR-seq validation | 4 (89 mAbs) | Panel 5 (full training) | 89 mAbs | Pending inference |
