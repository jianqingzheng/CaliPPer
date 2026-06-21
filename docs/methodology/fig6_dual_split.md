# Fig 6 — Dual-Split Design (Panel C/D vs Panel E test sets)

**Scope:** This note explains why three of the five retrospective studies in Fig 6 (deepAntigen, AntibioticsAI, BigMHC) use **different cal/test pairs** in Panel C/D (performance prediction) versus Panel E+ (recalibration). The other two studies (XBCR-net, PanPep) use the same cal/test pair in both panels.

**TL;DR:** Panel C/D evaluates `predict_metric`, which requires an unbiased test set; Panel E evaluates Δ-metrics from recalibration, which are well-defined on selection-biased cohorts because the bias cancels in the before/after comparison.

---

## Per-study cal/test pairs

| Study | Panel C/D (prediction) | Panel E+ (recalibration) |
|---|---|---|
| **deepAntigen** | zero_shot cal (1{,}714) → ImmuneCODE (50K, independent) | zero_shot cal (1{,}714) → neoantigen (100, pre-selected by model) |
| **AntibioticsAI** | main_test (283) cal → beta-lactam (505, cross-dataset) | main_test halfsplit cal=odd / test=even (~141 each) |
| **BigMHC** | HLA halfsplit within MANAFEST (~400 cal / ~434 test) | im_val (688) → full MANAFEST (834, independent cross-dataset) |
| XBCR-net | Panel 1 (1{,}293 WT) → Panel 2 (21 Omicron) — same in both panels | Panel 1 → Panel 2 |
| PanPep | zeroshot peptide halfsplit (832 → 882) — same in both panels | zeroshot peptide halfsplit |

---

## Rationale

The published experimental cohorts — deepAntigen's 100 neoantigens, AntibioticsAI's 283-compound main_test, BigMHC's MANAFEST cohort — were **pre-selected by each model** as its top-ranked candidates for laboratory confirmation. They are **not independent test sets**; they are model-enriched (selection-biased) subsets of much larger candidate pools.

* **Performance prediction (Panel C/D) requires an unbiased test set.** The actual aggregate metric on a selection-biased cohort is itself distorted by pre-filtering (all candidates cluster at the high-score tail of the model output distribution, compressing AUROC toward chance). "Predicting" that distorted metric is not a meaningful test of a label-free predictor's generalisation claim. Panel C/D therefore evaluates prediction on **independent splits**:
    * within-domain halfsplits not produced by the model's filtering (BigMHC HLA halfsplit);
    * cross-dataset cohorts whose composition was not shaped by the published model (AntibioticsAI beta-lactam; deepAntigen ImmuneCODE);
    * or both-panels-same where the test cohort is itself unbiased (XBCR-net Panel 1→2; PanPep peptide halfsplit).

* **Recalibration improvement (Panel E+) is well-defined on selection-biased cohorts.** Δ-metrics (ΔAUROC, ΔAP, TDR uplift) are measured on the **same samples before and after recalibration**, so the selection-bias cancels in the comparison. The published experimental cohorts are precisely the appropriate test for "does recalibration improve the actual deployment-style readout that the original authors reported?"

---

## Manuscript reference

The dual-split rationale is documented in the manuscript main text for the most striking case (deepAntigen, Results §"Retrospective", main.tex ~L238):

> *"The 100 neoantigens were pre-selected by deepAntigen as top-ranked candidates from pools of 1{,}167–41{,}606 per patient, making them unsuitable as an independent test set for performance prediction but representing the realistic scenario for recalibration."*

The same logic applies to:
* **AntibioticsAI** — the 283-compound main_test was nominated by the model for chemical synthesis and antimicrobial assay; the independent beta-lactam set (compounds chosen by chemical chemistry, not by AntibioticsAI's prediction) is the appropriate prediction target.
* **BigMHC** — the MANAFEST cohort is clinically tested neoantigens, prioritised by computational scoring at the experimental design stage; an HLA halfsplit within MANAFEST (which removes model selection from cal-test partition) is the appropriate prediction target.

---

## Empirical verification

Applying `predict_metric` to the **Panel E** cal/test pairs (instead of Panel C/D's pairs) confirms that prediction degrades substantially on the selection-biased cohorts, validating the design choice.

| Study | Metric | Panel C/D \|err\| (independent split) | Panel E test \|err\| (selection-biased) |
|---|---|---|---|
| deepAntigen | AUROC | 0.049 | 0.030 ✓ (predictable either way) |
| deepAntigen | AP | 0.023 | **0.500 ✗ (fails on selection-biased cohort)** |
| AntibioticsAI | AUROC | 0.040 | 0.051 ✓ |
| AntibioticsAI | AP | 0.010 | 0.073 ✓ |
| BigMHC | AUROC | 0.015 | **0.285 ✗ (fails on selection-biased cohort)** |
| BigMHC | AP | 0.084 | **0.361 ✗ (fails on selection-biased cohort)** |

The same studies' recalibration succeeds on the same Panel E cohorts (e.g., BigMHC ΔAUROC = +0.033, deepAntigen ΔAP = +0.160) because Δ-metrics are bias-invariant.

---

## Where this is also documented

* **Per-study notes:** `docs/data_provenance/fig6/{deepantigen,bigmhc,antibioticsai}_inventory.md` each have a brief "Panel C/D vs Panel E" subsection that cross-references back to this doc.
* **Per-study inline comments:** `reproduce/scripts/fig6/compute_fig6_panel_c_d.py` records the per-study rationale inline at each study section (`# ── 1. deepAntigen ──`, etc.).
* **Script-level docstring:** the same script's module docstring summarises the design at the top of the file.
* **Manuscript main text:** L238 of `main.tex` (deepAntigen case, fully spelled out); same logic applies to AntibioticsAI and BigMHC implicitly.
