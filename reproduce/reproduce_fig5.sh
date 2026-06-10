#!/usr/bin/env bash
# reproduce_fig5.sh — Fig 5 reproduction (Bayesian recalibration across 10 models).
#
# Fig 5 reports S2DD recalibration performance across 5 TCR + 5 BCR models
# in 15 panels (a–o, panel a spans 2 cols, reindexed 2026-05-26 per
# PANEL_MANIFEST.md).
#
# ┌───────────────────────────────────────────────────────────────────────┐
# │ THIS BASH FILE = FROM-SCRATCH REPRODUCTION (no model retraining).     │
# │                                                                       │
# │ Reviewers do NOT need to retrain any model to reproduce Fig 5 TCR     │
# │ panels:                                                               │
# │     # (data committed to reproduce/data/input/; no external download needed)        │
# │     bash reproduce/reproduce_fig5.sh  # this script (~10 min)         │
# │                                                                       │
# │ TCR panels (b, c, d, e, f-TCR, g-TCR, h-TCR, i-TCR) reproduce         │
# │ from deposited TCR cross-test predictions. BCR portions need the      │
# │ Tier-2 retraining wrapper (BCR cal_predictions.csv files were lost    │
# │ in the 2026-05-20 incident; see retrain_fig3_inputs.sh which Fig 5   │
# │ reuses).                                                              │
# │                                                                       │
# │ OPTIONAL — Tier 2 (BCR retraining via Fig 3's wrapper):               │
# │     bash reproduce/retrain_fig3_inputs.sh --model bcr_ct_fold4cal     │
# │                                                                       │
# │ See BUILD_PROGRESS.md "REPRODUCIBILITY RULE — TWO-TIER MODEL" for     │
# │ the authoritative specification.                                      │
# └───────────────────────────────────────────────────────────────────────┘
#
# Pipeline (9 panel-generation scripts, BLOSUM-sqrt for TCR / Lev for BCR):
#   Stage 1. gen_fig5_additional_panels.py    → panels a (placeholder), m
#   Stage 2. gen_scatter_marginals.py         → panels b, c (per-sample scatter)
#   Stage 3. gen_roc_prc.py                   → panels d, e (ROC 5-model overlay)
#   Stage 4. gen_recal_paired_boxplot.py      → panels f, g (paired boxplots) ⚠ BCR
#   Stage 5. generate_combined_dumbbell.py    → panel h (10-model ΔAUROC) ⚠ BCR
#   Stage 6. gen_ap_dumbbell.py               → panel i (10-model ΔAP) ⚠ BCR
#   Stage 7. generate_fig5_new_panels.py      → panels j, k, l (before/after) ⚠ BCR
#   Stage 8. gen_fig5_lev_vs_blosum_recal.py  → panel n (Lev vs BLOSUM)
#   Stage 9. gen_subset_recal_scatter.py      → panel o (per-epitope before/after)
#
# Output: reproduce/figures/output/fig5/{lev-logtransf,blosum-sqrt}/fig5_*.{png,pdf}
#
# Reproduction scope (post Option-B fixes 2026-06-01, panel-level):
# Of the 15 PANEL_MANIFEST panels (a–o):
#
#   ✅ 12 panels reproduce with REAL CONTENT (TCR portion at minimum):
#      a (placeholder), b (TCR scatter), d (TCR ROC 5-model), f, g
#      (paired boxplots), h, i (dumbbells), j, k (before/after scatter),
#      l (perbin delta vs distance), n (Lev vs BLOSUM), o (subset scatter)
#
#   ✗ 3 panels (c, e, m) BCR-only, blocked by missing
#     bcr_bind_ct_fold4cal/{xbcr,deepaai,mambaaai,mint,rleaai}/cal_predictions.csv
#     (2026-05-20 incident, proven irrecoverable without retraining).
#     Run `bash reproduce/retrain_fig3_inputs.sh --model bcr_ct_fold4cal`
#     to recover via Tier-2 BCR retraining.
#
# Note: for panels f, g, h, i, j, k, l, n, o the BCR portion is plotted
# alongside the TCR portion. The TCR portion ALWAYS reproduces; the BCR
# portion is empty (no BCR data points) unless Tier-2 BCR retraining is
# applied. Once Tier-2 is run + cal_predictions.csv copied to
# INPUT_DIR/results/bcr_bind_ct_fold4cal/{model}/, re-running this bash
# file fills in the BCR portions.
#
# Option-B fixes (2026-06-01):
#   1. stage_lev_distances_for_fig5.py (new Stage 0): generates the 30
#      missing Lev per-model TCR distance arrays from the shared fig2
#      LogDist computation. Reuses fig2's compute_fig2_levlog_distances.py
#      to produce {ts}_dist.npy then copies each to per-model
#      {model}_ct_{ts}_dist.npy (same distance for all 5 TCR models
#      because distances are sequence-derived, not model-output-derived).
#   2. Defensive guards added to 7 BCR-dependent scripts: when
#      cal_predictions.csv is missing, the BCR loop skips that model
#      cleanly (logged warning + continue) rather than crashing the whole
#      stage. TCR portion of each affected panel still produces.
#
# Usage:
#   bash reproduce/reproduce_fig5.sh                 # run all 9 stages
#   bash reproduce/reproduce_fig5.sh --show-training # Tier-2 retraining info
#   bash reproduce/reproduce_fig5.sh -h              # show this help
#
# Exit codes: 0 = success (with documented partial-failure stages), 1 = hard fail

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

# Defaults
SKIP_REGEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-regen) SKIP_REGEN=1; shift ;;  # skip blosum-sqrt distance regen (use cached files)
    --show-training)
      echo "=== Fig 5 Tier-2 retraining ==="
      echo
      echo "Fig 5 reuses the same per-model TCR + BCR predictions as Fig 3."
      echo "Use the Fig 3 retraining wrapper:"
      echo
      echo "    bash $REPRO_DIR/retrain_fig3_inputs.sh --list"
      echo "    bash $REPRO_DIR/retrain_fig3_inputs.sh --validate"
      echo "    bash $REPRO_DIR/retrain_fig3_inputs.sh --model bcr_ct_fold4cal"
      echo "    bash $REPRO_DIR/retrain_fig3_inputs.sh --all"
      echo
      echo "After retraining, copy {model}/cal_predictions.csv files into"
      echo "reproduce/data/input/results/bcr_bind_ct_fold4cal/{model}/"
      echo "then re-run this script for the BCR portions to populate."
      exit 0 ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "=== Fig 5 reproduction ==="
echo "Working directory: $ROOT"
echo

START=$(date +%s)
STAGE_FAILED=0
FAILED_STAGES=()

run_stage() {
  local label="$1"; local script="$2"
  echo "--- [fig5] $label: $script ---"
  (cd "$ROOT" && python3 "$REPRO_DIR/scripts/fig5/$script") 2>&1 | tail -3
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[fig5] ⚠ $label PARTIAL FAILURE (exit $rc) — continuing, but final summary will reflect"
    STAGE_FAILED=$((STAGE_FAILED + 1))
    FAILED_STAGES+=("$label")
  else
    echo "[fig5] ✓ $label done."
  fi
  echo
}

# Pre-stage cache clears so neither compute_fig2_levlog_distances.py nor
# stage_lev_distances_for_fig5.py can silently reuse stale arrays:
#  (1) data/output/fig2_cache_lev/ — Stage 1 source (per-file SKIP guard)
#  (2) INPUT_DIR/results/fig2_cache/{model}_ct_*_dist.npy — 30 per-model
#      staged copies (per-file no-overwrite guard in stage script)
INPUT_FIG2_CACHE="$REPRO_DIR/data/input/results/fig2_cache"
OUTPUT_FIG2_LEV="$REPRO_DIR/data/output/fig2_cache_lev"
if [[ -d "$OUTPUT_FIG2_LEV" ]]; then
  rm -rf "$OUTPUT_FIG2_LEV"
  echo "[fig5] cleared shared cache dir: data/output/fig2_cache_lev/"
fi
N_STAGED=$(find "$INPUT_FIG2_CACHE" -name "*_ct_*_dist.npy" -not -name "*_blosumsqrt*" 2>/dev/null | wc -l)
if [[ $N_STAGED -gt 0 ]]; then
  find "$INPUT_FIG2_CACHE" -name "*_ct_*_dist.npy" -not -name "*_blosumsqrt*" -delete 2>/dev/null
  echo "[fig5] cleared $N_STAGED staged per-model lev distance files in INPUT_DIR/results/fig2_cache/"
fi

# Optionally regenerate blosum-sqrt distance arrays (~5 min, used by 7 TCR panel scripts)
if [[ $SKIP_REGEN -eq 0 ]]; then
  N_BLOSUM=$(find "$INPUT_FIG2_CACHE" -name "*_ct_*_blosumsqrt_dist.npy" 2>/dev/null | wc -l)
  if [[ $N_BLOSUM -gt 0 ]]; then
    find "$INPUT_FIG2_CACHE" -name "*_ct_*_blosumsqrt*_dist.npy" -delete 2>/dev/null
    echo "[fig5] cleared $N_BLOSUM blosum-sqrt distance files in INPUT_DIR/results/fig2_cache/"
  fi
  echo "--- [fig5] Stage 0a (blosum-sqrt TCR CT distance regen): precompute_blosum_sqrt_distances.py --tcr-ct ---"
  (cd "$ROOT" && python3 "$REPRO_DIR/scripts/precompute_blosum_sqrt_distances.py" --tcr-ct) 2>&1 | tail -5
  echo
fi

run_stage "Stage 0 (Lev TCR distance staging for panels l, n, o)" stage_lev_distances_for_fig5.py
run_stage "Stage 1 (a placeholder, m)"                  gen_fig5_additional_panels.py
run_stage "Stage 2 (b, c — per-sample scatter)"         gen_scatter_marginals.py
run_stage "Stage 3 (d, e — ROC 5-model overlay)"        gen_roc_prc.py
run_stage "Stage 4 (f, g — paired boxplots)"            gen_recal_paired_boxplot.py
run_stage "Stage 5 (h — 10-model ΔAUROC dumbbell)"     generate_combined_dumbbell.py
run_stage "Stage 6 (i — 10-model ΔAP dumbbell)"        gen_ap_dumbbell.py
run_stage "Stage 7 (j, k, l — before/after scatter)"   generate_fig5_new_panels.py
run_stage "Stage 8 (n — Lev vs BLOSUM)"                gen_fig5_lev_vs_blosum_recal.py
run_stage "Stage 9 (o — per-epitope before/after)"     gen_subset_recal_scatter.py

END=$(date +%s)
echo
echo "[fig5] elapsed: $((END - START))s"

PANEL_DIR="$REPRO_DIR/figures/output/fig5"
if [[ -d "$PANEL_DIR" ]]; then
  N=$(find "$PANEL_DIR" -name "fig5*" 2>/dev/null | wc -l)
  echo
  if [[ $STAGE_FAILED -gt 0 ]]; then
    echo "⚠ Fig 5 finished with $STAGE_FAILED PARTIAL FAILURES out of 10 stages:"
    for s in "${FAILED_STAGES[@]}"; do echo "    ✗ $s"; done
    echo "  Output: $PANEL_DIR ($N files — some panels may be missing or contain empty axes)"
    echo "  Run \`bash $0\` again with --show-training for the Tier-2 retraining path."
  else
    echo "✓ Fig 5 reproduction complete (all 10 stages PASS)."
    echo "  Output: $PANEL_DIR ($N files)"
  fi
  echo
  echo "  Reproduction scope (post Option-B fixes 2026-06-01):"
  echo "    ✅ 12/15 panels reproduce with REAL CONTENT (TCR portion at minimum):"
  echo "       a (placeholder), b, d, f, g, h, i, j, k, l, n, o"
  echo "    ⚠ 3/15 panels (c, e, m) require BCR cal_predictions.csv — run"
  echo "       'bash reproduce/retrain_fig3_inputs.sh --model bcr_ct_fold4cal'"
  echo "       to recover (Tier-2 retraining; same wrapper as Fig 3)."
  echo
  echo "  Note: for the 9 panels f, g, h, i, j, k, l, n, o the BCR portion is"
  echo "  missing — only the TCR portion is plotted. After Tier-2 BCR retraining,"
  echo "  re-running this script populates the BCR portions of those panels."
else
  echo "[fig5] ✗ No output directory — all stages failed?"
  exit 1
fi

# Exit non-zero if any stage failed (was previously always exit 0 — silent fallback fix)
if [[ $STAGE_FAILED -gt 0 ]]; then
  exit 2
fi
exit 0
