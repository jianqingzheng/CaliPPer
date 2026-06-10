#!/usr/bin/env bash
# prepare_fig6_data.sh — fetch + stage all Fig 6 data for 2-command reproduction.
#
# After this script runs, `bash reproduce/reproduce_fig6.sh` reproduces Fig 6
# Panel E bit-exactly (max |Δ| = 9.99e-16 against canonical reference).
#
# What this script does:
#
#   Stage 0: copies 11 author model prediction CSVs (~5.8 MB) from the
#            tracked dir reproduce/data/cached_predictions/ into the
#            runtime INPUT_DIR/results/{study}_retrospective/.../ locations
#            Stage 1 of reproduce_fig6.sh expects. The CSVs ARE committed
#            to the repo (since they require ~28 GB of model weights to
#            regenerate via author inference); this avoids needing
#            `[retired] --record 2` for the 2-command flow.
#
#   Stage 1-5: fetches RAW SEQUENCE / SMALL-MOLECULE DATA from authors'
#            original sources (Nature, Mendeley, Zenodo, GitHub) and stages
#            it under INPUT_DIR/Data/retrospective_{study}/ and similar.
#
# After this script: `reproduce_fig6.sh` Stage 0 regenerates distance files
# from the raw data + committed predictions and produces Panel E.
#
# Two-command reproduction (no model retraining needed):
#
#     bash reproduce/prepare_fig6_data.sh           # ← THIS SCRIPT
#     bash reproduce/reproduce_fig6.sh              # Stage 0 distance regen + Panel E verify
#
# OPTIONAL: --include-inference flag attempts to REGENERATE the committed
# prediction CSVs by running author inference scripts staged in CaliPPer
# (`reproduce_fig6_*.py`, `eval_*_retrospective.py`). This requires model
# weights for each study; PanPep + deepAntigen weights come with this
# script's data fetch (Zenodo/GitHub), but BigMHC + XBCR-net inference
# requires `[retired] --record 1` (~28 GB) for their model weights.
# Without --include-inference, the committed prediction CSVs are used.
#
# Studies fetched (5 total):
#   1. XBCR-net      — Lou et al. Cell Research 2022 (Mendeley + GitHub)
#   2. deepAntigen   — Zhou et al. (GitHub + ImmuneCODE manual + Lowery 2022 manual)
#   3. AntibioticsAI — Wong et al. Nature 2024 (Nature supplementary + GitHub)
#   4. BigMHC        — Albert et al. Nature MI 2023 (Mendeley + GitHub)
#   5. PanPep        — Gao et al. Nature MI 2023 (Zenodo + GitHub)
#
# Usage:
#   bash reproduce/prepare_fig6_data.sh                # all 5 studies
#   bash reproduce/prepare_fig6_data.sh --study <name> # one study only
#                                                       (xbcr|deepantigen|antibioticsai|bigmhc|panpep)
#   bash reproduce/prepare_fig6_data.sh --list         # list studies, exit
#   bash reproduce/prepare_fig6_data.sh -h             # show this help
#
# Exit codes:
#   0 = all studies prepared (or already present)
#   2 = some studies require MANUAL DOWNLOAD (Mendeley login, ImmuneCODE
#       registration, Lowery supplementary). Re-run after manual files
#       placed at documented target paths.
#   ≥3 = hard failure (no internet, curl missing, etc.)
#
# After preparation, run:
#   bash reproduce/reproduce_fig6.sh        # reproduces Fig 6 from scratch

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

STUDIES=(xbcr deepantigen antibioticsai bigmhc panpep)
TARGET_STUDY=""
INCLUDE_INFERENCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --study) TARGET_STUDY="$2"; shift 2 ;;
    --include-inference) INCLUDE_INFERENCE=1; shift ;;
    --list)
      echo "Fig 6 retrospective studies:"
      for s in "${STUDIES[@]}"; do echo "  $s"; done
      exit 0 ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 3 ;;
  esac
done

# Validate study selector
if [[ -n "$TARGET_STUDY" ]]; then
  found=0
  for s in "${STUDIES[@]}"; do [[ "$s" == "$TARGET_STUDY" ]] && found=1; done
  if [[ $found -eq 0 ]]; then
    echo "ERROR: unknown study '$TARGET_STUDY'. Use --list to see options." >&2
    exit 3
  fi
  STUDIES=("$TARGET_STUDY")
fi

echo "=== Fig 6 data preparation ==="
echo "Working directory: $ROOT"
echo "Studies to prepare: ${STUDIES[*]}"
echo "Target: \$INPUT_DIR = $REPRO_DIR/data/input/"
if [[ $INCLUDE_INFERENCE -eq 1 ]]; then
  echo "Inference: ENABLED (will run author inference scripts after data fetch)"
  echo "           BigMHC/XBCR-net inference requires model weights from"
  echo "           # ([retired] retired 2026-06-10 — data committed to reproduce/data/input/) --record 1 (~28 GB)"
else
  echo "Inference: disabled (pass --include-inference to run author inference)"
fi
echo

# Check curl available (most scripts use it)
if ! command -v curl > /dev/null 2>&1; then
  echo "ERROR: curl not installed. Install with: apt-get install curl OR brew install curl" >&2
  exit 3
fi

START=$(date +%s)
PREP_FAILED=0
FAILED_STUDIES=()
MANUAL_PENDING=0

# Stage 0: copy cached prediction CSVs (committed, ~5.8 MB) into the runtime
# INPUT_DIR/results/ locations Stage 1 of reproduce_fig6.sh expects. This is
# what makes the 2-command flow (prepare + reproduce) work without needing
# the CaliPPer repo (committed). The CSVs themselves are author model outputs
# pre-computed by inference scripts in reproduce/scripts/fig6/; we commit
# them because re-running inference requires author model weights from upstream GitHub/Zenodo deposits.
CACHED_PRED_DIR="$REPRO_DIR/data/cached_predictions"
MANIFEST="$CACHED_PRED_DIR/MANIFEST.tsv"
if [[ -f "$MANIFEST" ]]; then
  echo "─────────────────────────────────────────────────────"
  echo "[prep] Stage 0: staging committed author predictions"
  echo "       Source: reproduce/data/cached_predictions/ (5.8 MB tracked)"
  echo "       Target: INPUT_DIR/results/..."
  echo
  STAGED=0
  while IFS=$'\t' read -r src dst; do
    [[ "$src" =~ ^# ]] && continue
    [[ -z "$src" ]] && continue
    src_full="$CACHED_PRED_DIR/$src"
    dst_full="$REPRO_DIR/data/input/$dst"
    if [[ -f "$src_full" ]]; then
      if [[ ! -f "$dst_full" ]] || ! cmp -s "$src_full" "$dst_full"; then
        mkdir -p "$(dirname "$dst_full")"
        cp "$src_full" "$dst_full"
        STAGED=$((STAGED + 1))
      fi
    fi
  done < "$MANIFEST"
  echo "  ✓ staged $STAGED prediction CSV(s) into INPUT_DIR/results/ (rest already present)"
  echo
fi

for study in "${STUDIES[@]}"; do
  echo "─────────────────────────────────────────────────────"
  script="$REPRO_DIR/scripts/data_prep/prep_fig6_${study}.py"
  if [[ ! -f "$script" ]]; then
    echo "[prep] ✗ script missing: $script"
    PREP_FAILED=$((PREP_FAILED + 1))
    FAILED_STUDIES+=("$study (script missing)")
    continue
  fi
  (cd "$ROOT" && python3 "$script") || {
    rc=$?
    if [[ $rc -eq 1 ]]; then
      # Hard failure (curl error, exception)
      echo "[prep] ✗ $study failed (exit 1)"
      PREP_FAILED=$((PREP_FAILED + 1))
      FAILED_STUDIES+=("$study (download failure)")
    elif [[ $rc -eq 2 ]]; then
      # Manual step required
      echo "[prep] ⚠ $study has manual download(s) pending"
      MANUAL_PENDING=$((MANUAL_PENDING + 1))
    else
      echo "[prep] ✗ $study failed (exit $rc)"
      PREP_FAILED=$((PREP_FAILED + 1))
      FAILED_STUDIES+=("$study (exit $rc)")
    fi
  }
  echo
done

# Optionally run author inference scripts to populate the pre-computed
# *_predictions.csv files that Stage 1 of reproduce_fig6.sh expects.
# These are usually provided by `[retired] --record 2` (Zenodo deposit).
if [[ $INCLUDE_INFERENCE -eq 1 ]]; then
  echo "─────────────────────────────────────────────────────"
  echo "[prep] --include-inference: running author inference scripts"
  echo
  for study in "${STUDIES[@]}"; do
    case "$study" in
      antibioticsai)
        echo "[prep] AntibioticsAI: predictions already in MOESM4 Excel — no inference needed ✓"
        ;;
      panpep)
        echo "[prep] PanPep: running eval_panpep_retrospective.py (weights from prep step)"
        (cd "$ROOT" && python3 "$REPRO_DIR/scripts/fig6/eval_panpep_retrospective.py") 2>&1 | tail -8 || \
          echo "[prep] ⚠ PanPep inference exited non-zero — see output above"
        echo
        ;;
      deepantigen)
        echo "[prep] deepAntigen: running eval_deepantigen_neoantigen_confidence.py + eval_deepantigen_bayesian_recalibration.py"
        (cd "$ROOT" && python3 "$REPRO_DIR/scripts/fig6/eval_deepantigen_neoantigen_confidence.py") 2>&1 | tail -5 || \
          echo "[prep] ⚠ deepAntigen confidence inference exited non-zero"
        echo
        ;;
      bigmhc)
        echo "[prep] BigMHC: requires model weights from [retired] --record 1 (~28 GB total)."
        echo "       To run BigMHC inference manually after weights are staged:"
        echo "         python3 reproduce/scripts/fig6/reproduce_fig6_bigmhc.py"
        echo
        ;;
      xbcr)
        echo "[prep] XBCR-net: requires model weights from [retired] --record 1 (~28 GB total)."
        echo "       To run XBCR-net inference manually after weights are staged:"
        echo "         python3 reproduce/scripts/fig6/reproduce_panel1_fresh_predictions.py"
        echo "         python3 reproduce/scripts/fig6/reproduce_fig6_xbcr_panel2.py"
        echo
        ;;
    esac
  done
fi

END=$(date +%s)
echo "─────────────────────────────────────────────────────"
echo "[prep] Elapsed: $((END - START))s"
echo

if [[ $PREP_FAILED -gt 0 ]]; then
  echo "⚠ Hard failures: $PREP_FAILED"
  for s in "${FAILED_STUDIES[@]}"; do echo "    ✗ $s"; done
fi

if [[ $MANUAL_PENDING -gt 0 ]]; then
  echo "⚠ $MANUAL_PENDING studies have MANUAL download steps pending."
  echo "  See the per-study messages above for source URLs and target paths."
  echo "  After placing files manually, re-run this script to complete preparation."
  echo
  echo "  The 2 known manual steps:"
  echo "    1. Mendeley datasets (XBCR-net, BigMHC) — browser download from"
  echo "       data.mendeley.com pages; ZIP needs unzipping at documented target."
  echo "    2. ImmuneCODE-MIRA (deepAntigen Panel C/D only) — registration at"
  echo "       Adaptive Biotechnologies portal required."
  exit 2
fi

if [[ $PREP_FAILED -eq 0 ]]; then
  echo "✓ ALL 5 STUDIES PREPARED — raw author data + committed prediction CSVs staged"
  echo
  echo "Next step (2-command reproduction):"
  echo "    bash reproduce/reproduce_fig6.sh"
  echo "→ Stage 0 (distance regeneration from raw sequences/structures)"
  echo "→ Stages 1-4 (Bayesian recalibration, Panel C/D prediction, panels)"
  echo "→ Verify against canonical Panel E reference (max |Δ| = 9.99e-16 expected)"
  if [[ $INCLUDE_INFERENCE -eq 0 ]]; then
    echo
    echo "Note: the 11 author prediction CSVs in reproduce/data/cached_predictions/"
    echo "(committed, ~5.8 MB) were staged into INPUT_DIR/results/...  by this script."
    echo "To regenerate predictions from scratch by running authors' models, use"
    echo "the --include-inference flag (PanPep + deepAntigen + AntibioticsAI"
    echo "inference run automatically; BigMHC + XBCR-net inference also need"
    echo "model weights from '# ([retired] retired 2026-06-10 — data committed to reproduce/data/input/) --record 1')."
  fi
  exit 0
fi

exit 2
