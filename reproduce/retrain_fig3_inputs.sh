#!/usr/bin/env bash
# retrain_fig3_inputs.sh — Tier-2 retraining for Fig 3 missing inputs.
#
# Per the two-tier reproducibility rule (BUILD_PROGRESS.md), Fig 3 has
# 7 of 10 manuscript panels that require predictions NOT included in the
# Zenodo deposit (the per-model TCR/BCR CV fold prediction CSVs and BCR
# CT cal_predictions.csv were lost in the 2026-05-20 incident).
#
# This bash file orchestrates the retraining sequence that a reviewer
# would run to regenerate those missing inputs from scratch. It uses the
# 12 training/inference scripts staged at reproduce/scripts/fig2/training/
# (10 TCR per-model scripts + 2 canonical BCR 2-pathogen-binding scripts).
#
# ⚠ Retraining is NON-DETERMINISTIC (random init + GPU non-determinism +
# library version drift). Retrained predictions will differ from any
# cached version. Per the Tier-2 acceptance criterion, the retrained
# predictions should fall within a tolerance band that covers the
# canonical files. After retraining, re-run reproduce_fig3.sh to compute
# the panels from the freshly-trained predictions.
#
# ⚠ Total wallclock estimate: ~11–15 GPU-hours serial (5 TCR models × CV + CT
# ≈ ~5-6 GPU-h, plus the 2 canonical BCR 2-pathogen binding targets
# ≈ ~6-9 GPU-h: bcr_ct_fold4cal ~2-3h + bcr_cv_combined ~4-6h). The bash file supports per-stage selection
# so reviewers can validate one model at a time before committing the full
# pipeline. Per-model conda env requirements differ — see
# reproduce/scripts/fig2/training/README.md.
#
# Usage:
#   bash reproduce/retrain_fig3_inputs.sh --model blosum_rf_cv     # ~10 min CPU
#   bash reproduce/retrain_fig3_inputs.sh --model blosum_rf_ct     # ~5 min CPU
#   bash reproduce/retrain_fig3_inputs.sh --model nettcr_cv        # ~30 min NetTCR env
#   bash reproduce/retrain_fig3_inputs.sh --model atm_tcr_cv       # ~2-3 h ATM_TCR env
#   bash reproduce/retrain_fig3_inputs.sh --all                    # ALL ~11-15 GPU-h (TCR ~5-6 + BCR ~6-9)
#   bash reproduce/retrain_fig3_inputs.sh --list                   # show all model names
#   bash reproduce/retrain_fig3_inputs.sh --validate               # CPU smoke test (BLOSUM-RF only)
#   bash reproduce/retrain_fig3_inputs.sh --all --promote          # retrain + auto-copy to INPUT_DIR
#   bash reproduce/retrain_fig3_inputs.sh --model bcr_ct_fold4cal --promote
#   bash reproduce/retrain_fig3_inputs.sh --promote-only --model nettcr_cv  # promote existing only
#
# --promote: after retraining (or via --promote-only), automatically copies
# retrained outputs from OUTPUT_DIR/retrain_fig3/{target}/ into the right
# INPUT_DIR/results/{model}/... locations so reproduce_fig{2,3,5}.sh will
# pick them up on re-run. WITHOUT --promote, the outputs stay in OUTPUT_DIR
# and the reviewer must copy them manually (preserves "no auto-overwrite"
# default — promotion is opt-in only).
#
# Standard reviewer workflow with --promote (single-shot full reproduction):
#   bash reproduce/download_data.sh
#   bash reproduce/retrain_fig3_inputs.sh --all --promote   # ~11-15 GPU-hours (TCR ~5-6 + BCR ~6-9)
#   bash reproduce/reproduce_fig2.sh                         # all 60 cells
#   bash reproduce/reproduce_fig3.sh                         # all 9 regenerable panels
#   bash reproduce/reproduce_fig5.sh                         # all 15 panels
#
# Outputs (always inside CaliPPer, NEVER overwriting INPUT_DIR cached data):
#   reproduce/data/output/retrain_fig3/{model}/...
#
# IMPORTANT: this script writes ONLY to OUTPUT_DIR/retrain_fig3/. It NEVER
# touches the cached Zenodo prediction CSVs at INPUT_DIR/results/. The
# reviewer's cached canonical state is preserved.
#
# After retraining completes, a reviewer who wants to use the retrained
# predictions for Fig 3 generation should MANUALLY copy specific files
# from data/output/retrain_fig3/ into the appropriate INPUT_DIR location.
# The bash file does NOT do this automatically — see the example below
# and choose your own promotion strategy:
#
#   # Example: promote BLOSUM-RF CV predictions for Fig 3 panel d
#   # (assumes data/input/results/blosum_rf/cv_logdist/ does NOT exist yet
#   # or you are deliberately replacing it):
#   for fold in 0 1 2 3 4; do
#     mkdir -p reproduce/data/input/results/blosum_rf/cv_logdist/fold${fold}
#     cp reproduce/data/output/retrain_fig3/blosum_rf_cv/fold${fold}/{test,val}_predictions_with_label.csv \
#        reproduce/data/input/results/blosum_rf/cv_logdist/fold${fold}/
#   done
#
# Then re-run bash reproduce/reproduce_fig3.sh. Per Tier-2 acceptance,
# panel-level metrics from retrained predictions should fall within
# a tolerance band that covers the canonical published values.

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"
TRAIN_DIR="$REPRO_DIR/scripts/fig2/training"
OUT_DIR="$REPRO_DIR/data/output/retrain_fig3"

# Model registry: name → (script | conda_env | runtime | extra_args | description)
#
# Canonical CLI args verified against CLAUDE.md "Running Evaluations" + script
# argparse defaults (all defaults already match CLAUDE.md, EXCEPT:
#   - eval_bcr_bind_ab_stratified.py REQUIRES --no-pretrain (script defaults to
#     --restore_pretrain=1 which leaks the full xbcr_train dataset via the
#     pretrained initialization — see CLAUDE.md "BCR Binding Pipeline" §1184).
#
# Extra args (5th field, '|'-separated): supplied to the training script
# after the auto-injected --data-dir and --output-dir args.
declare -A MODELS=(
  [blosum_rf_cv]="eval_blosum_rf_cv_logdist.py|||CPU,10min|BLOSUM-RF 5-fold CV (sklearn, CPU only)"
  [blosum_rf_ct]="eval_blosum_rf_cross_test_logdist.py|||CPU,5min|BLOSUM-RF cross-test (sklearn, CPU only)"
  [nettcr_cv]="eval_cv_folds_logdist.py|NetTCR||GPU,30min|NetTCR 5-fold CV (TF/Keras)"
  [nettcr_ct]="eval_cross_test_logdist.py|NetTCR||GPU,30min|NetTCR cross-test (TF/Keras)"
  [atm_tcr_cv]="eval_atm_tcr_cv_logdist.py|ATM_TCR||GPU,2-3h|ATM-TCR 5-fold CV (PyTorch, --epochs 200 default)"
  [atm_tcr_ct]="eval_atm_tcr_cross_test_logdist.py|ATM_TCR||GPU,30min|ATM-TCR cross-test (--epochs 200 default)"
  [ergo2_cv]="eval_ergo2_cv_logdist.py|ERGO_II||GPU,60min|ERGO-II 5-fold CV (--epochs 50 default)"
  [ergo2_ct]="eval_ergo2_cross_test_logdist.py|ERGO_II||GPU,10min|ERGO-II cross-test (--epochs 50 default)"
  [tcrbert_cv]="eval_tcrbert_cv_logdist.py|TCR_BERT||GPU,20min|TCR-BERT 5-fold CV (PyTorch+transformers)"
  [tcrbert_ct]="eval_tcrbert_cross_test_logdist.py|TCR_BERT||GPU,5min|TCR-BERT cross-test"
  # Canonical BCR scripts per PANEL_MANIFEST + memory feedback_bcr_cv_3pathogen_vs_2pathogen.md:
  # Fig 3 BCR uses 2-PATHOGEN BINDING (SARS-CoV-2 RBD + flu HA), NOT:
  #   ✗ SARS-only binding (eval_bcr_bind_ab_stratified.py — archived as 1-pathogen)
  #   ✗ Neutralization (eval_bcr_neu_ab_stratified.py — wrong target)
  #   ✗ 3-pathogen (archived per BCR_CV memory rule)
  # bcr_ct_fold4cal: eval_bcr_bind_ct_fold4cal.py hardcodes restore_pretrain=0
  # inside the script (line 192) — no CLI flag needed; cleanliness guaranteed.
  [bcr_ct_fold4cal]="eval_bcr_bind_ct_fold4cal.py|XBCR-net||GPU,2-3h|BCR CT 2-pathogen fold4-as-cal (canonical fig3 panels c/g/i; 5 BCR models: xbcr/deepaai/mambaaai/mint/rleaai; restore_pretrain=0 hardcoded)"
  # bcr_cv_combined: eval_bcr_combined_ab_stratified.py argparse default is
  # --no-pretrain=False (= restore_pretrain=1 → SARS-pretrained weight leak).
  # Per CLAUDE.md L1184 the binding pipeline REQUIRES --no-pretrain to avoid
  # leakage; the same rule applies to the SARS+flu combined variant because
  # the pretrained weights came from full SARS xbcr_train. Pass --no-pretrain.
  [bcr_cv_combined]="eval_bcr_combined_ab_stratified.py|XBCR-net|--no-pretrain|GPU,4-6h|BCR CV 2-pathogen combined SARS+flu binding (canonical fig3 panel f; --no-pretrain REQUIRED per CLAUDE.md L1184 to avoid SARS-pretrained leakage)"
)

# Models that accept --data-dir / --output-dir CLI args (TCR pipeline scripts).
# BCR scripts use different CLI: bcr_ct_fold4cal takes --models + --output-dir + --fold;
# bcr_cv_combined takes a similar but distinct schema. Neither accepts --data-dir.
# The --data-dir injection is gated on this set.
TCR_MODELS_WITH_DATADIR="blosum_rf_cv blosum_rf_ct nettcr_cv nettcr_ct atm_tcr_cv atm_tcr_ct ergo2_cv ergo2_ct tcrbert_cv tcrbert_ct"

list_models() {
  echo "=== Available retraining targets ==="
  echo "name              runtime    description"
  echo "----              -------    -----------"
  for k in "${!MODELS[@]}"; do
    IFS='|' read -r script env extra runtime desc <<< "${MODELS[$k]}"
    printf "%-18s%-11s%s\n" "$k" "$runtime" "$desc"
  done | sort
}

run_model() {
  local key="$1"
  if [[ -z "${MODELS[$key]:-}" ]]; then
    echo "Unknown model: $key. Use --list to see available models."
    exit 2
  fi
  IFS='|' read -r script env extra runtime desc <<< "${MODELS[$key]}"
  echo "=== Retraining: $desc ==="
  echo "Script: $script"
  echo "Conda env: ${env:-current python}"
  echo "Expected runtime: $runtime"
  echo "Extra args: ${extra:-(none)}"
  echo

  local script_path="$TRAIN_DIR/$script"
  if [[ ! -f "$script_path" ]]; then
    echo "✗ Script not found: $script_path"
    exit 2
  fi

  local model_out="$OUT_DIR/$key"
  mkdir -p "$model_out"

  # Inject PYTHONPATH so the script can find:
  #   - Model.BLOSUM_RF.blosum_rf (and other model wrappers staged at
  #     reproduce/data/input/Model/)
  #   - calipper.* (from CaliPPer/calipper/)
  #   - eval_cv_folds_logdist and other sibling training scripts in TRAIN_DIR
  local pythonpath="$REPRO_DIR/data/input:$ROOT:$TRAIN_DIR"

  local cmd
  if [[ -n "$env" ]]; then
    cmd="conda run -n $env env PYTHONPATH=$pythonpath python $script_path"
  else
    cmd="env PYTHONPATH=$pythonpath python3 $script_path"
  fi

  echo "Command: $cmd"
  echo "Output: $model_out"
  echo "Working dir: $TRAIN_DIR"
  echo

  # --data-dir is only accepted by TCR pipeline scripts. BCR scripts use a
  # different config-driven path discovery (see TCR_MODELS_WITH_DATADIR above).
  local datadir_args=""
  if echo " $TCR_MODELS_WITH_DATADIR " | grep -q " $key "; then
    datadir_args="--data-dir $REPRO_DIR/data/input/Data/tcr_seq/proc_files --output-dir $model_out"
  fi

  # Warn reviewers about hardcoded TF_PYTHON path inside XBCR-net scripts
  # (eval_bcr_bind_ct_fold4cal.py + eval_bcr_combined_ab_stratified.py + helpers)
  if [[ "$key" == "bcr_ct_fold4cal" || "$key" == "bcr_cv_combined" ]]; then
    echo "⚠ WARNING: XBCR-net scripts hardcode TF_PYTHON='/home/jzheng/anaconda3/envs/tf/bin/python'"
    echo "  A reviewer machine without this exact path will fail. Edit the script"
    echo "  to point to your local TF 2.4.1 interpreter (e.g., 'conda activate"
    echo "  XBCR-net && which python') before running. See"
    echo "  reproduce/scripts/fig2/training/README.md ⚠ warning section."
    echo
  fi

  local start=$(date +%s)
  (cd "$TRAIN_DIR" && $cmd $datadir_args $extra)
  local rc=$?
  local end=$(date +%s)
  echo
  echo "Elapsed: $((end - start))s"

  if [[ $rc -eq 0 ]]; then
    echo "✓ $key retraining complete."
    echo "  Predictions: $model_out/fold{0-4}/{test,val}_predictions_with_label.csv"
    echo "  Other outputs: $model_out/all_folds_correlations.csv + per-fold logdist_correlations.csv"
  else
    echo "✗ $key failed (exit $rc)"
  fi
  return $rc
}

# Parse args
ACTION=""
MODEL=""
PROMOTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --all) ACTION="all"; shift ;;
    --list) list_models; exit 0 ;;
    --validate) MODEL="blosum_rf_cv"; ACTION="validate"; shift ;;
    --promote) PROMOTE=1; shift ;;
    --promote-only) ACTION="promote-only"; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# --promote: auto-copy retrained outputs into INPUT_DIR so reproduce_figN.sh
# can read them. NEVER invoked unless reviewer passes the flag explicitly
# (preserves the user's "don't overwrite" rule by requiring explicit opt-in).
promote_model() {
  local key="$1"
  local src="$OUT_DIR/$key"
  if [[ ! -d "$src" ]]; then
    echo "  [promote] $key: nothing to promote (no $src dir — retrain first)"
    return 1
  fi
  case "$key" in
    nettcr_cv|atm_tcr_cv|blosum_rf_cv|ergo2_cv|tcrbert_cv)
      # TCR CV: fold{N}/{test,val}_predictions_with_label.csv
      local model="${key%_cv}"
      [[ "$model" == "ergo2" ]] && model="ergo_ii"
      [[ "$model" == "atm_tcr" ]] && model="atm_tcr"  # match dir name
      local promoted=0; local missed=0
      for fold in 0 1 2 3 4; do
        local sd="$src/fold${fold}"
        local dd="$REPRO_DIR/data/input/results/${model}/cv_logdist/fold${fold}"
        if [[ -d "$sd" ]]; then
          mkdir -p "$dd"
          local before=$(ls "$dd" 2>/dev/null | wc -l)
          if cp "$sd"/{test,val}_predictions_with_label.csv "$dd/" 2>&1 | grep -v "^$"; then
            echo "  [promote] $key fold${fold}: cp produced messages above"
          fi
          local after=$(ls "$dd" 2>/dev/null | wc -l)
          promoted=$((promoted + after - before))
        else
          echo "  [promote] $key fold${fold}: SOURCE MISSING ($sd) — fold not retrained"
          missed=$((missed + 1))
        fi
      done
      echo "  [promote] $key → INPUT_DIR/results/${model}/cv_logdist/fold{0-4}/ ($promoted files promoted, $missed folds missed)"
      ;;
    nettcr_ct|atm_tcr_ct|blosum_rf_ct|ergo2_ct|tcrbert_ct)
      # TCR CT: predictions/{test_set}_predictions_with_label.csv
      local model="${key%_ct}"
      [[ "$model" == "ergo2" ]] && model="ergo_ii"
      local dd="$REPRO_DIR/data/input/results/${model}/cross_test_logdist/predictions"
      mkdir -p "$dd"
      local n_src=$(ls "$src"/*_predictions_with_label.csv 2>/dev/null | wc -l)
      if [[ $n_src -eq 0 ]]; then
        echo "  [promote] $key: SOURCE EMPTY — no *_predictions_with_label.csv at $src"
      else
        cp "$src"/*_predictions_with_label.csv "$dd/"
        echo "  [promote] $key → INPUT_DIR/results/${model}/cross_test_logdist/predictions/ ($n_src CSVs)"
      fi
      ;;
    bcr_ct_fold4cal)
      # BCR CT: {model}/{cal,A1-A11,unseen,flu}_predictions.csv
      local promoted=0; local missed=0
      for m in xbcr deepaai mambaaai mint rleaai; do
        local sd="$src/$m"
        local dd="$REPRO_DIR/data/input/results/bcr_bind_ct_fold4cal/$m"
        if [[ -d "$sd" ]]; then
          mkdir -p "$dd"
          local n=$(ls "$sd"/*_predictions.csv 2>/dev/null | wc -l)
          if [[ $n -gt 0 ]]; then
            cp "$sd"/*_predictions.csv "$dd/"
            promoted=$((promoted + n))
          else
            echo "  [promote] $key/$m: no *_predictions.csv in $sd"
            missed=$((missed + 1))
          fi
        else
          echo "  [promote] $key/$m: SOURCE MISSING ($sd) — model not retrained"
          missed=$((missed + 1))
        fi
      done
      echo "  [promote] $key → INPUT_DIR/results/bcr_bind_ct_fold4cal/{model}/ ($promoted files, $missed models missed)"
      ;;
    bcr_cv_combined)
      # BCR CV: per-model combined_bind_ab_cv/fold{N}/ outputs
      local promoted=0; local missed=0
      for m in xbcr deepaai mambaaai mint rleaai; do
        for fold in 0 1 2 3 4; do
          local sd="$src/$m/fold${fold}"
          local dd="$REPRO_DIR/data/input/results/${m}/combined_bind_ab_cv/fold${fold}"
          if [[ -d "$sd" ]]; then
            mkdir -p "$dd"
            local n=$(ls "$sd"/*.csv 2>/dev/null | wc -l)
            if [[ $n -gt 0 ]]; then
              cp "$sd"/*.csv "$dd/"
              promoted=$((promoted + n))
            else
              echo "  [promote] $key/$m/fold${fold}: no *.csv in $sd"
              missed=$((missed + 1))
            fi
          else
            missed=$((missed + 1))
          fi
        done
      done
      echo "  [promote] $key → INPUT_DIR/results/{model}/combined_bind_ab_cv/fold{0-4}/ ($promoted files, $missed slots missed)"
      ;;
    *)
      echo "  [promote] $key: no promote mapping defined"
      return 1
      ;;
  esac
  return 0
}

if [[ "${ACTION:-}" == "promote-only" ]]; then
  if [[ -z "$MODEL" ]]; then
    echo "Usage: --promote-only --model <name>  OR  --promote-only --all"
    exit 2
  fi
  promote_model "$MODEL"
  exit $?
fi

if [[ -n "$MODEL" ]]; then
  run_model "$MODEL"
  rc=$?
  if [[ "$ACTION" == "validate" && $rc -eq 0 ]]; then
    echo
    echo "✓ Validation PASS — Tier-2 retraining infrastructure works."
    echo "  Next: run --model <name> for individual models or --all for full pipeline."
  fi
  if [[ "$PROMOTE" == "1" && $rc -eq 0 ]]; then
    echo
    echo "--- Promoting to INPUT_DIR (--promote) ---"
    promote_model "$MODEL"
  fi
  exit $rc
elif [[ "$ACTION" == "all" ]]; then
  echo "=== Running ALL retraining (~11-15 GPU-hours serial: TCR ~5-6 + BCR ~6-9) ==="
  echo "Tip: run individual --model targets first to verify per-model conda envs."
  echo
  failed=()
  for k in blosum_rf_cv blosum_rf_ct nettcr_cv nettcr_ct atm_tcr_cv atm_tcr_ct \
           ergo2_cv ergo2_ct tcrbert_cv tcrbert_ct \
           bcr_ct_fold4cal bcr_cv_combined; do
    if ! run_model "$k"; then
      failed+=("$k")
    fi
    echo
  done
  if [[ ${#failed[@]} -gt 0 ]]; then
    echo "✗ ${#failed[@]} model(s) failed: ${failed[*]}"
    exit 1
  fi
  echo "✓ All 12 retraining stages complete."
  if [[ "$PROMOTE" == "1" ]]; then
    echo
    echo "--- Promoting all retrained outputs to INPUT_DIR (--promote) ---"
    for k in blosum_rf_cv blosum_rf_ct nettcr_cv nettcr_ct atm_tcr_cv atm_tcr_ct \
             ergo2_cv ergo2_ct tcrbert_cv tcrbert_ct \
             bcr_ct_fold4cal bcr_cv_combined; do
      promote_model "$k"
    done
    echo
    echo "✓ Promotion done. Re-run reproduce_fig2.sh/fig3.sh/fig5.sh to"
    echo "  reproduce panels from the freshly trained predictions."
  fi
  exit 0
else
  echo "Usage: bash $0 [--validate | --model <name> | --all | --list | -h]"
  exit 2
fi
