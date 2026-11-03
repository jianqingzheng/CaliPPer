#!/usr/bin/env bash
#
# Run the full S2DD data preprocessing pipeline.
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --config subsample_n=1000 subsample_seed=42
#   ./run_all.sh --project-root /path/to/tcr_ml.nosync
#
# Creates a timestamped output directory under outputs/ with:
#   - config.json           (run configuration + git SHA)
#   - 01_merge/             (merged paired TCRs)
#   - 02_cluster_fold/      (with clusters and fold assignments)
#   - 03_features/          (CDR1/2, MHC contacts, weights)
#   - 04_export/            (final train/val/test CSVs)
#   - logs/                 (per-stage log files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse arguments ---
PROJECT_ROOT=""
SUBSAMPLE_N=""
SUBSAMPLE_SEED="42"
SKIP_HYPEROPT=""
CACHED_10X_PARAMS=""
INCLUDE_10X=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root)
            PROJECT_ROOT="$2"; shift 2 ;;
        --include-10x)
            INCLUDE_10X="1"; shift ;;
        --skip-hyperopt)
            SKIP_HYPEROPT="1"; shift ;;
        --cached-10x-params)
            CACHED_10X_PARAMS="$2"; shift 2 ;;
        --config)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                key="${1%%=*}"
                val="${1#*=}"
                case "$key" in
                    subsample_n) SUBSAMPLE_N="$val" ;;
                    subsample_seed) SUBSAMPLE_SEED="$val" ;;
                    *) echo "Unknown config key: $key"; exit 1 ;;
                esac
                shift
            done
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default project root: two levels up from script dir
if [[ -z "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# --- Create timestamped output directory ---
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/outputs/$TIMESTAMP"
mkdir -p "$RUN_DIR/logs"

echo "=== S2DD Data Preprocessing Pipeline ==="
echo "Timestamp:    $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"
echo "Output dir:   $RUN_DIR"
[[ -n "$SUBSAMPLE_N" ]] && echo "Subsample:    n=$SUBSAMPLE_N, seed=$SUBSAMPLE_SEED"
echo ""

# --- Write config.json ---
GIT_SHA="$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
cat > "$RUN_DIR/config.json" <<EOF
{
    "timestamp": "$TIMESTAMP",
    "project_root": "$PROJECT_ROOT",
    "git_sha": "$GIT_SHA",
    "subsample_n": ${SUBSAMPLE_N:-null},
    "subsample_seed": $SUBSAMPLE_SEED,
    "include_10x": $([[ -n "$INCLUDE_10X" ]] && echo "true" || echo "false"),
    "skip_hyperopt": $([[ -n "$SKIP_HYPEROPT" ]] && echo "true" || echo "false"),
    "cached_10x_params": "${CACHED_10X_PARAMS}",
    "python": "$(uv run python --version 2>&1)"
}
EOF
echo "Config written to $RUN_DIR/config.json"
echo ""

# --- Build subsample args ---
SUBSAMPLE_ARGS=""
if [[ -n "$SUBSAMPLE_N" ]]; then
    SUBSAMPLE_ARGS="--subsample-n $SUBSAMPLE_N --subsample-seed $SUBSAMPLE_SEED"
fi

# --- Helper to run a stage ---
run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local stage_dir="$RUN_DIR/$stage_name"
    local log_file="$RUN_DIR/logs/${stage_name}.log"
    shift 2

    echo "--- Stage $stage_num: $stage_name ---"
    local start_time
    start_time=$(date +%s)

    mkdir -p "$stage_dir"

    # Run the script, tee to log
    if "$@" 2>&1 | tee "$log_file"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo "  Completed in ${elapsed}s"
        echo ""
    else
        echo "  FAILED — see $log_file"
        exit 1
    fi
}

# --- Run stages ---

# Stage 0: 10x denoising — only runs when --include-10x is set.
MERGE_TENX_ARG=""
if [[ -n "$INCLUDE_10X" ]]; then
    if [[ -n "$CACHED_10X_PARAMS" && -f "$CACHED_10X_PARAMS" ]]; then
        mkdir -p "$RUN_DIR/00_denoise_10x"
        cp "$CACHED_10X_PARAMS" "$RUN_DIR/00_denoise_10x/optimal_parameters.json"
        echo "Seeded cached 10x params from $CACHED_10X_PARAMS"
    fi

    DENOISE_ARGS="--seed $SUBSAMPLE_SEED"
    [[ -n "$SKIP_HYPEROPT" ]] && DENOISE_ARGS="$DENOISE_ARGS --skip-hyperopt"

    run_stage 0 "00_denoise_10x" \
        uv run python "$SCRIPT_DIR/00_denoise_10x.py" \
        --project-root "$PROJECT_ROOT" \
        --output-dir "$RUN_DIR/00_denoise_10x" \
        $DENOISE_ARGS

    MERGE_TENX_ARG="--denoised-10x-csv $RUN_DIR/00_denoise_10x/single_10x_binders.csv"
else
    echo "--- Stage 0 skipped (10x denoising disabled; pass --include-10x to enable) ---"
    echo ""
fi

run_stage 1 "01_merge" \
    uv run python "$SCRIPT_DIR/01_merge_databases.py" \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$RUN_DIR/01_merge" \
    $MERGE_TENX_ARG \
    $SUBSAMPLE_ARGS

run_stage 2 "02_cluster_fold" \
    uv run python "$SCRIPT_DIR/02_cluster_and_fold.py" \
    --input-dir "$RUN_DIR/01_merge" \
    --output-dir "$RUN_DIR/02_cluster_fold" \
    --seed "$SUBSAMPLE_SEED"

run_stage 3 "03_features" \
    uv run python "$SCRIPT_DIR/03_assign_features.py" \
    --input-dir "$RUN_DIR/02_cluster_fold" \
    --output-dir "$RUN_DIR/03_features" \
    --project-root "$PROJECT_ROOT"

run_stage 4 "04_export" \
    uv run python "$SCRIPT_DIR/04_export_jq_format.py" \
    --input-dir "$RUN_DIR/03_features" \
    --output-dir "$RUN_DIR/04_export" \
    --seed "$SUBSAMPLE_SEED"

# --- Summary ---
echo "=== Pipeline complete ==="
echo "Output directory: $RUN_DIR"
echo ""
echo "Final outputs:"
for f in "$RUN_DIR/04_export"/*.csv; do
    lines=$(wc -l < "$f" | tr -d ' ')
    echo "  $(basename "$f"): $lines lines"
done
