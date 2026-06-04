#!/usr/bin/env bash
# download_data.sh — fetch Zenodo data + model weights
#
# Fetches:
#   Record 1 — model weights (~28 GB) → models/<model>/weights/ + models/retrospective/<model>/weights/
#   Record 2 — bulk inputs (~9 GB)   → reproduce/data/input/
#
# DOI placeholders are in reproduce/_zenodo.env (sed-replaced at P11 with real DOIs).
#
# Usage:
#   bash reproduce/download_data.sh                 # both records
#   bash reproduce/download_data.sh --record 1      # weights only
#   bash reproduce/download_data.sh --record 2      # inputs only
#   bash reproduce/download_data.sh --check         # check what's missing (no download)
#
# Env vars:
#   CALIPPER_DATA_MIRROR — optional HTTPS mirror prefix (override Zenodo URLs)

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

# Source DOI configuration
# shellcheck source=_zenodo.env
source "$REPRO_DIR/_zenodo.env"

# Parse args
RECORD="all"
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --record) RECORD="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# DOI placeholder check
if [[ "$ZENODO_WEIGHTS_DOI" == *XXXXXX* ]] || [[ "$ZENODO_DATA_DOI" == *YYYYYY* ]]; then
  echo "==============================================================" >&2
  echo "ERROR: Zenodo DOIs are still placeholders in reproduce/_zenodo.env." >&2
  echo "" >&2
  echo "This is expected for the v1.0 pre-release. The actual DOIs will be" >&2
  echo "populated after Zenodo upload (build plan phase P11)." >&2
  echo "" >&2
  echo "For now, please obtain the data through the manuscript-supplied link" >&2
  echo "or contact the corresponding author. After v1.0 release, this script" >&2
  echo "will fetch the data automatically." >&2
  echo "==============================================================" >&2
  exit 1
fi

# Override URLs with mirror if set
WEIGHTS_URL="${CALIPPER_DATA_MIRROR:-$ZENODO_WEIGHTS_URL}"
DATA_URL="${CALIPPER_DATA_MIRROR:-$ZENODO_DATA_URL}"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "  ($(basename "$out") already present, skipping)"
    return 0
  fi
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "  WOULD DOWNLOAD: $url → $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "  Fetching $(basename "$out")..."
  curl --fail --location --silent --show-error \
       --retry 3 --retry-delay 5 \
       -o "$out" "$url" || {
    echo "  ERROR: download failed for $url" >&2
    return 1
  }
}

# Record 1 — model weights
fetch_weights() {
  echo "=== Record 1: model weights (~28 GB) ==="
  # Each model is its own archive in the Zenodo record
  # Primary models
  for m in nettcr atm_tcr blosum_rf ergo_ii tcr_bert xbcr_net deepaai mambaaai mint rleaai; do
    download "$WEIGHTS_URL/${m}_weights.tar.gz" "$ROOT/models/$m/weights/${m}_weights.tar.gz"
    if [[ $CHECK_ONLY -eq 0 && -f "$ROOT/models/$m/weights/${m}_weights.tar.gz" ]]; then
      tar -xzf "$ROOT/models/$m/weights/${m}_weights.tar.gz" -C "$ROOT/models/$m/weights/" --strip-components=1
      rm -f "$ROOT/models/$m/weights/${m}_weights.tar.gz"
    fi
  done
  # Retrospective models
  for m in deepantigen panpep bigmhc antibioticsai; do
    download "$WEIGHTS_URL/${m}_retrospective_weights.tar.gz" "$ROOT/models/retrospective/$m/weights/${m}_weights.tar.gz"
    if [[ $CHECK_ONLY -eq 0 && -f "$ROOT/models/retrospective/$m/weights/${m}_weights.tar.gz" ]]; then
      tar -xzf "$ROOT/models/retrospective/$m/weights/${m}_weights.tar.gz" \
        -C "$ROOT/models/retrospective/$m/weights/" --strip-components=1
      rm -f "$ROOT/models/retrospective/$m/weights/${m}_weights.tar.gz"
    fi
  done
}

# Record 2 — bulk inputs
fetch_data() {
  echo "=== Record 2: bulk inputs (~9 GB) ==="
  # A single bundle for simplicity; could be split per group (tcr/, bcr/, retrospective/) if needed
  download "$DATA_URL/calipper_data_v1.0.tar.gz" "$REPRO_DIR/data/input/calipper_data.tar.gz"
  if [[ $CHECK_ONLY -eq 0 && -f "$REPRO_DIR/data/input/calipper_data.tar.gz" ]]; then
    tar -xzf "$REPRO_DIR/data/input/calipper_data.tar.gz" -C "$REPRO_DIR/data/input/" --strip-components=1
    rm -f "$REPRO_DIR/data/input/calipper_data.tar.gz"
  fi
}

# Dispatch
case "$RECORD" in
  1|weights) fetch_weights ;;
  2|data) fetch_data ;;
  all) fetch_weights; fetch_data ;;
  *) echo "Unknown record: $RECORD (use 1, 2, or all)"; exit 2 ;;
esac

echo
echo "==============================="
if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "Check complete (no files downloaded). Run without --check to actually download."
else
  echo "Download complete. Run: bash reproduce/verify_environment.sh"
fi
