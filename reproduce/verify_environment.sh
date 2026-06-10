#!/usr/bin/env bash
# verify_environment.sh — pre-flight check before reproduce.sh
#
# Verifies:
#   1. Python >= 3.9
#   2. Required packages importable (numpy, pandas, scipy, sklearn, matplotlib, Levenshtein)
#   3. calipper, PAPE, MCBPE importable
#   4. Optional GPU detection (non-fatal warning if absent)
#   5. reproduce/data/input/ populated (hints to run [retired] if not)
#   6. Write permission to reproduce/data/output/ and reproduce/figures/output/
#
# Exit codes:
#   0 = ready to reproduce
#   1 = critical failure (Python missing, calipper not importable, etc.)
#   2 = data missing (run [retired])

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

PASS="\033[32m✓\033[0m"
FAIL="\033[31m✗\033[0m"
WARN="\033[33m!\033[0m"

err=0
warn=0

echo "=== verify_environment.sh ==="
echo "ROOT: $ROOT"
echo

# 1. Python version
py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "missing")
if [[ "$py_version" == "missing" ]]; then
  printf "$FAIL Python 3 not found\n"
  err=1
else
  py_major=$(echo "$py_version" | cut -d. -f1)
  py_minor=$(echo "$py_version" | cut -d. -f2)
  if [[ $py_major -eq 3 && $py_minor -ge 9 ]]; then
    printf "$PASS Python $py_version (>= 3.9)\n"
  else
    printf "$FAIL Python $py_version is too old (need >= 3.9)\n"
    err=1
  fi
fi

# 2. Required packages
echo
echo "Required packages:"
for pkg in numpy pandas scipy sklearn matplotlib Levenshtein; do
  if python3 -c "import $pkg" 2>/dev/null; then
    ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
    printf "  $PASS $pkg $ver\n"
  else
    printf "  $FAIL $pkg (run: pip install $pkg)\n"
    err=1
  fi
done

# 3. CaliPPer packages
echo
echo "CaliPPer packages:"
for pkg in calipper PAPE MCBPE; do
  if (cd "$ROOT" && python3 -c "import sys; sys.path.insert(0, '.'); import $pkg" 2>/dev/null); then
    printf "  $PASS $pkg importable\n"
  else
    printf "  $FAIL $pkg not importable\n"
    err=1
  fi
done

# 4. GPU (optional, non-fatal)
echo
echo "GPU (optional, not required for figure reproduction):"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  printf "  $PASS $gpu_count GPU(s) detected\n"
else
  printf "  $WARN No GPU detected (figure reproduction is CPU-only; this is fine)\n"
fi

# 5. Data presence
echo
echo "Input data:"
data_input_dir="$REPRO_DIR/data/input"
if [[ -d "$data_input_dir" ]] && [[ -n "$(find "$data_input_dir" -type f -print -quit 2>/dev/null)" ]]; then
  file_count=$(find "$data_input_dir" -type f | wc -l)
  printf "  $PASS data/input/ populated ($file_count files)\n"
else
  printf "  $FAIL data/input/ empty or missing\n"
  echo "     → Run: # ([retired] retired 2026-06-10 — data committed to reproduce/data/input/)"
  err=2
fi

# 6. Write permissions
echo
echo "Output directories (write permission):"
for d in "$REPRO_DIR/data/output" "$REPRO_DIR/figures/output"; do
  mkdir -p "$d" 2>/dev/null
  if [[ -w "$d" ]]; then
    printf "  $PASS $d\n"
  else
    printf "  $FAIL $d (not writable)\n"
    err=1
  fi
done

echo
echo "============================="
if [[ $err -eq 0 ]]; then
  printf "$PASS ENVIRONMENT READY — you can run: bash reproduce/reproduce.sh\n"
  exit 0
elif [[ $err -eq 2 ]]; then
  printf "$FAIL DATA MISSING — run: # ([retired] retired 2026-06-10 — data committed to reproduce/data/input/)\n"
  exit 2
else
  printf "$FAIL ENVIRONMENT FAILED — fix the errors above before running reproduce.sh\n"
  exit 1
fi
