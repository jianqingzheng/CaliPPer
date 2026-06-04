#!/usr/bin/env bash
# reproduce.sh — regenerate manuscript figure values
#
# Single dispatcher (replaces 5 per-figure scripts + reproduce_all.sh).
# Invokes reproduce/scripts/figN.py drivers and runs verify.sh after.
#
# Usage:
#   bash reproduce/reproduce.sh                  # all figures (2,3,4,5,6)
#   bash reproduce/reproduce.sh --figure 2       # just fig2
#   bash reproduce/reproduce.sh --figure 2,4,6   # subset
#   bash reproduce/reproduce.sh --verify-only    # skip regeneration, just verify
#   bash reproduce/reproduce.sh --no-verify      # regenerate, skip verify
#
# Outputs per figure:
#   reproduce/data/output/figN_values.csv  (numerical, verified)
#   reproduce/figures/output/figN/*.pdf    (rendered, cosmetic only — not verified)
#
# Exit codes: 0 = all PASS, 1 = at least one failure

# NOT using `set -e` because we want to collect per-figure exit codes,
# not abort on the first failure.
set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

# Defaults
FIGURES="2,3,4,5,6"
VERIFY_ONLY=0
RUN_VERIFY=1

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --figure) FIGURES="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --no-verify) RUN_VERIFY=0; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done
[[ "$FIGURES" == "all" ]] && FIGURES="2,3,4,5,6"

# Pre-flight (skip in --verify-only mode since drivers don't run)
if [[ $VERIFY_ONLY -eq 0 ]]; then
  echo "=== Pre-flight check ==="
  if ! bash "$REPRO_DIR/verify_environment.sh"; then
    rc=$?
    echo
    if [[ $rc -eq 2 ]]; then
      echo "→ Run: bash reproduce/download_data.sh"
    fi
    exit $rc
  fi
  echo
fi

# Per-figure dispatch with exit-code collection
declare -A EXITS
declare -A TIMES

if [[ $VERIFY_ONLY -eq 0 ]]; then
  echo "=== Regenerating figures ==="
  for n in ${FIGURES//,/ }; do
    n_trim=$(echo "$n" | tr -d ' ')
    driver="$REPRO_DIR/scripts/fig${n_trim}.py"
    if [[ ! -f "$driver" ]]; then
      echo "  [fig$n_trim] ✗ driver missing: $driver"
      EXITS[fig$n_trim]=127
      continue
    fi
    echo "  [fig$n_trim] running $driver ..."
    start_time=$(date +%s)
    (cd "$ROOT" && python3 "$driver" \
      --data-root "$REPRO_DIR/data/input" \
      --out-data  "$REPRO_DIR/data/output" \
      --out-fig   "$REPRO_DIR/figures/output/fig$n_trim")
    EXITS[fig$n_trim]=$?
    end_time=$(date +%s)
    TIMES[fig$n_trim]=$((end_time - start_time))
    if [[ ${EXITS[fig$n_trim]} -eq 0 ]]; then
      echo "  [fig$n_trim] ✓ regenerated in ${TIMES[fig$n_trim]}s"
    else
      echo "  [fig$n_trim] ✗ failed (exit ${EXITS[fig$n_trim]}, ${TIMES[fig$n_trim]}s)"
    fi
  done
  echo
fi

# Run verification
if [[ $RUN_VERIFY -eq 1 ]]; then
  echo "=== Verification ==="
  bash "$REPRO_DIR/verify.sh" --figure "$FIGURES"
  VERIFY_EXIT=$?
else
  VERIFY_EXIT=0
fi

# Summary
echo
echo "=== Summary ==="
if [[ $VERIFY_ONLY -eq 0 ]]; then
  printf "%-8s %-8s %s\n" "Figure" "Status" "Time"
  printf "%-8s %-8s %s\n" "------" "------" "----"
  for k in "${!EXITS[@]}"; do
    status="PASS"; [[ ${EXITS[$k]} -ne 0 ]] && status="FAIL"
    printf "%-8s %-8s %ss\n" "$k" "$status" "${TIMES[$k]:-?}"
  done
fi
if [[ $RUN_VERIFY -eq 1 ]]; then
  if [[ $VERIFY_EXIT -eq 0 ]]; then
    echo "Verification: ✓ ALL CLEAR (numerical values match reference within 1e-10)"
  else
    echo "Verification: ✗ FAILURES — see reproduce/verification.json"
  fi
fi

# Exit non-zero if any figure failed or verification failed
for v in "${EXITS[@]}"; do
  if [[ $v -ne 0 ]]; then
    exit 1
  fi
done
exit $VERIFY_EXIT
