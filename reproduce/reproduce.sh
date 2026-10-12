#!/usr/bin/env bash
# reproduce.sh — regenerate manuscript figure values (dispatcher).
#
# Thin dispatcher over the per-figure scripts reproduce_fig{2,3,4,5,6}.sh,
# which are the documented reproduction entry points (see README.md).
# Figure 6 runs its own numerical verification (verify.sh) internally; the
# other figures regenerate their panel data.
#
# Usage:
#   bash reproduce/reproduce.sh                  # all figures (2,3,4,5,6)
#   bash reproduce/reproduce.sh --figure 2       # just fig2
#   bash reproduce/reproduce.sh --figure 2,4,6   # subset
#   bash reproduce/reproduce.sh --verify-only    # skip regeneration; verify fig6 values only
#   bash reproduce/reproduce.sh --no-verify      # regenerate, skip fig6 verify gate
#
# Exit codes: 0 = all requested figures succeeded (fig6 verify passed), 1 = failure

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIGURES="2,3,4,5,6"
VERIFY_ONLY=0
RUN_VERIFY=1

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

# --verify-only: re-check fig6 numerical values against the committed reference
# (fig6 is the only figure with a committed scalar reference); nothing to regenerate.
if [[ $VERIFY_ONLY -eq 1 ]]; then
  bash "$REPRO_DIR/verify.sh" --figure 6
  exit $?
fi

# Pre-flight environment check
echo "=== Pre-flight check ==="
if ! bash "$REPRO_DIR/verify_environment.sh"; then
  rc=$?
  if [[ $rc -eq 2 ]]; then
    echo "→ Data missing: ensure reproduce/data/input/ is present (committed in this repo)."
  fi
  exit $rc
fi
echo

# Per-figure dispatch to the documented reproduce_figN.sh scripts
declare -A EXITS
echo "=== Regenerating figures ==="
for n in ${FIGURES//,/ }; do
  n_trim=$(echo "$n" | tr -d ' ')
  shell="$REPRO_DIR/reproduce_fig${n_trim}.sh"
  if [[ ! -f "$shell" ]]; then
    echo "  [fig$n_trim] ✗ script missing: $shell"
    EXITS[fig$n_trim]=127
    continue
  fi
  echo "  [fig$n_trim] running reproduce_fig${n_trim}.sh ..."
  if [[ "$n_trim" == "6" && $RUN_VERIFY -eq 0 ]]; then
    bash "$shell" --no-verify
  else
    bash "$shell"
  fi
  EXITS[fig$n_trim]=$?
done
echo

# Summary
echo "=== Summary ==="
printf "%-8s %-8s\n" "Figure" "Status"
printf "%-8s %-8s\n" "------" "------"
FAIL=0
for k in "${!EXITS[@]}"; do
  status="PASS"; [[ ${EXITS[$k]} -ne 0 ]] && { status="FAIL"; FAIL=1; }
  printf "%-8s %-8s\n" "$k" "$status"
done
exit $FAIL
