#!/usr/bin/env bash
# check_verification_logs.sh
# Inspect the verification log for the keywords that must be present when
# everything works, and for failure markers that must be absent. Exits non-zero
# (and prints a summary) if the run was not clean.

set -uo pipefail

ORCH_DIR="${LAZ_ORCH_DIR:-/home/rajoh/projects/ML_verify_all_ML_repos_work}"
LOG_FILE="${1:-${ORCH_DIR}/logs/laz_logs.txt}"

# Whether the .laz training verification keywords are also required.
RUN_LAZ_VERIFY="${RUN_LAZ_VERIFY:-0}"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "CHECK_FAILED: log file not found: ${LOG_FILE}"
  exit 1
fi

# Keywords that MUST appear for a successful run.
required=(
  "DOCKER_BUILD_OK"
  "SMOKE_BEGIN"
  "SMOKE_TORCH_OK"
  "SMOKE_TORCH_GEOMETRIC_OK"
  "SMOKE_TORCH_SCATTER_OK"
  "SMOKE_PGEOF_OK"
  "SMOKE_GRID_GRAPH_OK"
  "SMOKE_CUT_PURSUIT_OK"
  "SMOKE_FRNN_OK"
  "SMOKE_CUDA_AVAILABLE_OK"
  "SMOKE_CUDA_MATMUL_OK"
  "SMOKE_ALL_OK"
  "OVERALL_RESULT: PASS"
)

if [[ "${RUN_LAZ_VERIFY}" == "1" ]]; then
  required+=("LAZ_TRAIN_OK")
fi

# Markers that MUST NOT appear.
forbidden=(
  "DOCKER_BUILD_FAILED"
  "DOCKER_UNAVAILABLE"
  "SMOKE_FAILED"
  "_FAIL"
  "LAZ_TRAIN_FAILED"
  "OVERALL_RESULT: FAIL"
  "Traceback (most recent call last)"
)

missing=()
for kw in "${required[@]}"; do
  if ! grep -qF -- "${kw}" "${LOG_FILE}"; then
    missing+=("${kw}")
  fi
done

present_bad=()
for kw in "${forbidden[@]}"; do
  if grep -qF -- "${kw}" "${LOG_FILE}"; then
    present_bad+=("${kw}")
  fi
done

echo "=== check_verification_logs: ${LOG_FILE} ==="
if (( ${#missing[@]} == 0 )) && (( ${#present_bad[@]} == 0 )); then
  echo "CHECK_OK: all required keywords present, no failure markers found"
  exit 0
fi

if (( ${#missing[@]} > 0 )); then
  echo "CHECK_FAILED: missing required keywords:"
  printf '  - %s\n' "${missing[@]}"
fi
if (( ${#present_bad[@]} > 0 )); then
  echo "CHECK_FAILED: found failure markers:"
  printf '  - %s\n' "${present_bad[@]}"
fi
exit 1
