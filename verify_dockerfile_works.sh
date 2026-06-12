#!/usr/bin/env bash
# verify_dockerfile_works.sh
# Build the Superpoint Transformer Docker image and run the smoke test (and,
# once present, the .laz support test) inside the container with GPU access.
# Output is logged to <repo>/logs/laz_logs.txt (override the directory with
# LAZ_LOG_DIR); that directory is mounted so the container writes on the host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory that receives the log file. Defaults to the repo's own logs/ dir.
LOG_DIR="${LAZ_LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_DIR}/laz_logs.txt"

IMAGE_TAG="${SPT_IMAGE:-spt_merged:latest}"
SKIP_BUILD="${SPT_SKIP_BUILD:-0}"
SHM_SIZE="${SPT_SHM_SIZE:-32g}"
# Set RUN_LAZ_VERIFY=1 to also run the .laz training verification (phase 2+).
RUN_LAZ_VERIFY="${RUN_LAZ_VERIFY:-0}"

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

log() { echo "[verify_dockerfile_works] $*" >&2; }
append() { echo "$*" >> "${LOG_FILE}"; }

append "verify_dockerfile_works log"
append "Generated: $(date -Iseconds)"
append "Image: ${IMAGE_TAG}"
append "Log dir mount: ${LOG_DIR}"
append ""

if ! command -v docker >/dev/null 2>&1; then
  append "DOCKER_UNAVAILABLE: docker command not found"
  log "docker not found"
  exit 1
fi

rc=0

if [[ "${SKIP_BUILD}" != "1" ]]; then
  append "=== docker build -t ${IMAGE_TAG} ${SCRIPT_DIR} ==="
  if docker build -t "${IMAGE_TAG}" "${SCRIPT_DIR}" >> "${LOG_FILE}" 2>&1; then
    append "DOCKER_BUILD_OK"
  else
    append "DOCKER_BUILD_FAILED"
    log "docker build failed; see ${LOG_FILE}"
    exit 1
  fi
else
  append "=== SPT_SKIP_BUILD=1: using existing image ${IMAGE_TAG} ==="
fi

run_in_container() {
  # $1: human label, $2: command to run inside the container (cwd /app).
  local label="$1" cmd="$2"
  append ""
  append "=== ${label} ==="
  if docker run --gpus all --rm --shm-size="${SHM_SIZE}" \
      -v "${LOG_DIR}:${LOG_DIR}" \
      -e PYTHONUNBUFFERED=1 \
      "${IMAGE_TAG}" \
      bash -lc "set -o pipefail; ${cmd} 2>&1 | tee -a '${LOG_FILE}'"; then
    return 0
  fi
  return 1
}

# Phase 1: environment smoke test.
run_in_container "smoke test (verify_everything_works_smoke.py)" \
  "cd /app && python verify_everything_works_smoke.py" || rc=1

# Phase 2+: short .laz training run (only when the script exists and is enabled).
if [[ "${RUN_LAZ_VERIFY}" == "1" ]]; then
  run_in_container "laz support test (verify_laz_support_works.py)" \
    "cd /app && python verify_laz_support_works.py" || rc=1
fi

append ""
if (( rc == 0 )); then
  append "OVERALL_RESULT: PASS"
else
  append "OVERALL_RESULT: FAIL"
fi

log "Done. Log: ${LOG_FILE}"
exit "${rc}"
