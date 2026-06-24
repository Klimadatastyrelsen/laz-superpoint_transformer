#!/usr/bin/env bash
# Run predict_many.py inside Docker with correct volume mounts.
#
# Interactive by default: live output on screen plus logs on disk.
#   - Host session log:  <output>/docker_session.log  (via tee)
#   - In-container log:  <output>/predict_run.log      (via predict_many --run_log)
#
# Usage:
#   ./scripts/run_predict_docker.sh \
#     --ckpt /host/path/to/last.ckpt \
#     --input /host/path/to/laz_dir_or_file \
#     --output /host/path/to/results \
#     [--get_accuracy] [--log /host/path/to/docker_train.log]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SPT_IMAGE:-spt_merged:latest}"
SHM_SIZE="${SPT_SHM_SIZE:-32g}"
CONFIG="${PREDICT_CONFIG:-experiment=semantic/vox025toy_laz_dataset}"

CKPT=""
INPUT=""
OUTPUT=""
TRAINING_LOG=""
SESSION_LOG=""
GET_ACCURACY=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --log) TRAINING_LOG="$2"; shift 2 ;;
    --session_log) SESSION_LOG="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --get_accuracy) GET_ACCURACY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "${CKPT}" ]] || { echo "Error: --ckpt is required" >&2; usage 1; }
[[ -n "${INPUT}" ]] || { echo "Error: --input is required" >&2; usage 1; }
[[ -n "${OUTPUT}" ]] || { echo "Error: --output is required" >&2; usage 1; }

CKPT="$(readlink -f "${CKPT}")"
INPUT="$(readlink -f "${INPUT}")"
# OUTPUT may not exist yet; readlink -f fails on missing paths (set -e exits silently).
mkdir -p "${OUTPUT}"
OUTPUT="$(readlink -f "${OUTPUT}")"

if [[ -n "${TRAINING_LOG}" ]]; then
  TRAINING_LOG="$(readlink -f "${TRAINING_LOG}")"
fi

SESSION_LOG="${SESSION_LOG:-${OUTPUT}/docker_session.log}"
touch "${SESSION_LOG}"

CKPT_DIR="$(dirname "${CKPT}")"
CKPT_BASE="$(basename "${CKPT}")"
CONTAINER_CKPT="/app/checkpoints/${CKPT_BASE}"

if [[ -f "${INPUT}" ]]; then
  INPUT_DIR="$(dirname "${INPUT}")"
  INPUT_BASE="$(basename "${INPUT}")"
  CONTAINER_INPUT="/app/input/${INPUT_BASE}"
elif [[ -d "${INPUT}" ]]; then
  INPUT_DIR="${INPUT}"
  CONTAINER_INPUT="/app/input"
else
  echo "Error: --input must be a .laz/.las file or directory: ${INPUT}" >&2
  exit 1
fi

CONTAINER_OUTPUT="/app/output"
CONTAINER_RUN_LOG="${CONTAINER_OUTPUT}/predict_run.log"

MOUNTS=(
  -v "${CKPT_DIR}:/app/checkpoints:ro"
  -v "${INPUT_DIR}:/app/input:ro"
  -v "${OUTPUT}:${CONTAINER_OUTPUT}"
)

EXTRA_ARGS=()
if [[ "${GET_ACCURACY}" == "1" ]]; then
  EXTRA_ARGS+=(--get_accuracy)
fi
if [[ -n "${TRAINING_LOG}" ]]; then
  LOG_DIR="$(dirname "${TRAINING_LOG}")"
  LOG_BASE="$(basename "${TRAINING_LOG}")"
  MOUNTS+=(-v "${LOG_DIR}:/app/logs:ro")
  EXTRA_ARGS+=(--log "/app/logs/${LOG_BASE}")
fi

DOCKER_IT=()
if [[ -t 1 ]]; then
  DOCKER_IT=(-it)
fi

PREDICT_CMD="cd /app && python predict_many.py \
  --ckpt_path ${CONTAINER_CKPT} \
  --inputlaz ${CONTAINER_INPUT} \
  --output_folder ${CONTAINER_OUTPUT} \
  --config ${CONFIG} \
  --run_log ${CONTAINER_RUN_LOG}"
for arg in "${EXTRA_ARGS[@]}"; do
  PREDICT_CMD+=" ${arg}"
done

echo "======================================================================"
echo "run_predict_docker.sh  $(date -Iseconds)"
echo "======================================================================"
echo "Image:        ${IMAGE}"
echo "Checkpoint:   ${CKPT}"
echo "Input:        ${INPUT}"
echo "Output:       ${OUTPUT}"
echo "Session log:  ${SESSION_LOG}"
echo "Run log:      ${OUTPUT}/predict_run.log"
echo "Get accuracy: ${GET_ACCURACY}"
echo "Training log: ${TRAINING_LOG:-<none>}"
echo "======================================================================"

{
  docker run --gpus all --rm "${DOCKER_IT[@]}" --shm-size="${SHM_SIZE}" \
    "${MOUNTS[@]}" \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_MODE=disabled \
    "${IMAGE}" \
    bash -lc "${PREDICT_CMD}"
} 2>&1 | tee -a "${SESSION_LOG}"

exit "${PIPESTATUS[0]}"
