#!/usr/bin/env bash
# Run all four toy-LAZ prediction jobs inside Docker.
# Full console output is tee'd to logs/predict_all_runs.log; each job also
# writes output/<dir>/predict_run.log via predict_many.py --run_log.
#
# Override checkpoint dir: CKPT_DIR=logs/.../checkpoints ./scripts/run_all_predictions.sh
# Override image: SPT_IMAGE=kds_spt_laz_pytorch:20260617 ./scripts/run_all_predictions.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# shellcheck source=scripts/docker_defaults.sh
source "${HERE}/scripts/docker_defaults.sh"
IMAGE="${SPT_IMAGE}"
SHM_SIZE="${SPT_SHM_SIZE:-32g}"
mkdir -p output logs
MASTER_LOG="${HERE}/logs/predict_all_runs.log"
TRAIN_LOG="logs/docker_train.log"

RUNS_GLOB="logs/vox025toy_laz_dataset/runs/*/checkpoints"
CKPT_DIR="${CKPT_DIR:-$(ls -td ${RUNS_GLOB} 2>/dev/null | head -1)}"
BEST_CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -name 'epoch_*.ckpt' 2>/dev/null | head -1)"
LAST_CKPT="${CKPT_DIR}/last.ckpt"

if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "[run_all_predictions] No checkpoint directory found under ${RUNS_GLOB}" >&2
  exit 1
fi
if [[ -z "${BEST_CKPT}" || ! -f "${BEST_CKPT}" ]]; then
  echo "[run_all_predictions] No best checkpoint (epoch_*.ckpt) in ${CKPT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${LAST_CKPT}" ]]; then
  echo "[run_all_predictions] Missing last.ckpt in ${CKPT_DIR}" >&2
  exit 1
fi

echo "[run_all_predictions] CKPT_DIR=${CKPT_DIR}"
echo "[run_all_predictions] BEST_CKPT=${BEST_CKPT}"
echo "[run_all_predictions] LAST_CKPT=${LAST_CKPT}"
echo "[run_all_predictions] IMAGE=${IMAGE}"

run_job() {
  local name="$1" ckpt="$2" input="$3" out="$4" extra_args="${5:-}"
  echo ""
  echo "======================================================================"
  echo "JOB: ${name}  $(date -Iseconds)"
  echo "======================================================================"
  docker run --gpus all --rm --shm-size="${SHM_SIZE}" \
    -v "${HERE}/data:/app/data" \
    -v "${HERE}/output:/app/output" \
    -v "${HERE}/logs:/app/logs" \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_MODE=disabled \
    "${IMAGE}" \
    bash -lc "cd /app && python predict_many.py \
      --ckpt_path ${ckpt} \
      --inputlaz ${input} \
      --output_folder ${out} \
      --get_accuracy \
      --run_log ${out}/predict_run.log \
      ${extra_args}"
}

rc=0
{
  echo "predict_all_runs started: $(date -Iseconds)"

  run_job best_train "${BEST_CKPT}" data/toy_laz_dataset/raw/train output/best_train || rc=1
  run_job best_test  "${BEST_CKPT}" data/toy_laz_dataset/raw/test  output/best_test  "--log ${TRAIN_LOG}" || rc=1
  run_job last_train "${LAST_CKPT}" data/toy_laz_dataset/raw/train output/last_train || rc=1
  run_job last_test  "${LAST_CKPT}" data/toy_laz_dataset/raw/test  output/last_test || rc=1

  echo "predict_all_runs finished: $(date -Iseconds) exit=${rc}"
  exit "${rc}"
} 2>&1 | tee -a "${MASTER_LOG}"
