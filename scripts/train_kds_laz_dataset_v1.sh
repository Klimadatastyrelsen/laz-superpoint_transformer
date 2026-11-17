#!/usr/bin/env bash
# Train, smoke-test, or resume the SPT model on KDS_laz_dataset_v1.
#
# Usage:
#   ./scripts/train_kds_laz_dataset_v1.sh smoke
#   ./scripts/train_kds_laz_dataset_v1.sh
#   ./scripts/train_kds_laz_dataset_v1.sh resume logs/vox025kds_laz_dataset_v1/runs/<time>/checkpoints/last.ckpt
#
# smoke uses the mini split, a coarser voxel, and two batches. It does not
# reuse the full-training preprocess cache.
#
# Overrides: SPT_ENV, CONDA_DIR, KDS_LAZ_SRC, KDS_LAZ_DATA_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

SPT_ENV="${SPT_ENV:-spt}"
CONDA_DIR="${CONDA_DIR:-${HOME}/miniforge3}"
export KDS_LAZ_SRC="${KDS_LAZ_SRC:-/mnt/T/mnt/trainingdata/lidar/KDS_laz_dataset_v1}"
export KDS_LAZ_DATA_DIR="${KDS_LAZ_DATA_DIR:-/mnt/T/mnt/trainingdata/lidar}"

if [[ ! -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  echo "Could not find ${CONDA_DIR}/etc/profile.d/conda.sh. Set CONDA_DIR." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${SPT_ENV}"
export PYTHONUNBUFFERED=1

"${HERE}/scripts/prepare_kds_laz_dataset_v1.sh"

MODE="${1:-train}"
mkdir -p "${HERE}/logs"

# The .h5 cache lives on CIFS. Several DataLoader workers reading it at
# once stall in uninterruptible sleep, so the full run uses one worker.
COMMON=(
  src/train.py
  experiment=semantic/vox025kds_laz_dataset_v1
  logger=csv
  datamodule.dataloader.num_workers=0
  datamodule.dataloader.persistent_workers=False
)

case "${MODE}" in
  smoke)
    python "${COMMON[@]}" \
      datamodule.mini=True \
      datamodule.voxel=0.5 \
      datamodule.xy_tiling=2 \
      trainer.max_epochs=1 \
      +trainer.limit_train_batches=2 \
      +trainer.limit_val_batches=2 \
      +trainer.num_sanity_val_steps=0 \
      test=False \
      optimized_metric=null \
      seed=42
    ;;
  train)
    python "${COMMON[@]}" 2>&1 | tee -a "${HERE}/logs/kds_laz_dataset_v1_train.log"
    ;;
  resume)
    CKPT="${2:-}"
    if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
      echo "Usage: $0 resume /path/to/last.ckpt" >&2
      exit 1
    fi
    python "${COMMON[@]}" "ckpt_path=${CKPT}" 2>&1 | tee -a "${HERE}/logs/kds_laz_dataset_v1_train.log"
    ;;
  *)
    echo "Usage: $0 [smoke|train|resume <ckpt>]" >&2
    exit 1
    ;;
esac
