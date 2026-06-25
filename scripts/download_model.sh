#!/usr/bin/env bash
# Download the pretrained vox025toy_laz SPT checkpoint from the public
# HuggingFace model repo rasmuspjohansson/KDS_spt_laz into checkpoints/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="rasmuspjohansson/KDS_spt_laz"
MODEL_FILE="vox025toy_laz_best.ckpt"
DEST_DIR="${SPT_MODEL_DIR:-${HERE}/checkpoints}"
DEST="${DEST_DIR}/${MODEL_FILE}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${DEST_DIR}"

if [[ -s "${DEST}" ]]; then
  echo "[download_model] already present: ${DEST}"
  echo "[download_model] use with: --ckpt_path ${DEST}"
  exit 0
fi

echo "[download_model] downloading ${MODEL_FILE} from ${REPO_ID}"
curl -fL --retry 3 -o "${DEST}" "${BASE_URL}/${MODEL_FILE}"

echo "[download_model] done. Checkpoint: ${DEST}"
echo "[download_model] use with: --ckpt_path ${DEST}"
