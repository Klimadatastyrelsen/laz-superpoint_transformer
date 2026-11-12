#!/usr/bin/env bash
# Download the pretrained vox025toy_laz SPT checkpoint from the private
# HuggingFace model repo Danish-Agency-for-Cimate-Data/KDS_spt_laz into checkpoints/.
# Needs a token: hf auth login. Non-commercial access: rajoh@kds.dk.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="Danish-Agency-for-Cimate-Data/KDS_spt_laz"
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
auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  auth=(-H "Authorization: Bearer ${HF_TOKEN}")
fi
curl -fL --retry 3 "${auth[@]}" -o "${DEST}" "${BASE_URL}/${MODEL_FILE}"

echo "[download_model] done. Checkpoint: ${DEST}"
echo "[download_model] use with: --ckpt_path ${DEST}"
