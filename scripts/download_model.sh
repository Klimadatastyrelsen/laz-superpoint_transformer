#!/usr/bin/env bash
# Download the pretrained vox025toy_laz SPT checkpoint from the private
# HuggingFace model repo Danish-Agency-for-Cimate-Data/KDS_spt_laz into checkpoints/.
# Needs `hf auth login` (or HF_TOKEN). Non-commercial access: rajoh@kds.dk.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="Danish-Agency-for-Cimate-Data/KDS_spt_laz"
MODEL_FILE="vox025toy_laz_best.ckpt"
DEST_DIR="${SPT_MODEL_DIR:-${HERE}/checkpoints}"
DEST="${DEST_DIR}/${MODEL_FILE}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[download_model] hf not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "[download_model] hf auth login is required (or set HF_TOKEN)." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

if [[ -s "${DEST}" ]]; then
  echo "[download_model] already present: ${DEST}"
  echo "[download_model] use with: --ckpt_path ${DEST}"
  exit 0
fi

echo "[download_model] downloading ${MODEL_FILE} from ${REPO_ID}"
hf download "${REPO_ID}" "${MODEL_FILE}" \
  --repo-type model \
  --local-dir "${DEST_DIR}"

echo "[download_model] done. Checkpoint: ${DEST}"
echo "[download_model] use with: --ckpt_path ${DEST}"
