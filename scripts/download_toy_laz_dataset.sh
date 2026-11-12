#!/usr/bin/env bash
# Download the toy .laz dataset tiles from the private HuggingFace dataset
# Danish-Agency-for-Cimate-Data/KDS_laz_dataset into data/toy_laz_dataset/raw/.
# Needs `hf auth login` (or HF_TOKEN). Non-commercial access: rajoh@kds.dk.
# These tiles match the splits declared in
# src/datasets/toy_laz_dataset_config.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TOY_LAZ_DATA_DIR:-${HERE}/data/toy_laz_dataset/raw}"
REPO_ID="Danish-Agency-for-Cimate-Data/KDS_laz_dataset"

TILES=(
  "train/1km_6170_728.laz"
  "train/1km_6171_728.laz"
  "train/1km_6143_589.laz"   # val split tile, stored under raw/train/
  "train/1km_6172_728.laz"
  "train/1km_6173_728.laz"
  "test/1km_6147_588.laz"
  "test/1km_6143_590.laz"
)

if ! command -v hf >/dev/null 2>&1; then
  echo "[download_toy_laz_dataset] hf not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "[download_toy_laz_dataset] hf auth login is required (or set HF_TOKEN)." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}/train" "${DATA_DIR}/test"
echo "[download_toy_laz_dataset] downloading ${#TILES[@]} tiles from ${REPO_ID}"
hf download "${REPO_ID}" \
  --repo-type dataset \
  --local-dir "${DATA_DIR}" \
  "${TILES[@]}"

echo "[download_toy_laz_dataset] done. Files under ${DATA_DIR}"
