#!/usr/bin/env bash
# Download the toy .laz dataset tiles from the public HuggingFace dataset
# rasmuspjohansson/KDS_laz_dataset into data/toy_laz_dataset/raw/.
# These tiles match the splits declared in
# src/datasets/toy_laz_dataset_config.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TOY_LAZ_DATA_DIR:-${HERE}/data/toy_laz_dataset/raw}"
BASE_URL="https://huggingface.co/datasets/rasmuspjohansson/KDS_laz_dataset/resolve/main"

# split/filename pairs required by the toy dataset config.
TILES=(
  "train/1km_6170_728.laz"
  "train/1km_6171_728.laz"
  "train/1km_6143_590.laz"   # 'val' split tile, stored under raw/train/
  "test/1km_6147_588.laz"
)

mkdir -p "${DATA_DIR}/train" "${DATA_DIR}/test"

for tile in "${TILES[@]}"; do
  dest="${DATA_DIR}/${tile}"
  if [[ -s "${dest}" ]]; then
    echo "[download_toy_laz_dataset] already present: ${tile}"
    continue
  fi
  echo "[download_toy_laz_dataset] downloading ${tile}"
  curl -fL --retry 3 -o "${dest}" "${BASE_URL}/${tile}"
done

echo "[download_toy_laz_dataset] done. Files under ${DATA_DIR}"
