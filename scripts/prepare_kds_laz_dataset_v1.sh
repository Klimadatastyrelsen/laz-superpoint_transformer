#!/usr/bin/env bash
# Check that every KDS_laz_dataset_v1 train/val/test tile is present.
#
# The split is TILES in src/datasets/kds_laz_dataset_v1_config.py. Tiles are
# read from KDS_LAZ_SRC in place (this NAS does not support symlinks).
# Processed files are written under ${KDS_LAZ_DATA_DIR}/KDS_laz_dataset_v1/processed/.
#
# Usage:
#   ./scripts/prepare_kds_laz_dataset_v1.sh
#   KDS_LAZ_SRC=/path/to/laz ./scripts/prepare_kds_laz_dataset_v1.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

export KDS_LAZ_SRC="${KDS_LAZ_SRC:-/mnt/T/mnt/trainingdata/lidar/KDS_laz_dataset_v1}"
export KDS_LAZ_DATA_DIR="${KDS_LAZ_DATA_DIR:-/mnt/T/mnt/trainingdata/lidar}"
SPT_ENV="${SPT_ENV:-spt}"
CONDA_DIR="${CONDA_DIR:-${HOME}/miniforge3}"

if [[ ! -d "${KDS_LAZ_SRC}" ]]; then
  echo "Source folder not found: ${KDS_LAZ_SRC}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${CONDA_DIR}/envs/${SPT_ENV}/bin/python" ]]; then
  PY="${CONDA_DIR}/envs/${SPT_ENV}/bin/python"
else
  PY="$(command -v python3)"
fi

TILE_LIST="$(mktemp)"
trap 'rm -f "${TILE_LIST}"' EXIT
"${PY}" - > "${TILE_LIST}" << 'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kds_laz_dataset_v1_config", "src/datasets/kds_laz_dataset_v1_config.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for split, names in mod.TILES.items():
    for name in names:
        print(f"{split}\t{name}")
PY
mapfile -t ROWS < "${TILE_LIST}"

if [[ "${#ROWS[@]}" -ne 48 ]]; then
  echo "Expected 48 tiles from the config, got ${#ROWS[@]}" >&2
  exit 1
fi

missing=0
declare -A COUNT=([train]=0 [val]=0 [test]=0)
for row in "${ROWS[@]}"; do
  split="${row%%$'\t'*}"
  name="${row#*$'\t'}"
  src="${KDS_LAZ_SRC}/${name}.laz"
  if [[ ! -f "${src}" ]]; then
    echo "Missing source tile: ${src}" >&2
    missing=1
    continue
  fi
  COUNT["${split}"]=$((COUNT["${split}"] + 1))
done

if [[ "${missing}" -ne 0 ]]; then
  echo "One or more tiles are missing from ${KDS_LAZ_SRC}" >&2
  exit 1
fi

mkdir -p "${KDS_LAZ_DATA_DIR}/KDS_laz_dataset_v1"
echo "All 48 tiles are in ${KDS_LAZ_SRC}"
echo "  train ${COUNT[train]}"
echo "  val   ${COUNT[val]}"
echo "  test  ${COUNT[test]}"
echo "Processed cache: ${KDS_LAZ_DATA_DIR}/KDS_laz_dataset_v1/processed/"
