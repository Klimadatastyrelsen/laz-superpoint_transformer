#!/usr/bin/env bash
# Upload the toy .laz dataset tiles to the HuggingFace dataset
# rasmuspjohansson/KDS_laz_dataset from data/toy_laz_dataset/raw/.
# Layout matches src/datasets/toy_laz_dataset_config.py and
# scripts/download_toy_laz_dataset.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCH_ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_DIR="${TOY_LAZ_DATA_DIR:-${HERE}/data/toy_laz_dataset/raw}"
REPO_ID="rasmuspjohansson/KDS_laz_dataset"
README_TEMPLATE="${HERE}/scripts/KDS_laz_dataset_README.md"

TILES=(
  "train/1km_6170_728.laz"
  "train/1km_6171_728.laz"
  "train/1km_6143_589.laz"
  "train/1km_6172_728.laz"
  "train/1km_6173_728.laz"
  "test/1km_6147_588.laz"
  "test/1km_6143_590.laz"
)

STALE_HF_PATHS=(
  "train/1km_6143_590.laz"
)

resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
    return 0
  fi
  echo "[upload_toy_laz_dataset] huggingface_hub CLI not found." >&2
  echo "[upload_toy_laz_dataset] Install with: pip install huggingface_hub" >&2
  return 1
}

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  local token_file="${HF_TOKEN_FILE:-${ORCH_ROOT}/my_huggingface_token.txt}"
  if [[ ! -f "${token_file}" ]]; then
    echo "[upload_toy_laz_dataset] No HF_TOKEN and token file missing: ${token_file}" >&2
    return 1
  fi
  HF_TOKEN="$(tr -d '[:space:]' < "${token_file}")"
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "[upload_toy_laz_dataset] Token file is empty: ${token_file}" >&2
    return 1
  fi
  export HF_TOKEN
}

delete_stale_hf_path() {
  local hf_cli="$1"
  local stale_path="$2"

  echo "[upload_toy_laz_dataset] removing stale HF path (if present): ${stale_path}"
  if [[ "${hf_cli}" == hf ]]; then
    "${hf_cli}" repos delete-files "${REPO_ID}" "${stale_path}" \
      --repo-type dataset \
      --commit-message "Remove stale ${stale_path}" \
      2>/dev/null || true
    return 0
  fi

  "${hf_cli}" delete-file "${REPO_ID}" "${stale_path}" \
    --repo-type dataset \
    2>/dev/null || true
}

upload_file() {
  local hf_cli="$1"
  local local_path="$2"
  local hf_path="$3"

  echo "[upload_toy_laz_dataset] uploading ${local_path} -> ${hf_path} ($(du -h "${local_path}" | cut -f1))"
  "${hf_cli}" upload "${REPO_ID}" "${local_path}" "${hf_path}" --repo-type dataset
}

HF_CLI="$(resolve_hf_cli)"
resolve_hf_token

if [[ ! -f "${README_TEMPLATE}" ]]; then
  echo "[upload_toy_laz_dataset] README template missing: ${README_TEMPLATE}" >&2
  exit 1
fi

for tile in "${TILES[@]}"; do
  local_path="${DATA_DIR}/${tile}"
  if [[ ! -s "${local_path}" ]]; then
    echo "[upload_toy_laz_dataset] missing local file: ${local_path}" >&2
    exit 1
  fi
done

for stale_path in "${STALE_HF_PATHS[@]}"; do
  delete_stale_hf_path "${HF_CLI}" "${stale_path}"
done

for tile in "${TILES[@]}"; do
  upload_file "${HF_CLI}" "${DATA_DIR}/${tile}" "${tile}"
done

echo "[upload_toy_laz_dataset] uploading dataset card README.md"
upload_file "${HF_CLI}" "${README_TEMPLATE}" "README.md"

echo "[upload_toy_laz_dataset] done. Dataset: https://huggingface.co/datasets/${REPO_ID}"
