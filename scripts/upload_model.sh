#!/usr/bin/env bash
# Upload the vox025toy_laz SPT checkpoint to the HuggingFace model repo
# rasmuspjohansson/KDS_spt_laz as vox025toy_laz_best.ckpt.
# Layout matches scripts/download_model.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCH_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ID="rasmuspjohansson/KDS_spt_laz"
MODEL_FILE="vox025toy_laz_best.ckpt"
README_TEMPLATE="${HERE}/scripts/KDS_spt_model_README.md"
CKPT_PATH="${CKPT_PATH:-/mnt/T/mnt/logs_and_models/pointcloud/vox025toy_laz_dataset/runs/2026-06-16_11-14-06/checkpoints/epoch_909.ckpt}"

resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
    return 0
  fi
  echo "[upload_model] huggingface_hub CLI not found." >&2
  echo "[upload_model] Install with: pip install huggingface_hub" >&2
  return 1
}

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  local token_file="${HF_TOKEN_FILE:-}"
  if [[ -z "${token_file}" ]]; then
    if [[ -f "${HERE}/hftoken_write.txt" ]]; then
      token_file="${HERE}/hftoken_write.txt"
    elif [[ -f "${ORCH_ROOT}/my_huggingface_token.txt" ]]; then
      token_file="${ORCH_ROOT}/my_huggingface_token.txt"
    else
      token_file="${ORCH_ROOT}/my_huggingface_token.txt"
    fi
  fi
  if [[ ! -f "${token_file}" ]]; then
    echo "[upload_model] No HF_TOKEN and token file missing: ${token_file}" >&2
    return 1
  fi
  HF_TOKEN="$(tr -d '[:space:]' < "${token_file}")"
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "[upload_model] Token file is empty: ${token_file}" >&2
    return 1
  fi
  export HF_TOKEN
}

upload_file() {
  local hf_cli="$1"
  local local_path="$2"
  local hf_path="$3"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[upload_model] would upload ${local_path} -> ${hf_path} ($(du -h "${local_path}" | cut -f1))"
    return 0
  fi

  echo "[upload_model] uploading ${local_path} -> ${hf_path} ($(du -h "${local_path}" | cut -f1))"
  "${hf_cli}" upload "${REPO_ID}" "${local_path}" "${hf_path}" --repo-type model
}

HF_CLI="$(resolve_hf_cli)"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  resolve_hf_token
fi

if [[ ! -f "${README_TEMPLATE}" ]]; then
  echo "[upload_model] README template missing: ${README_TEMPLATE}" >&2
  exit 1
fi

if [[ ! -s "${CKPT_PATH}" ]]; then
  echo "[upload_model] checkpoint missing: ${CKPT_PATH}" >&2
  exit 1
fi

upload_file "${HF_CLI}" "${CKPT_PATH}" "${MODEL_FILE}"
upload_file "${HF_CLI}" "${README_TEMPLATE}" "README.md"

echo "[upload_model] done. Model: https://huggingface.co/${REPO_ID}"
