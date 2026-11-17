#!/usr/bin/env bash
# Create or update the conda env used to train SPT on this Blackwell GPU.
#
# The published Docker image is PyTorch 2.2 + CUDA 11.8 (compute 7.5) and
# does not run on compute capability 12.0. This script pins a CUDA 12.8
# build instead. Re-running it reinstalls those pins.
#
# Usage:
#   ./scripts/install_spt_blackwell.sh
#   SPT_ENV=spt ./scripts/install_spt_blackwell.sh
#
# Overrides: SPT_ENV, CONDA_DIR, SPT_PYTHON_VERSION, SPT_TORCH_VERSION,
# SPT_CUDA_TAG, TORCH_CUDA_ARCH_LIST.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

SPT_ENV="${SPT_ENV:-spt}"
CONDA_DIR="${CONDA_DIR:-${HOME}/miniforge3}"
PYTHON_VERSION="${SPT_PYTHON_VERSION:-3.11}"
TORCH_VERSION="${SPT_TORCH_VERSION:-2.11.0}"
CUDA_TAG="${SPT_CUDA_TAG:-cu128}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

if [[ ! -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  echo "Could not find ${CONDA_DIR}/etc/profile.d/conda.sh. Set CONDA_DIR." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${SPT_ENV}"; then
  echo "Creating conda env ${SPT_ENV} (python ${PYTHON_VERSION})"
  conda create --name "${SPT_ENV}" "python=${PYTHON_VERSION}" -y
else
  echo "Conda env ${SPT_ENV} already exists; reinstalling pinned packages"
fi
conda activate "${SPT_ENV}"

python -m pip install --upgrade pip
python -m pip install \
  "torch==${TORCH_VERSION}" torchvision \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -m pip install "torchmetrics==0.11.4"

PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
echo "Installing PyG extensions from ${PYG_INDEX}"
if ! python -m pip install pyg_lib torch_scatter torch_cluster -f "${PYG_INDEX}"; then
  echo "PyG wheel index failed; installing torch_scatter and torch_cluster from PyPI" >&2
  python -m pip install torch_scatter torch_cluster
fi
# 2.3 keeps Data.keys as a list. 2.4+ makes it a method and breaks this repo.
python -m pip install "torch_geometric==2.3.0"

python -m pip install \
  matplotlib \
  "plyfile==1.0.3" \
  "pandas==2.2.3" \
  h5py \
  colorhash \
  seaborn \
  numba \
  pytorch-lightning \
  pyrootutils \
  hydra-core \
  hydra-colorlog \
  hydra-submitit-launcher \
  "rich<=14.0" \
  wandb \
  open3d \
  gdown \
  ipyfilechooser \
  torch-ransac3d \
  scikit-learn \
  GitPython \
  pgeof \
  pycut-pursuit \
  pygrid-graph \
  torch-graph-components \
  "laspy[lazrs]" \
  "numpy==1.26.4"

# Keep the NumPy 1.x ABI these extension wheels were built with. A later
# package must not upgrade NumPy to 2.x.
python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "plyfile==1.0.3" \
  "pycut-pursuit==0.1.4" \
  "pgeof==0.3.4" \
  "pygrid-graph==0.0.4"

# FRNN's CUDA extension imports torch at setup time and needs nvcc.
# The driver alone is not enough; the toolkit is installed into this env.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Installing cuda-nvcc into ${SPT_ENV}"
  conda install -y -c nvidia "cuda-nvcc=12.8" "cuda-cudart-dev=12.8"
fi
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# PyTorch's pip wheels ship CUDA headers under site-packages/nvidia, which
# nvcc and the host compiler do not search. Link them into the env include
# dir (already on both compilers' search paths) before building FRNN.
NVIDIA_ROOT="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia"
if [[ -d "${NVIDIA_ROOT}" ]]; then
  mkdir -p "${CONDA_PREFIX}/include" "${CONDA_PREFIX}/lib"
  for inc in "${NVIDIA_ROOT}"/*/include; do
    [[ -d "${inc}" ]] || continue
    ln -sfn "${inc}"/* "${CONDA_PREFIX}/include/"
  done
  for lib in "${NVIDIA_ROOT}"/*/lib; do
    [[ -d "${lib}" ]] || continue
    ln -sfn "${lib}"/* "${CONDA_PREFIX}/lib/"
  done
fi

mkdir -p src/dependencies
if [[ ! -d src/dependencies/FRNN/.git ]]; then
  git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
else
  git -C src/dependencies/FRNN submodule update --init --recursive
fi
# Both packages import torch from setup.py, so build isolation must be off.
python -m pip install --no-build-isolation src/dependencies/FRNN/external/prefix_sum
python -m pip install --no-build-isolation src/dependencies/FRNN

python - << 'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this env")
x = torch.randn(64, 64, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul_ok", float(y.reshape(-1)[0]))
print("capability", tuple(torch.cuda.get_device_capability()))
import laspy  # noqa: F401
import pytorch_lightning  # noqa: F401
import torch_geometric  # noqa: F401
import hydra  # noqa: F401
from src.dependencies.FRNN import frnn  # noqa: F401
print("SPT_ENV_OK")
PY

echo "Env ${SPT_ENV} is ready. Activate it with: conda activate ${SPT_ENV}"
