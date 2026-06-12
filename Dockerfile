# Dockerfile for Superpoint Transformer (self-contained, no code mount required).
# Mirrors install.sh: Python 3.8, PyTorch 2.2.0 + CUDA 11.8, PyG, FRNN (built
# from source), and the pgeof / cut-pursuit / grid-graph pip packages.
# TORCH_CUDA_ARCH_LIST is set for the Quadro RTX 8000 (compute 7.5).

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV PYTHONUNBUFFERED=1

# System dependencies + Python 3.8 from deadsnakes.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    gcc-11 \
    g++-11 \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
    python -m pip install --upgrade pip

WORKDIR /app

# Some base images ship a system 'blinker' that pip refuses to upgrade.
RUN pip install --ignore-installed blinker

# Notebook / plotting stack (matches install.sh).
RUN pip install matplotlib && \
    pip install plotly==5.9.0 && \
    pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash && \
    pip install "notebook>=5.3" "ipywidgets>=7.5" && \
    pip install ipykernel

# PyTorch 2.2.0 + CUDA 11.8.
RUN pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# torchmetrics + PyTorch Geometric.
RUN pip install torchmetrics==0.11.4 && \
    pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html && \
    pip install torch_geometric==2.3.0

# Remaining SPT dependencies (pgeof / cut-pursuit / grid-graph now ship as
# pip packages, so no source build is needed for them).
RUN pip install plyfile h5py colorhash seaborn numba && \
    pip install pytorch-lightning pyrootutils && \
    pip install hydra-core --upgrade && \
    pip install hydra-colorlog hydra-submitit-launcher && \
    pip install "rich<=14.0" && \
    pip install torch_tb_profiler wandb open3d gdown && \
    pip install ipyfilechooser && \
    pip install torch-ransac3d pgeof pycut-pursuit pygrid-graph torch-graph-components && \
    pip install "laspy[laszip]" pyproj

# Build FRNN (and its prefix_sum helper) from source. The project imports it as
# `from src.dependencies.FRNN import frnn`, so it must live under the repo tree.
# Cloning before the final COPY keeps this (slow) build layer cached across code
# changes; COPY merges the repo on top without removing this directory.
RUN mkdir -p /app/src/dependencies && \
    git clone --recursive https://github.com/lxxue/FRNN.git /app/src/dependencies/FRNN && \
    cd /app/src/dependencies/FRNN/external/prefix_sum && pip install . && \
    cd /app/src/dependencies/FRNN && pip install .

# Bake the project in so the image is self-contained (works on HPC without
# mounting code). Logs/data are still meant to be mounted at runtime.
COPY . /app

CMD ["bash"]
