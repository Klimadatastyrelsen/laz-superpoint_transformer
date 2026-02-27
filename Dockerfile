# Dockerfile for Superpoint Transformer
# Image provides runtime + compiled deps at /dependencies; mount project at /app when running.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA architecture for older GPUs
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Install system dependencies and add deadsnakes PPA for Python 3.8
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

# Set python3.8 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Upgrade pip
RUN python -m pip install --upgrade pip

# Force reinstall conflicting system packages
RUN pip install --ignore-installed blinker

# Install Python dependencies
RUN pip install matplotlib && \
    pip install plotly==5.9.0 && \
    pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash && \
    pip install "notebook>=5.3" "ipywidgets>=7.5" && \
    pip install ipykernel

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install torchmetrics and PyTorch Geometric dependencies
RUN pip install torchmetrics==0.11.4 && \
    pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html && \
    pip install torch_geometric==2.3.0

# Install remaining pip dependencies
RUN pip install plyfile h5py colorhash seaborn numba && \
    pip install pytorch-lightning pyrootutils && \
    pip install hydra-core --upgrade && \
    pip install hydra-colorlog hydra-submitit-launcher && \
    pip install "rich<=14.0" && \
    pip install torch_tb_profiler wandb open3d gdown && \
    pip install ipyfilechooser && \
    pip install "laspy[laszip]" pyproj && \
    pip install huggingface_hub

# Install point_geometric_features
RUN pip install git+https://github.com/drprojects/point_geometric_features.git@4102aa9

# Build compiled dependencies under /dependencies (nothing from repo under /app in image)
RUN mkdir -p /dependencies /build

# Copy only scripts needed to build dependencies
COPY scripts/ /build/scripts/

# Clone and install FRNN under /dependencies
RUN git clone --recursive https://github.com/lxxue/FRNN.git /dependencies/FRNN

# Install FRNN prefix_sum
WORKDIR /dependencies/FRNN/external/prefix_sum
RUN python setup.py install

# Install FRNN (into site-packages)
WORKDIR /dependencies/FRNN
RUN python setup.py install

# Clone grid_graph and parallel_cut_pursuit under /dependencies
RUN git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git /dependencies/parallel_cut_pursuit && \
    git clone https://gitlab.com/1a7r0ch3/grid-graph.git /dependencies/grid_graph

# Compile grid_graph and parallel_cut_pursuit using DEPENDENCIES_DIR so script finds /dependencies
WORKDIR /build
ENV DEPENDENCIES_DIR=/dependencies
RUN python /build/scripts/setup_dependencies.py build_ext

# Runtime: repo is mounted at /app; deps live at /dependencies
# FRNN and prefix_sum are installed as eggs; add egg paths explicitly (PYTHONPATH does not trigger .pth processing)
ENV SPT_DEPS_DIR=/dependencies
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/prefix_sum-0.0.0-py3.8-linux-x86_64.egg:/usr/lib/python3.8/site-packages/frnn-0.0.0-py3.8-linux-x86_64.egg:/dependencies/grid_graph/python/bin:/dependencies/parallel_cut_pursuit/python/wrappers:${PYTHONPATH}"

WORKDIR /app

CMD ["bash"]
