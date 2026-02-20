# Dockerfile for Superpoint Transformer
# Image provides runtime + compiled deps; mount project at /app when running.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA architecture for older GPUs
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
# FRNN/prefix_sum (site-packages) + compiled extensions (mounted at run)
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/frnn-0.0.0-py3.8-linux-x86_64.egg:/usr/lib/python3.8/site-packages/prefix_sum-0.0.0-py3.8-linux-x86_64.egg:${PYTHONPATH}"

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

# Set working directory
WORKDIR /app

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
    pip install "laspy[laszip]" pyproj

# Install point_geometric_features
RUN pip install git+https://github.com/drprojects/point_geometric_features.git@4102aa9

# Copy only what is needed to build compiled dependencies
COPY scripts/ /app/scripts/
RUN mkdir -p /app/src/dependencies

# Clone and install FRNN
RUN rm -rf /app/src/dependencies/FRNN && \
    git clone --recursive https://github.com/lxxue/FRNN.git /app/src/dependencies/FRNN

# Install FRNN prefix_sum
WORKDIR /app/src/dependencies/FRNN/external/prefix_sum
RUN python setup.py install

# Install FRNN
WORKDIR /app/src/dependencies/FRNN
RUN python setup.py install

# Clone parallel-cut-pursuit and grid-graph
WORKDIR /app
RUN rm -rf /app/src/dependencies/parallel_cut_pursuit /app/src/dependencies/grid_graph && \
    git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git /app/src/dependencies/parallel_cut_pursuit && \
    git clone https://gitlab.com/1a7r0ch3/grid-graph.git /app/src/dependencies/grid_graph

# Compile grid_graph and parallel_cut_pursuit
RUN python scripts/setup_dependencies.py build_ext

# Copy compiled extensions to a fixed path so they work when /app is mounted
RUN mkdir -p /opt/spt-deps/bin /opt/spt-deps/wrappers && \
    cp -a /app/src/dependencies/grid_graph/python/bin/. /opt/spt-deps/bin/ && \
    cp -a /app/src/dependencies/parallel_cut_pursuit/python/bin/. /opt/spt-deps/bin/ && \
    cp -a /app/src/dependencies/parallel_cut_pursuit/python/wrappers/. /opt/spt-deps/wrappers/

# Prefer image's compiled libs over mounted src/dependencies
ENV PYTHONPATH="/opt/spt-deps/bin:/opt/spt-deps/wrappers:${PYTHONPATH}"

WORKDIR /app

CMD ["bash"]
