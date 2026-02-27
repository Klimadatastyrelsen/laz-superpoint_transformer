# laz-superpoint_transformer

This project is built on [superpoint_transformer](https://github.com/drprojects/superpoint_transformer) with added functionality for handling **.las / .laz** files directly.
It is used by the danish agency of climate data for processing lidar point clouds
Please have a look at the original project in order to learn more.

## 💻  Environment requirements
This project was tested with:
- Linux OS
- **64G** RAM
- NVIDIA Quadro RTX 8000  (49 Gig)

- CUDA 12.1

<br>

## 🏗  Installation
### DOCKER
You can use the prebuilt image from Docker Hub (no need to build yourself):

```bash
docker pull rasmuspjohansson/kds_spt_laz_pytorch:latest
```

Or pull a specific version by date, e.g.:

```bash
docker pull rasmuspjohansson/kds_spt_laz_pytorch:20260226
```

**Run with the project mounted** so you can edit code, configs, and data on the host:

```bash
# From the repo root; /app in the container is your local repo
docker run --gpus all --rm --shm-size=80g -it \
    -v "$(pwd)":/app \
    rasmuspjohansson/kds_spt_laz_pytorch:latest \
    bash -c "python src/train.py experiment=semantic/vox025toy_laz_dataset.yaml logger=csv; bash"
```

Logs and data live under the mounted repo (`./logs`, `./data`) unless you add extra mounts, e.g. to log elsewhere:

```bash
docker run --gpus all --rm --shm-size=80g -it \
    -v "$(pwd)":/app \
    -v /mnt/T/mnt/logs_and_models/pointcloud:/app/logs \
    rasmuspjohansson/kds_spt_laz_pytorch:latest \
    bash -c "python src/train.py experiment=semantic/vox025toy_laz_dataset.yaml logger=csv; bash"
```

**Alternatively**, build the image yourself (installs libraries and compiled extensions; no project code is baked in):

```bash
docker build -t spt-laz:latest .
```

Then use `spt-laz:latest` instead of `rasmuspjohansson/kds_spt_laz_pytorch:latest` in the `docker run` commands above.

### Data: toy LAZ dataset
The toy LAZ dataset (`.laz` files under `data/toy_laz_dataset/raw/`) is not in the repo. Download it from Hugging Face so that training works:

```bash
# From the repo root. Requires: pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='rasmuspjohansson/KDS_laz_dataset',
    repo_type='dataset',
    local_dir='data/toy_laz_dataset/raw',
)
print('Downloaded to data/toy_laz_dataset/raw (train/ and test/).')
"
```

This creates `data/toy_laz_dataset/raw/train/` and `data/toy_laz_dataset/raw/test/` with the `.laz` files. Then run the Docker training command above.

## Direct .sh install

install_CUDA_ARCH_7_5.sh is setup for usage on GPU: Quadro RTX 8000
install.sh should work on a more modern GPU but have not been tested for this project. (see [superpoint_transformer](https://github.com/drprojects/superpoint_transformer) for more info on installation)


### Installation part 1.
In order for the installation to work you need cuda 12.1 and gcc-11
If there are problems with instaling cuda 12.1 , make sure to purge cuda from the system
### purge cuda and nvidia driver (only if needed ) 
sudo apt-get --purge remove 'cuda*'

sudo apt-get --purge remove 'libcublas*' 'libnccl*' 'libnpp*' 'libcufft*' 'libcurand*' 'libcusolver*' 'libcusparse*'

sudo apt-get autoremove

sudo apt-get autoclean

sudo rm -rf /usr/local/cuda*

### install nvidia driver (only if needed)

### install cuda 12.1 
Follow instructions from link below with one change . 
replace the last install comand to this if you allready have installed the nvidia driver 
sudo apt-get -y install cuda-toolkit-12-1
(https://developer.nvidia.com/cuda-12-1-0-download-archive)


### install gcc-11
sudo apt install gcc-11 g++-11
### update ./bashrc so nvcc is loaded

e.g with

export PATH=/usr/local/cuda-12.1/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH



### Installation part 2.
please note that two different .sh scripts are provided for installation. install.sh and install_CUDA_ARCH_7_5.sh depending on the GPU you have acess to
Simply run [`install.sh`](install.sh) to install all dependencies in a new conda environment 
named `spt`. 
```bash
# Creates a conda env named 'spt' env and installs dependencies
./install.sh
```


<br>


<br>

## 🚀  Usage
### Prediction
Given a folder with .laz files you can classify all points in the laz files with 
```bash
python python predict_many.py --inputlaz path/to/(folder or las.laz) --output_folder path/to/outputfolder --ckpt_path path/to/checkpoints/a_file.ckpt --config path/to/config
```

To classify the example dataset with a pretrained model do the following 
```bash
python predict_many.py --inputlaz data/toy_laz_dataset/raw/test/ --ckpt_path models/example.ckpt --output_folder outputfolder/ --config experiment=semantic/vox025kds
```

### Training


```bash
# Train SPT on the toy_laz_dataset that comes with the repo
python src/train.py experiment=semantic/vox025toy_laz_dataset.yaml
```

### CUDA Out-Of-Memory Errors
Having some CUDA OOM errors 💀💾 ? Here are some parameters you can play 
with to mitigate GPU memory use, based on when the error occurs.

<details>
<summary><b>Parameters affecting CUDA memory.</b></summary>

**Legend**: 🟡 Preprocessing | 🔴 Training | 🟣 Inference (including validation and testing during training)

| Parameter                                   | Description                                                                                                                                                                                                                        |  When  |
|:--------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------:|
| `datamodule.xy_tiling`                      | Splits dataset tiles into xy_tiling^2 smaller tiles, based on a regular XY grid. Ideal square-shaped tiles à la DALES. Note this will affect the number of training steps.                                                         |  🟡🟣  |
| `datamodule.pc_tiling`                      | Splits dataset tiles into 2^pc_tiling smaller tiles, based on a their principal component. Ideal for varying tile shapes à la S3DIS and KITTI-360. Note this will affect the number of training steps.                             |  🟡🟣  |
| `datamodule.max_num_nodes`                  | Limits the number of $P_1$ partition nodes/superpoints in the **training batches**.                                                                                                                                                |   🔴   |
| `datamodule.max_num_edges`                  | Limits the number of $P_1$ partition edges in the **training batches**.                                                                                                                                                            |   🔴   |
| `datamodule.voxel`                          | Increasing voxel size will reduce preprocessing, training and inference times but will reduce performance.                                                                                                                         | 🟡🔴🟣 |
| `datamodule.pcp_regularization`             | Regularization for partition levels. The larger, the fewer the superpoints.                                                                                                                                                        | 🟡🔴🟣 |
| `datamodule.pcp_spatial_weight`             | Importance of the 3D position in the partition. The smaller, the fewer the superpoints.                                                                                                                                            | 🟡🔴🟣 |
| `datamodule.pcp_cutoff`                     | Minimum superpoint size. The larger, the fewer the superpoints.                                                                                                                                                                    | 🟡🔴🟣 |
| `datamodule.graph_k_max`                    | Maximum number of adjacent nodes in the superpoint graphs. The smaller, the fewer the superedges.                                                                                                                                  | 🟡🔴🟣 |
| `datamodule.graph_gap`                      | Maximum distance between adjacent superpoints int the superpoint graphs. The smaller, the fewer the superedges.                                                                                                                    | 🟡🔴🟣 |
| `datamodule.graph_chunk`                    | Reduce to avoid OOM when `RadiusHorizontalGraph` preprocesses the superpoint graph.                                                                                                                                                |   🟡   |
| `datamodule.dataloader.batch_size`          | Controls the number of loaded tiles. Each **train batch** is composed of `batch_size`*`datamodule.sample_graph_k` spherical samplings. Inference is performed on **entire validation and test tiles**, without spherical sampling. |  🔴🟣  |
| `datamodule.sample_segment_ratio`           | Randomly drops a fraction of the superpoints at each partition level.                                                                                                                                                              |   🔴   |
| `datamodule.sample_graph_k`                 | Controls the number of spherical samples in the **train batches**.                                                                                                                                                                 |   🔴   |
| `datamodule.sample_graph_r`                 | Controls the radius of spherical samples in the **train batches**. Set to `sample_graph_r<=0` to use the entire tile without spherical sampling.                                                                                   |   🔴   |
| `datamodule.sample_point_min`               | Controls the minimum number of $P_0$ points sampled per superpoint in the **train batches**.                                                                                                                                       |   🔴   |
| `datamodule.sample_point_max`               | Controls the maximum number of $P_0$ points sampled per superpoint in the **train batches**.                                                                                                                                       |   🔴   |
| `callbacks.gradient_accumulator.scheduling` | Gradient accumulation. Can be used to train with smaller batches, with more training steps.                                                                                                                                        |   🔴   |

<br>
</details>

<br>


## Citing 
When using the code of the [superpoint_transformer](https://github.com/drprojects/superpoint_transformer) you should cite the original work

```
@article{robert2023spt,
  title={Efficient 3D Semantic Segmentation with Superpoint Transformer},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@article{robert2024scalable,
  title={Scalable 3D Panoptic Segmentation as Superpoint Graph Clustering},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE International Conference on 3D Vision},
  year={2024}
}
```
