---
license: apache-2.0
tags:
  - lidar
  - point-cloud
  - laz
  - als
  - semantic-segmentation
  - superpoint-transformer
library_name: pytorch-lightning
---

# KDS SPT LAZ model

Pretrained Superpoint Transformer (SPT) checkpoint for semantic segmentation on
the toy LAZ dataset (`vox025toy_laz_dataset`) from
[laz-superpoint_transformer](https://github.com/Klimadatastyrelsen/laz-superpoint_transformer).

## Model file

| File | Description |
|------|-------------|
| `vox025toy_laz_best.ckpt` | PyTorch Lightning checkpoint (~2.8 MB) |

## Training details

| Field | Value |
|-------|-------|
| Experiment | `experiment=semantic/vox025toy_laz_dataset` |
| Training run | `2026-06-16_11-14-06` |
| Best epoch | 909 (monitored on `val/miou`) |
| Best val/mIoU | ~41.15 |

## Companion dataset

Training tiles are in the HuggingFace dataset
[`rasmuspjohansson/KDS_laz_dataset`](https://huggingface.co/datasets/rasmuspjohansson/KDS_laz_dataset).
Download with `./scripts/download_toy_laz_dataset.sh` from the project repo.

## Download

From a clone of laz-superpoint_transformer:

```bash
./scripts/download_model.sh
```

This places the checkpoint at `checkpoints/vox025toy_laz_best.ckpt`.

## Usage

```bash
python predict_many.py \
  --ckpt_path checkpoints/vox025toy_laz_best.ckpt \
  --inputlaz /path/to/tiles \
  --output_folder /path/to/results
```

Docker wrapper:

```bash
./scripts/run_predict_docker.sh \
  --ckpt checkpoints/vox025toy_laz_best.ckpt \
  --input /path/to/tiles \
  --output /path/to/results
```

## Republish (maintainers)

To sync a new best checkpoint back to this HuggingFace model repo:

```bash
./scripts/upload_model.sh
```

Requires write access and a HuggingFace token (`HF_TOKEN` or
`my_huggingface_token.txt` at the orchestrator repo root, or `hftoken_write.txt`
in the laz-superpoint_transformer repo root).
