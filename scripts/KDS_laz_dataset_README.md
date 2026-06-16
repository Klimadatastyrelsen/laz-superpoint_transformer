---
license: cc-by-4.0
task_categories:
  - image-segmentation
tags:
  - lidar
  - point-cloud
  - laz
  - als
  - semantic-segmentation
language:
  - en
size_categories:
  - n<1K
---

# KDS toy LAZ dataset

Small airborne LiDAR (ALS) tile set in `.laz` format for the `toy_laz_dataset`
in [laz-superpoint_transformer](https://github.com/Klimadatastyrelsen/laz-superpoint_transformer).

## Splits

Splits are defined in `src/datasets/toy_laz_dataset_config.py`. The val tile is
stored under `raw/train/` on disk (same layout as this repo).

| Split | Tiles | HF path |
|-------|-------|---------|
| train | `1km_6170_728`, `1km_6171_728`, `1km_6172_728`, `1km_6173_728` | `train/*.laz` |
| val | `1km_6143_589` (includes vehicle class) | `train/1km_6143_589.laz` |
| test | `1km_6147_588`, `1km_6143_590` | `test/*.laz` |

## Files

| Path | Split |
|------|-------|
| `train/1km_6170_728.laz` | train |
| `train/1km_6171_728.laz` | train |
| `train/1km_6143_589.laz` | val |
| `train/1km_6172_728.laz` | train |
| `train/1km_6173_728.laz` | train |
| `test/1km_6147_588.laz` | test |
| `test/1km_6143_590.laz` | test |

Seven `.laz` files, ~700 MB total.

## Download

From a clone of laz-superpoint_transformer:

```bash
./scripts/download_toy_laz_dataset.sh
```

This places tiles under `data/toy_laz_dataset/raw/{train,test}/`.

## Republish (maintainers)

To sync local tiles back to this HuggingFace dataset:

```bash
./scripts/upload_toy_laz_dataset.sh
```

Requires write access and a HuggingFace token (`HF_TOKEN` or
`my_huggingface_token.txt` at the orchestrator repo root).
