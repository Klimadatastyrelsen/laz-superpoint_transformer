#!/usr/bin/env python
"""Verify the .laz support by running a short training run on the toy dataset.

Runs ``src/train.py`` on the bundled ``toy_laz_dataset`` (mini split, a couple
of batches for a single epoch) so the full .laz path is exercised: reading
.laz tiles with laspy, preprocessing into a superpoint hierarchy, and one
optimisation step. Prints ``LAZ_TRAIN_OK`` on success and ``LAZ_TRAIN_FAILED``
otherwise, so check_verification_logs.sh can grep the result.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(REPO_ROOT, "data", "toy_laz_dataset", "raw")
REQUIRED_TILES = [
    os.path.join("train", "1km_6170_728.laz"),
    os.path.join("train", "1km_6171_728.laz"),
    os.path.join("train", "1km_6143_590.laz"),
    os.path.join("test", "1km_6147_588.laz"),
]


def ensure_data():
    missing = [t for t in REQUIRED_TILES
               if not os.path.isfile(os.path.join(RAW_DIR, t))]
    if not missing:
        print("LAZ_DATA_PRESENT", flush=True)
        return True
    print(f"LAZ_DATA_MISSING ({len(missing)} tiles); attempting download",
          flush=True)
    script = os.path.join(REPO_ROOT, "scripts", "download_toy_laz_dataset.sh")
    try:
        subprocess.run(["bash", script], check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"LAZ_DATA_DOWNLOAD_FAILED ({type(exc).__name__}: {exc})",
              flush=True)
        return False
    still_missing = [t for t in REQUIRED_TILES
                     if not os.path.isfile(os.path.join(RAW_DIR, t))]
    if still_missing:
        print(f"LAZ_DATA_DOWNLOAD_FAILED (still missing: {still_missing})",
              flush=True)
        return False
    print("LAZ_DATA_PRESENT", flush=True)
    return True


def main():
    print("LAZ_VERIFY_BEGIN", flush=True)
    if not ensure_data():
        print("LAZ_TRAIN_FAILED (dataset unavailable)", flush=True)
        return 1

    # Short training run: mini split, single epoch, a couple of batches, CSV
    # logger (no wandb), no test stage. A coarser voxel keeps preprocessing of
    # the large tiles fast while still exercising the full .laz pipeline.
    overrides = [
        "experiment=semantic/vox025toy_laz_dataset",
        "datamodule.mini=True",
        "datamodule.voxel=0.5",
        "datamodule.xy_tiling=2",
        "logger=csv",
        "trainer=gpu",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=2",
        "+trainer.num_sanity_val_steps=0",
        "test=False",
        "optimized_metric=null",
        "seed=42",
    ]
    cmd = [sys.executable, "src/train.py", *overrides]
    print("LAZ_TRAIN_CMD: " + " ".join(cmd), flush=True)

    env = dict(os.environ)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode == 0:
        print("LAZ_TRAIN_OK", flush=True)
        return 0
    print(f"LAZ_TRAIN_FAILED (train.py exit code {proc.returncode})",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
