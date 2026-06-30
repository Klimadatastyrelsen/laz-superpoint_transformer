#!/usr/bin/env python
"""
Semantic segmentation inference on toy LAZ tiles (vox025toy_laz_dataset).

Runs the same XY tiling + preprocessing pipeline as training, writes
classified LAZ files, and optionally reports point-wise metrics.
"""

import argparse
import os
import re
import sys
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path

import hydra
import laspy
import numpy as np
import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.datasets.toy_laz_dataset import read_toy_laz_dataset_tile
from src.datasets.toy_laz_dataset_config import (
    CLASS_NAMES,
    ID2TRAINID,
    TOY_DATASET_NUM_CLASSES,
)
from src.metrics.semantic import ConfusionMatrix
from src.transforms import NAGRemoveKeys, SampleXYTiling, instantiate_datamodule_transforms
from src.utils import init_config

DEFAULT_CKPT = "checkpoints/vox025toy_laz_best.ckpt"
DEFAULT_CONFIG = "experiment=semantic/vox025toy_laz_dataset"

# Tolerance when comparing to training log metrics (--log). Point-wise LAZ
# inference differs from the Lightning test dataloader (tiling, stitching).
LOG_MIOU_TOL = 5.0
LOG_OA_TOL = 0.5

# Output LAZ layout. "native" mirrors the input (extra bytes, VLRs, etc.).
# "standard" writes LAS 1.4 point format 6 with 30-byte records (no vendor
# extra dimensions) so FugroViewer / QGIS Untwine can open the result.
OUTPUT_FORMATS = ("native", "standard")
STANDARD_POINT_FORMAT = 6
STANDARD_FILE_VERSION = "1.4"


class Tee:
    """Write to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(run_log_path):
    """Mirror stdout/stderr to run_log_path for later inspection."""
    run_log_path = Path(run_log_path)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(run_log_path, "a", encoding="utf-8")
    log_file.write(f"\n{'=' * 70}\n")
    log_file.write(f"predict_many run started: {datetime.now().isoformat()}\n")
    log_file.write(f"{'=' * 70}\n")
    log_file.flush()
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_file


def build_trainid2id(id2trainid, num_classes):
    """Inverse map: training id -> representative LAS classification code."""
    trainid2id = np.zeros(num_classes, dtype=np.uint8)
    for las_id, train_id in enumerate(id2trainid):
        if 0 <= train_id < num_classes:
            trainid2id[train_id] = las_id
    return trainid2id


TRAINID2ID = build_trainid2id(ID2TRAINID, TOY_DATASET_NUM_CLASSES)


def get_input_files(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(
                f"Input file must be .las or .laz, got: {input_path.suffix}"
            )
        return [input_path]

    if input_path.is_dir():
        las_files = sorted(
            set(input_path.glob("*.las"))
            | set(input_path.glob("*.laz"))
            | set(input_path.glob("*.LAS"))
            | set(input_path.glob("*.LAZ"))
        )
        if not las_files:
            raise ValueError(f"No .las or .laz files found in: {input_path}")
        return las_files

    raise ValueError(f"Input path must be a file or directory: {input_path}")


def xy_tile_indices(pos, x, y, tiling):
    """Return point indices for one XY tile (same logic as SampleXYTiling)."""
    if isinstance(tiling, int):
        tx, ty = tiling, tiling
    else:
        tx, ty = tiling
    tiling_t = torch.as_tensor((tx, ty), device=pos.device)

    xy = pos[:, :2].clone().view(-1, 2)
    xy -= xy.min(dim=0).values.view(1, 2)
    span = xy.max(dim=0).values.view(1, 2)
    span = torch.where(span > 0, span, torch.ones_like(span))
    xy /= span
    eps = 1e-6
    xy = xy.clip(min=0, max=1 - eps) * tiling_t.view(1, 2)
    xy = xy.long()
    return torch.where((xy[:, 0] == x) & (xy[:, 1] == y))[0]


def summarize_laz_header(las):
    """One-line summary of LAS/LAZ layout for logging."""
    h = las.header
    std_dims = {
        "X", "Y", "Z", "intensity", "return_number", "number_of_returns",
        "synthetic", "key_point", "withheld", "overlap", "scanner_channel",
        "scan_direction_flag", "edge_of_flight_line", "classification",
        "user_data", "scan_angle", "point_source_id", "gps_time",
        "red", "green", "blue",
    }
    extra = [d for d in h.point_format.dimension_names if d not in std_dims]
    try:
        crs = h.parse_crs()
        crs_label = crs.to_epsg() if crs else None
    except Exception:
        crs_label = "unparsed"
    return (
        f"version={h.version} pf={h.point_format.id} "
        f"record_len={h.point_format.size} points={h.point_count} "
        f"vlrs={len(h.vlrs)} extra_dims={extra or 'none'} crs={crs_label}"
    )


def _write_native_laz(las, output_file, las_classifications):
    las.classification = np.asarray(
        las_classifications, dtype=las.classification.dtype
    )
    las.write(str(output_file), do_compress=True)


def _write_standard_laz(las, output_file, las_classifications):
    """LAS 1.4 pf 6 with 30-byte records; drops vendor extra bytes."""
    out = laspy.create(
        point_format=STANDARD_POINT_FORMAT,
        file_version=STANDARD_FILE_VERSION,
    )
    out.header.scales = las.header.scales
    out.header.offsets = las.header.offsets
    out.header.global_encoding = las.header.global_encoding
    for vlr in las.header.vlrs:
        out.header.vlrs.append(vlr)

    for dim in out.point_format.dimension_names:
        if dim in las.point_format.dimension_names:
            out[dim] = np.array(las[dim])

    out.classification = np.asarray(
        las_classifications, dtype=out.classification.dtype
    )
    out.write(str(output_file), do_compress=True)


def write_classified_laz(
    input_file, output_file, las_classifications, output_format="native"
):
    """Write LAZ with updated classification.

    :param output_format: "native" preserves the input layout; "standard"
        normalises to LAS 1.4 point format 6 without vendor extra bytes.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}"
        )

    las = laspy.read(str(input_file))
    print(f"  Input layout: {summarize_laz_header(las)}")

    if output_format == "native":
        _write_native_laz(las, output_file, las_classifications)
    else:
        _write_standard_laz(las, output_file, las_classifications)
        written = laspy.read(str(output_file))
        print(f"  Output layout: {summarize_laz_header(written)}")

    print(f"  Saved predictions to: {output_file} ({output_format})")


def compute_metrics(pred_train, gt_train, num_classes):
    """Point-wise OA / mAcc / mIoU using the project ConfusionMatrix."""
    pred = torch.as_tensor(pred_train, dtype=torch.long)
    target = torch.as_tensor(gt_train, dtype=torch.long)
    valid = (pred >= 0) & (target >= 0) & (target < num_classes)
    if valid.sum() == 0:
        return None

    cm = ConfusionMatrix(num_classes)
    cm.update(pred[valid], target[valid])
    iou, seen = cm.iou(as_percent=True)
    return {
        "oa": cm.oa(as_percent=True),
        "macc": cm.macc(as_percent=True),
        "miou": cm.miou(as_percent=True),
        "iou_per_class": iou.cpu().numpy(),
        "seen_class": seen.cpu().numpy(),
        "n_points": int(valid.sum().item()),
    }


def print_metrics(label, metrics):
    print(f"\n--- {label} ---")
    print(f"  Points evaluated: {metrics['n_points']}")
    print(f"  OA:   {metrics['oa']:.4f}%")
    print(f"  mAcc: {metrics['macc']:.4f}%")
    print(f"  mIoU: {metrics['miou']:.4f}%")
    if "iou_per_class" in metrics and "seen_class" in metrics:
        for i, name in enumerate(CLASS_NAMES[:TOY_DATASET_NUM_CLASSES]):
            if i < len(metrics["iou_per_class"]) and metrics["seen_class"][i]:
                print(f"    IoU {name}: {metrics['iou_per_class'][i]:.4f}%")


def parse_training_log(log_path):
    """Extract test metrics from a training log (Lightning test table)."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    metrics = {}
    for key in ("test/miou", "test/oa", "test/macc"):
        match = re.search(rf"\│\s*{re.escape(key)}\s*\│\s*([0-9.eE+-]+)\s*\│", text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def compare_to_training_log(computed, log_metrics):
    print(f"\n{'=' * 70}")
    print("TRAINING LOG COMPARISON")
    print(f"{'=' * 70}")
    ok = True
    checks = [
        ("test/miou", computed["miou"], LOG_MIOU_TOL),
        ("test/oa", computed["oa"], LOG_OA_TOL),
    ]
    for key, got, tol in checks:
        if key not in log_metrics:
            print(f"  SKIP {key}: not found in training log")
            continue
        expected = log_metrics[key]
        diff = abs(got - expected)
        passed = diff <= tol
        ok = ok and passed
        status = "OK" if passed else "FAIL"
        print(
            f"  {status} {key}: log={expected:.4f} computed={got:.4f} "
            f"diff={diff:.4f} (tol={tol})"
        )
    if ok:
        print("LOG_CHECK_OK")
    else:
        print("LOG_CHECK_FAIL")
    return ok


def run_tile_inference(data_tile, cfg, transforms_dict, model):
    """Preprocess one XY subtile and return full-res train-id predictions."""
    nag = transforms_dict["pre_transform"](data_tile)
    nag = NAGRemoveKeys(
        level=0,
        keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys],
    )(nag)
    nag = NAGRemoveKeys(
        level="1+",
        keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys],
    )(nag)
    nag = nag.cuda()
    nag = transforms_dict["on_device_test_transform"](nag)
    with torch.no_grad():
        output = model(nag)
    return output.full_res_semantic_pred(
        super_index_level0_to_level1=nag[0].super_index,
        sub_level0_to_raw=nag[0].sub,
    ).cpu().long()


def process_file(
    input_file,
    output_folder,
    cfg,
    transforms_dict,
    model,
    xy_tiling,
    min_points_per_subtile=0,
    get_accuracy=False,
    output_format="native",
):
    """Process one LAZ file with XY tiling and stitch full-cloud predictions."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {input_file.name}")
    print(f"{'=' * 70}")

    print("  Step 1: Reading input LAZ")
    data_full = read_toy_laz_dataset_tile(
        str(input_file), semantic=get_accuracy, remap=True
    )
    n_points = data_full.num_points
    gt_train = data_full.y.cpu().numpy() if get_accuracy and hasattr(data_full, "y") else None

    pred_train = np.full(n_points, -1, dtype=np.int64)
    if isinstance(xy_tiling, int):
        tx = ty = xy_tiling
    else:
        tx, ty = xy_tiling

    print(f"  Step 2: Tiled inference ({tx}x{ty} subtiles)")
    for x, y in product(range(tx), range(ty)):
        idx = xy_tile_indices(data_full.pos, x, y, (tx, ty))
        if idx.numel() == 0:
            print(f"    Tile ({x + 1},{y + 1}): empty, skipping")
            continue
        data_tile = data_full.select(idx)[0]
        n_tile = data_tile.num_points
        if min_points_per_subtile > 0 and n_tile < min_points_per_subtile:
            print(
                f"    Tile ({x + 1},{y + 1}): {n_tile} points — skipping "
                f"({n_tile} < min_points_per_subtile={min_points_per_subtile})"
            )
            continue
        print(f"    Tile ({x + 1},{y + 1}): {n_tile} points")
        tile_pred = run_tile_inference(data_tile, cfg, transforms_dict, model)
        pred_train[idx.cpu().numpy()] = tile_pred.numpy()

    missing = int((pred_train < 0).sum())
    if missing:
        print(f"  Warning: {missing} points have no prediction (kept original class)")

    las = laspy.read(str(input_file))
    orig_class = np.array(las.classification)
    out_class = orig_class.copy()
    predicted_mask = pred_train >= 0
    out_class[predicted_mask] = TRAINID2ID[pred_train[predicted_mask]]

    output_file = output_folder / input_file.name
    print(f"  Step 3: Writing classified LAZ (output_format={output_format})")
    write_classified_laz(
        input_file, output_file, out_class, output_format=output_format
    )

    file_metrics = None
    if get_accuracy and gt_train is not None:
        print("  Step 4: Computing accuracy")
        file_metrics = compute_metrics(pred_train, gt_train, TOY_DATASET_NUM_CLASSES)
        if file_metrics:
            print_metrics(input_file.name, file_metrics)
        else:
            print("  No valid points for metric computation")

    print(f"  Done: {input_file.name}")
    return file_metrics


def aggregate_metrics(per_file_metrics):
    """Weighted average of per-file metrics by n_points."""
    total = sum(m["n_points"] for m in per_file_metrics)
    if total == 0:
        return None
    agg = {
        "oa": sum(m["oa"] * m["n_points"] for m in per_file_metrics) / total,
        "macc": sum(m["macc"] * m["n_points"] for m in per_file_metrics) / total,
        "miou": sum(m["miou"] * m["n_points"] for m in per_file_metrics) / total,
        "n_points": total,
    }
    return agg


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Semantic segmentation inference on toy LAZ tiles "
            "(vox025toy_laz_dataset checkpoints)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_many.py \\
      --inputlaz data/toy_laz_dataset/raw/test \\
      --output_folder output/best_test \\
      --ckpt_path logs/.../epoch_1159.ckpt \\
      --get_accuracy \\
      --log logs/docker_train.log \\
      --run_log logs/predict_best_test.log
        """,
    )
    parser.add_argument("--inputlaz", required=True, help="Input .laz file or directory")
    parser.add_argument("--output_folder", required=True, help="Output directory")
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT, help="Model checkpoint")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Hydra experiment")
    parser.add_argument(
        "--get_accuracy",
        action="store_true",
        help="Compare predictions to GT LAS labels; print OA/mAcc/mIoU",
    )
    parser.add_argument(
        "--log",
        dest="training_log",
        default=None,
        help="Training log to compare test metrics against (with --get_accuracy)",
    )
    parser.add_argument(
        "--run_log",
        default=None,
        help="Append all stdout/stderr to this file (default: <output_folder>/predict_run.log)",
    )
    parser.add_argument(
        "--min_points_per_subtile",
        type=int,
        default=None,
        help=(
            "Skip XY subtiles with fewer points than this (same as training "
            "preprocessing). Default: datamodule config value."
        ),
    )
    parser.add_argument(
        "--output_format",
        choices=OUTPUT_FORMATS,
        default="standard",
        help=(
            "Output LAZ layout: 'standard' (default) = LAS 1.4 pf 6, 30-byte "
            "records, no vendor extra bytes (FugroViewer/QGIS friendly); "
            "'native' = preserve input layout including extra bytes"
        ),
    )
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    run_log = args.run_log or str(output_folder / "predict_run.log")
    log_file = setup_run_logging(run_log)

    exit_code = 0
    try:
        print("=" * 70)
        print("TOY LAZ SEMANTIC SEGMENTATION INFERENCE")
        print("=" * 70)
        print(f"Input:         {args.inputlaz}")
        print(f"Output folder: {output_folder}")
        print(f"Checkpoint:    {args.ckpt_path}")
        print(f"Config:        {args.config}")
        print(f"Get accuracy:  {args.get_accuracy}")
        print(f"Training log:  {args.training_log}")
        print(f"Run log:       {run_log}")
        print(f"Output format: {args.output_format}")
        print("=" * 70)

        if not Path(args.ckpt_path).exists():
            print(f"Error: checkpoint not found: {args.ckpt_path}")
            sys.exit(1)

        input_files = get_input_files(args.inputlaz)
        print(f"\nFound {len(input_files)} file(s):")
        for f in input_files:
            print(f"  - {f.name}")

        print("\nLoading configuration and model...")
        cfg = init_config(
            overrides=[args.config, "datamodule.load_full_res_idx=True"]
        )
        xy_tiling = cfg.datamodule.xy_tiling
        if args.min_points_per_subtile is not None:
            min_points_per_subtile = max(int(args.min_points_per_subtile), 0)
        else:
            min_points_per_subtile = max(
                int(getattr(cfg.datamodule, "min_points_per_subtile", 0) or 0), 0
            )
        print(f"  xy_tiling: {xy_tiling}")
        if min_points_per_subtile > 0:
            print(f"  min_points_per_subtile: {min_points_per_subtile}")

        transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)
        model = hydra.utils.instantiate(cfg.model)
        model = model._load_from_checkpoint(args.ckpt_path)
        model = model.eval().cuda()
        print("  Model loaded")

        per_file_metrics = []
        failed = 0
        for i, input_file in enumerate(input_files, 1):
            print(f"\n[{i}/{len(input_files)}]", end=" ")
            try:
                m = process_file(
                    input_file,
                    output_folder,
                    cfg,
                    transforms_dict,
                    model,
                    xy_tiling,
                    min_points_per_subtile=min_points_per_subtile,
                    get_accuracy=args.get_accuracy,
                    output_format=args.output_format,
                )
                if m:
                    per_file_metrics.append(m)
            except Exception as exc:
                failed += 1
                print(f"  Error processing {input_file.name}: {exc}")
                traceback.print_exc()

        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Processed: {len(input_files) - failed}/{len(input_files)} files")
        print(f"Output:    {output_folder}")

        if args.get_accuracy and per_file_metrics:
            if len(per_file_metrics) > 1:
                agg = aggregate_metrics(per_file_metrics)
                print_metrics("AGGREGATE", agg)
            else:
                agg = per_file_metrics[0]

            if args.training_log:
                if not Path(args.training_log).exists():
                    print(f"Error: training log not found: {args.training_log}")
                    exit_code = 1
                else:
                    log_metrics = parse_training_log(args.training_log)
                    print(f"\nParsed from training log: {log_metrics}")
                    if not compare_to_training_log(agg, log_metrics):
                        exit_code = 1

        if failed:
            exit_code = 1

    finally:
        log_file.write(f"\npredict_many finished: {datetime.now().isoformat()}\n")
        log_file.flush()
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
