# Robustness improvements brought in from `lars-flem/superpoint_transformer_new`

Source: the Master's-thesis fork
[`lars-flem/superpoint_transformer_new`](https://github.com/lars-flem/superpoint_transformer_new)
(`ROBUSTNESS_IMPROVEMENTS.md` and `SETUP_GUIDE.md`). That work hardened SPT for
national-scale airborne LiDAR, where unfiltered LAS/LAZ tiles routinely contain
water, narrow strips, and sparse subregions that the upstream code assumed away.

The user's priority is **handling frames/tiles with too few points**. The
centerpiece is the sparse-subtile skipping (#5); the remaining guards make the
preprocessing and model robust to empty / degenerate inputs.

## Status

All ten improvements have been adapted to the current upstream code and applied
to this repo. Where upstream APIs had drifted from the fork, the change was
re-expressed against the current implementation.

| # | Scenario | File (here) | Status |
|---|---|---|---|
| 1 | Empty `edge_index` in `scatter_nearest_neighbor` | [src/utils/scatter.py](src/utils/scatter.py) | done |
| 2 | Empty `edge_index` in `subedges` | [src/utils/graph.py](src/utils/graph.py) | done |
| 3 | Missing optional keys when batching | [src/data/data.py](src/data/data.py) | done |
| 4 | NAG shallower than model down-stages | [src/models/components/spt.py](src/models/components/spt.py) | done |
| 5 | Sparse / near-empty subtiles (centerpiece) | [src/datasets/base.py](src/datasets/base.py) | done |
| 6 | RANSAC failure on degenerate ground | [src/utils/ground.py](src/utils/ground.py) | done |
| 7 | Empty input to `GridSampling3D` / `QuantizePointCoordinates` | [src/transforms/sampling.py](src/transforms/sampling.py) | done |
| 8 | Degenerate / empty `SampleXYTiling` | [src/transforms/sampling.py](src/transforms/sampling.py) | done |
| 9 | Empty input to `KNN` / `Inliers` / `Outliers` / `PointFeatures` | [src/transforms/neighbors.py](src/transforms/neighbors.py), [src/transforms/point.py](src/transforms/point.py) | done |
| 10 | Single-node NAG level in horizontal graph | [src/transforms/graph.py](src/transforms/graph.py) | done |

## Details

### 1. `scatter_nearest_neighbor` empty-edge guard
At deep hierarchy levels with very few superpoints the radius graph can be
empty; the chunking branch then `torch.cat`s an empty list. Added an early
return of correctly-shaped empty tensors.

### 2. `subedges` empty-edge guard
Same family as #1: after `to_trimmed()` removes all edges, return empty
`edge_index` / `ST_pairs` / `ST_uid`. Absorbs the empty graph emitted by #10.

### 3. Dynamic key exclusion in `Batch.from_data_list`
Optional keys (`super_index`, `sub`, ...) may exist on some samples and not
others when hierarchies differ. We now exclude any key not shared by all
samples before delegating to PyG's `collate()`, with supporting `if k not in d`
/ `if k not in batch` guards.

### 4. NAG-level bounds check in the SPT forward pass
A sparse input may produce a hierarchy shallower than the configured number of
down-stages. The down-stage loop now `break`s before indexing a non-existent
NAG level. Defensive once #5 is enabled.

### 5. Sparse subtile skipping with a `.skip` sentinel (centerpiece)
Added a `min_points_per_subtile` parameter on `BaseDataset` (exposed in the
datamodule config, default `0` = disabled). During preprocessing, subtiles with
fewer points than the threshold get a `.skip` sentinel written next to where the
`.h5` would live, and are then excluded from the dataset:
- `_skip_path()` resolves the sentinel path;
- `_process_single_cloud` writes the sentinel and returns early;
- `_process` treats a tile as done if its `.h5` **or** `.skip` exists (no
  endless re-processing);
- `_valid_processed_paths` returns only existing `.h5` paths; `__len__`,
  `__getitem__`, in-memory loading, and class-weight computation use it.

When `min_points_per_subtile == 0`, behavior is identical to upstream (no disk
scans, no `.skip` logic).

### 6. RANSAC flat-plane fallback for degenerate ground
On XY-degenerate ground candidates (e.g. flat water), sklearn's
`RANSACRegressor` raises `ValueError`. The CPU ground-plane fit is now wrapped
in `try/except ValueError` with a fallback to a flat plane at the mean Z.

### 7. Empty-input guards in `GridSampling3D` / `QuantizePointCoordinates`
`torch_cluster.grid_cluster` rejects empty input. Both transforms now early-
return on a 0-point cloud (setting `coords` / `grid_size` as appropriate).

### 8. `SampleXYTiling` span clamp + empty guard
Zero spatial span no longer produces NaN/inf (span clamped to 1); upper-edge
points stay in the last tile (`clip` to `1 - eps`); empty input returns an empty
selection. The fork's silent "most-populated-quadrant" fallback was intentionally
**not** brought in (it masked real config issues and is unnecessary with #5).

### 9. Empty-cloud guards in neighbor / feature transforms
`KNN`, `Inliers`, `Outliers`, and `PointFeatures` early-return on empty input,
initialising empty neighbor attributes for `KNN`.

### 10. Single-node level guard in horizontal graph construction
A NAG level collapsing to one node previously raised. It now logs a warning and
emits an empty `edge_index` (with `edge_attr = None`), absorbed downstream by #2.
