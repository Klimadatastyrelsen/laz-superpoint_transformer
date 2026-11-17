"""Labels and splits for KDS_laz_dataset_v1.

19 LAS classes. Vegetation is stored as codes 3, 4, and 5.
Codes that are not listed map to the ignore label.
"""

import os

import numpy as np

# Tiles are read in place. The NAS that holds them does not support
# symlinks. Override with KDS_LAZ_SRC.
DEFAULT_LAZ_DIR = os.environ.get(
    'KDS_LAZ_SRC',
    '/mnt/T/mnt/trainingdata/lidar/KDS_laz_dataset_v1',
)


########################################################################
#                              Data splits                             #
########################################################################

# Held-out tiles are one 1 km cell from each geographic block.
# The toy dataset's val/test tiles (1km_6143_589, 1km_6147_588,
# 1km_6143_590) are not in this capture.
VAL_TILE = '1km_6148_682'
TEST_TILE = '1km_6177_722'

_SOUTH = [
    f'1km_{e}_{n}'
    for e in range(6147, 6153)
    for n in range(681, 685)
]
_NORTH = [
    f'1km_{e}_{n}'
    for e in range(6176, 6182)
    for n in range(722, 726)
]

TILES = {
    'train': [t for t in _SOUTH + _NORTH if t not in (VAL_TILE, TEST_TILE)],
    'val': [VAL_TILE],
    'test': [TEST_TILE],
}


########################################################################
#                                Labels                                #
########################################################################

# Train id equals the index in LAS_CODES. The ignore label is
# KDS_LAZ_DATASET_V1_NUM_CLASSES (one past the last train id).
LAS_CODES = np.asarray(
    [1, 2, 3, 4, 5, 6, 7, 9, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 30],
    dtype=np.int64,
)

KDS_LAZ_DATASET_V1_NUM_CLASSES = int(len(LAS_CODES))

# Index is the LAS classification code. Unmapped codes, including 0,
# stay at the ignore label. Do not initialize this with 0: that would
# train unknown points as Ground.
ID2TRAINID = np.full(
    int(LAS_CODES.max()) + 1, KDS_LAZ_DATASET_V1_NUM_CLASSES, dtype=np.int64)
for _train_id, _las_code in enumerate(LAS_CODES.tolist()):
    ID2TRAINID[int(_las_code)] = _train_id

# Train id -> LAS code written by inference. Same order as LAS_CODES.
TRAINID2LAS = LAS_CODES.astype(np.uint8)

CLASS_NAMES = [
    'Other',
    'Ground',
    'LowVegetation',
    'MediumVegetation',
    'HighVegetation',
    'BuildingRoof',
    'LowNoise',
    'Water',
    'BridgeDeck',
    'HighNoise',
    'RoofObjects',
    'Vehicles',
    'Walls',
    'BelowGround',
    'LowVoltageWires',
    'HighVoltageWires',
    'LowVoltageTowers',
    'HighVoltageTowers',
    'Fences',
    'Ignored',
]

CLASS_COLORS = np.asarray([
    [180, 180, 180],  # Other
    [188, 151, 98],   # Ground
    [144, 238, 144],  # Low vegetation
    [34, 139, 34],    # Medium vegetation
    [0, 100, 0],      # High vegetation
    [220, 20, 60],    # Building roof
    [255, 192, 203],  # Low noise
    [30, 144, 255],   # Water
    [255, 140, 0],    # Bridge deck
    [255, 0, 255],    # High noise
    [178, 34, 34],    # Roof objects
    [255, 215, 0],    # Vehicles
    [139, 69, 19],    # Walls
    [112, 128, 144],  # Below ground
    [0, 255, 255],    # Low-voltage wires
    [0, 0, 139],      # High-voltage wires
    [127, 255, 212],  # Low-voltage towers
    [75, 0, 130],     # High-voltage towers
    [154, 205, 50],   # Fences
])

# Intensity in these tiles is 16-bit (median ~1500, 99th percentile ~4500).
# The toy clip of 600 would saturate almost every point.
MAX_INTENSITY = 8192

# Instance segmentation is not used for this semantic run.
MIN_OBJECT_SIZE = 100
THING_CLASSES = [5, 8, 10, 11, 12, 14, 15, 16, 17, 18]
STUFF_CLASSES = [
    i for i in range(KDS_LAZ_DATASET_V1_NUM_CLASSES) if i not in THING_CLASSES
]

assert len(CLASS_NAMES) == KDS_LAZ_DATASET_V1_NUM_CLASSES + 1
assert len(CLASS_COLORS) == KDS_LAZ_DATASET_V1_NUM_CLASSES
assert len(TILES['train']) == 46
assert TILES['val'] == [VAL_TILE]
assert TILES['test'] == [TEST_TILE]
assert len({*TILES['train'], *TILES['val'], *TILES['test']}) == 48
