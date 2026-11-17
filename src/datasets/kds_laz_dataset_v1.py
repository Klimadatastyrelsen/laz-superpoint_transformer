import logging
import os
import os.path as osp
from typing import List

import laspy
import numpy as np
import torch

from src.data import Data
from src.datasets.base import BaseDataset
from src.datasets.kds_laz_dataset_v1_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_LAZ_DIR,
    ID2TRAINID,
    KDS_LAZ_DATASET_V1_NUM_CLASSES,
    MAX_INTENSITY,
    STUFF_CLASSES,
    TILES,
)


log = logging.getLogger(__name__)

# Occasional DataLoader issues with many workers. Same workaround as the
# toy LAZ dataset.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['KDS_laz_dataset_v1', 'MiniKDS_laz_dataset_v1']

DATA_SUBDIR_NAME = 'KDS_laz_dataset_v1'


def read_kds_laz_dataset_v1_tile(
        filepath: str,
        xyz: bool = True,
        intensity: bool = True,
        semantic: bool = True,
        instance: bool = False,
        remap: bool = False,
        max_intensity: int = MAX_INTENSITY) -> Data:
    """Read one KDS_laz_dataset_v1 tile saved as .laz.

    When ``remap`` is True, LAS classification codes are mapped to
    contiguous train ids. Codes absent from ``ID2TRAINID`` become the
    ignore label ``KDS_LAZ_DATASET_V1_NUM_CLASSES``.
    """
    data = Data()
    las = laspy.read(filepath)

    if xyz:
        pos = torch.stack([
            torch.from_numpy(np.array(las[axis], dtype=np.float64))
            for axis in ("X", "Y", "Z")
        ], dim=-1)
        pos *= torch.as_tensor(np.array(las.header.scales), dtype=torch.float64)
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    if intensity:
        data.intensity = torch.from_numpy(
            np.array(las['intensity'], dtype=np.float32)
        ).clip(min=0, max=max_intensity) / max_intensity

    if semantic:
        y = torch.as_tensor(np.asarray(las['classification']), dtype=torch.long)
        if remap:
            max_class = int(y.max()) if y.numel() else 0
            if max_class >= len(ID2TRAINID):
                mapping = np.full(
                    max_class + 1, KDS_LAZ_DATASET_V1_NUM_CLASSES, dtype=np.int64)
                mapping[:len(ID2TRAINID)] = ID2TRAINID
            else:
                mapping = ID2TRAINID
            data.y = torch.from_numpy(np.ascontiguousarray(mapping))[y]
        else:
            data.y = y

    if instance:
        raise NotImplementedError("KDS_laz_dataset_v1 tiles have no instance labels.")

    return data


class KDS_laz_dataset_v1(BaseDataset):
    """KDS_laz_dataset_v1 tiles (LAS classification, semantic only).

    ``.laz`` files are read from ``KDS_LAZ_SRC`` (see ``DEFAULT_LAZ_DIR``).
    Preprocessed ``.h5`` files are written under
    ``{data_dir}/KDS_laz_dataset_v1/processed/``.
    """

    @property
    def data_subdir_name(self) -> str:
        return DATA_SUBDIR_NAME

    @property
    def raw_dir(self) -> str:
        return DEFAULT_LAZ_DIR

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return KDS_LAZ_DATASET_V1_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> dict:
        return TILES

    def download_dataset(self) -> None:
        """Tiles are checked by ``scripts/prepare_kds_laz_dataset_v1.sh``."""
        return None

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_kds_laz_dataset_v1_tile(
            raw_cloud_path, intensity=True, semantic=True, instance=False,
            remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.raw_dir}/
        └── {{tile_name}}.laz
    {self.root}/
        └── processed/
            └── {{train, val, test}}/
            """

    def _split_for_base_id(self, base_id: str) -> str:
        for split, ids in TILES.items():
            if base_id in ids:
                return split
        raise ValueError(f"Unknown tile id '{base_id}'")

    def id_to_relative_raw_path(self, id: str) -> str:
        base_id = self.id_to_base_id(id)
        self._split_for_base_id(base_id)
        return base_id + '.laz'

    def processed_to_raw_path(self, processed_path: str) -> str:
        _stage, _hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]
        base_id = self.id_to_base_id(cloud_id)
        self._split_for_base_id(base_id)
        return osp.join(self.raw_dir, base_id + '.laz')


class MiniKDS_laz_dataset_v1(KDS_laz_dataset_v1):
    """A few subtiles, for the smoke run. Shares the full dataset directory."""

    _NUM_MINI = 2

    @property
    def all_cloud_ids(self) -> dict:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return DATA_SUBDIR_NAME

    def process(self) -> None:
        super().process()

    def download(self) -> None:
        super().download()
