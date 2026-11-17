import logging

from src.datamodules.base import BaseDataModule
from src.datasets.kds_laz_dataset_v1 import (
    KDS_laz_dataset_v1,
    MiniKDS_laz_dataset_v1,
)


log = logging.getLogger(__name__)


class KDS_LAZ_DATASET_V1_DataModule(BaseDataModule):
    """LightningDataModule for the KDS_laz_dataset_v1 LAZ tiles."""

    _DATASET_CLASS = KDS_laz_dataset_v1
    _MINIDATASET_CLASS = MiniKDS_laz_dataset_v1
