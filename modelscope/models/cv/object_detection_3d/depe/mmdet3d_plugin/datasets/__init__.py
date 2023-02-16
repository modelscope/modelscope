"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly avaialbe at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/datasets
"""
from .nuscenes_dataset import CustomNuScenesDataset

__all__ = [
    'CustomNuScenesDataset',
]
