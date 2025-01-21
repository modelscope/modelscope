"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/core/bbox/assigners
"""
from .hungarian_assigner_3d import HungarianAssigner3D

__all__ = ['HungarianAssigner3D']
