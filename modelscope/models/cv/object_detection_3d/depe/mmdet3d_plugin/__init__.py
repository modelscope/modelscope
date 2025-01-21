"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin
"""
from .core.bbox.assigners import HungarianAssigner3D
from .core.bbox.coders import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .datasets import CustomNuScenesDataset
from .datasets.pipelines import NormalizeMultiviewImage, PadMultiViewImage
from .models.backbones import VoVNet
from .models.dense_heads import PETRv2DEDNHead
from .models.detectors import Petr3D
from .models.necks import CPFPN
