# The implementation refers to the VitAdapter
# available at
# https://github.com/czczup/ViT-Adapter.git
from mmcv.utils import Registry, build_from_cfg

PIXEL_SAMPLERS = Registry('pixel sampler')


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
