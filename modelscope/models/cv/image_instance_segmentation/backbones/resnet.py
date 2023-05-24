# Part of the implementation is borrowed and modified from Detectron2, publicly available at
# https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/deeplab/resnet.py

import torch.nn.functional as F
from torch import nn

from modelscope.models.cv.image_human_parsing.backbone.deeplab_resnet import (
    BottleneckBlock, DeeplabResNet, get_norm)
from modelscope.models.cv.image_instance_segmentation.maskdino.utils import \
    Conv2d


class BasicStem(nn.Module):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, norm='BN'):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 4
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


def build_resnet_backbone(out_features, depth, num_groups, width_per_group,
                          norm, stem_out_channels, res2_out_channels,
                          stride_in_1x1, res4_dilation, res5_dilation,
                          res5_multi_grid, input_shape):
    stem = BasicStem(
        in_channels=input_shape['channels'],
        out_channels=stem_out_channels,
        norm=norm)
    bottleneck_channels = num_groups * width_per_group
    in_channels = stem_out_channels
    out_channels = res2_out_channels

    assert res4_dilation in {
        1, 2
    }, 'res4_dilation cannot be {}.'.format(res4_dilation)
    assert res5_dilation in {
        1, 2, 4
    }, 'res5_dilation cannot be {}.'.format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4

    num_blocks_per_stage = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }[depth]

    stages = []
    out_stage_idx = [{
        'res2': 2,
        'res3': 3,
        'res4': 4,
        'res5': 5
    }[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2
        stride_per_block = [first_stride]
        stride_per_block += [1] * (num_blocks_per_stage[idx] - 1)
        stage_kargs = {
            'num_blocks': num_blocks_per_stage[idx],
            'stride_per_block': stride_per_block,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'norm': norm,
            'bottleneck_channels': bottleneck_channels,
            'stride_in_1x1': stride_in_1x1,
            'dilation': dilation,
            'num_groups': num_groups,
            'block_class': BottleneckBlock
        }
        if stage_idx == 5:
            stage_kargs.pop('dilation')
            stage_kargs['dilation_per_block'] = [
                dilation * mg for mg in res5_multi_grid
            ]
        blocks = DeeplabResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return DeeplabResNet(stem, stages, out_features=out_features)
