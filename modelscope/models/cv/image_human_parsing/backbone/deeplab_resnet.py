# Part of the implementation is borrowed and modified from Detectron2, publicly available at
# https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/deeplab/resnet.py

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modelscope.models.cv.image_instance_segmentation.maskdino.utils import \
    Conv2d


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            'BN': torch.nn.BatchNorm2d,
            'GN': lambda channels: nn.GroupNorm(32, channels),
            'nnSyncBN': nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, *, stride=1, norm='BN'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels))

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 bottleneck_channels,
                 stride=1,
                 num_groups=1,
                 norm='BN',
                 stride_in_1x1=False,
                 dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeepLabStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=128, norm='BN'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 4
        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2))
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2))
        self.conv3 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class DeeplabResNet(nn.Module):

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            num_stages = max([{
                'res2': 1,
                'res3': 2,
                'res4': 3,
                'res5': 4
            }.get(f, 0) for f in out_features])
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, nn.Module), block

            name = 'res' + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = curr_channels = blocks[
                -1].out_channels
        self.stage_names = tuple(
            self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(
                ', '.join(children))

    def forward(self, x):
        assert x.dim(
        ) == 4, f'ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!'
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {
            name: dict(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return 0

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels,
                   **kwargs):
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f'same length as num_blocks={num_blocks}.')
                    newk = k[:-len('_per_block')]
                    assert newk not in kwargs, f'Cannot call make_stage with both {k} and {newk}!'
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **curr_kwargs))
            in_channels = out_channels
        return blocks


def build_resnet_deeplab_backbone(out_features, depth, num_groups,
                                  width_per_group, norm, stem_out_channels,
                                  res2_out_channels, stride_in_1x1,
                                  res4_dilation, res5_dilation,
                                  res5_multi_grid, input_shape):
    stem = DeepLabStem(
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
