"""
Copyright (c) Microsoft
Licensed under the MIT License.
Written by Bin Xiao (Bin.Xiao@microsoft.com)
Modified by Ke Sun (sunk@mail.ustc.edu.cn)
https://github.com/HRNet/HRNet-Image-Classification/blob/master/lib/models/cls_hrnet.py
"""

import functools
import logging
import os

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F

from modelscope.utils.logger import get_logger

BN_MOMENTUM = 0.01  # 0.01 for seg
logger = get_logger()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 fuse_method,
                 multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels,
                             num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.info(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.info(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.info(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(
            block(self.num_inchannels[branch_index],
                  num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index],
                      num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False),
                            nn.BatchNorm2d(
                                num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 leaky_relu=False,
                 attn_weight=1,
                 fix_domain=1,
                 domain_center_model='',
                 **kwargs):
        super(HighResolutionNet, self).__init__()

        self.criterion_attn = torch.nn.MSELoss(reduction='sum')
        self.domain_center_model = domain_center_model
        self.attn_weight = attn_weight
        self.fix_domain = fix_domain
        self.cosine = 1

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        num_channels = 64
        block = blocks_dict['BOTTLENECK']
        num_blocks = 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # -- stage 2
        self.stage2_cfg = {}
        self.stage2_cfg['NUM_MODULES'] = 1
        self.stage2_cfg['NUM_BRANCHES'] = 2
        self.stage2_cfg['BLOCK'] = 'BASIC'
        self.stage2_cfg['NUM_BLOCKS'] = [4, 4]
        self.stage2_cfg['NUM_CHANNELS'] = [40, 80]
        self.stage2_cfg['FUSE_METHOD'] = 'SUM'

        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # -- stage 3
        self.stage3_cfg = {}
        self.stage3_cfg['NUM_MODULES'] = 4
        self.stage3_cfg['NUM_BRANCHES'] = 3
        self.stage3_cfg['BLOCK'] = 'BASIC'
        self.stage3_cfg['NUM_BLOCKS'] = [4, 4, 4]
        self.stage3_cfg['NUM_CHANNELS'] = [40, 80, 160]
        self.stage3_cfg['FUSE_METHOD'] = 'SUM'

        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
        last_inp_channels = np.int(np.sum(pre_stage_channels)) + 256
        self.redc_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(True),
        )

        self.aspp = nn.ModuleList(aspp(in_channel=128))

        # additional layers specfic for Phase 3
        self.pred_conv = nn.Conv2d(128, 512, 3, padding=1)
        self.pred_bn = nn.BatchNorm2d(512)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        # Specially for hidden domain
        # Set the domain for learnable parameters
        domain_center_src = np.load(self.domain_center_model)
        G_SHA = torch.from_numpy(domain_center_src['G_SHA']).view(1, -1, 1, 1)
        G_SHB = torch.from_numpy(domain_center_src['G_SHB']).view(1, -1, 1, 1)
        G_QNRF = torch.from_numpy(domain_center_src['G_QNRF']).view(
            1, -1, 1, 1)

        self.n_domain = 3

        self.G_all = torch.cat(
            [G_SHA.clone(), G_SHB.clone(),
             G_QNRF.clone()], dim=0)

        self.G_all = nn.Parameter(self.G_all)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
        )

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i],
                                momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches, block, num_blocks,
                                     num_inchannels, num_channels, fuse_method,
                                     reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_head_1 = x

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        x = self.stage3(x_list)

        # Replace the classification heaeder with custom setting
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x_head_1], 1)
        # first, reduce the channel down
        x = self.redc_layer(x)

        pred_attn = self.GAP(F.relu_(self.pred_bn(self.pred_conv(x))))
        pred_attn = F.softmax(pred_attn, dim=1)
        pred_attn_list = torch.chunk(pred_attn, 4, dim=1)

        aspp_out = []
        for k, v in enumerate(self.aspp):
            if k % 2 == 0:
                aspp_out.append(self.aspp[k + 1](v(x)))
            else:
                continue
        # Using Aspp add, and relu inside
        for i in range(4):
            x = x + F.relu_(aspp_out[i] * 0.25) * pred_attn_list[i]

        bz = x.size(0)
        # -- Besides, we also need to let the prediction attention be close to visable domain
        # -- Calculate the domain distance and get the weights
        # - First, detach domains
        G_all_d = self.G_all.detach()  # use detached G_all for calulcating
        pred_attn_d = pred_attn.detach().view(bz, 512, 1, 1)

        if self.cosine == 1:
            G_A, G_B, G_Q = torch.chunk(G_all_d, self.n_domain, dim=0)

            cos_dis_A = F.cosine_similarity(pred_attn_d, G_A, dim=1).view(-1)
            cos_dis_B = F.cosine_similarity(pred_attn_d, G_B, dim=1).view(-1)
            cos_dis_Q = F.cosine_similarity(pred_attn_d, G_Q, dim=1).view(-1)

            cos_dis_all = torch.stack([cos_dis_A, cos_dis_B,
                                       cos_dis_Q]).view(bz, -1)  # bz*3

            cos_dis_all = F.softmax(cos_dis_all, dim=1)

            target_attn = cos_dis_all.view(bz, self.n_domain, 1, 1, 1).expand(
                bz, self.n_domain, 512, 1, 1) * self.G_all.view(
                    1, self.n_domain, 512, 1, 1).expand(
                        bz, self.n_domain, 512, 1, 1)
            target_attn = torch.sum(
                target_attn, dim=1, keepdim=False)  # bz * 512 * 1 * 1

            if self.fix_domain:
                target_attn = target_attn.detach()

        else:
            raise ValueError('Have not implemented not cosine distance yet')

        x = self.last_layer(x)
        x = F.relu_(x)

        x = F.interpolate(
            x, size=(x0_h * 2, x0_w * 2), mode='bilinear', align_corners=False)

        return x, pred_attn, target_attn

    def init_weights(
        self,
        pretrained='',
    ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                logger.info(f'=> loading {k} pretrained model {pretrained}')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            assert 1 == 2


def aspp(aspp_num=4, aspp_stride=2, in_channel=512, use_bn=True):
    aspp_list = []
    for i in range(aspp_num):
        pad = (i + 1) * aspp_stride
        dilate = pad
        conv_aspp = nn.Conv2d(
            in_channel, in_channel, 3, padding=pad, dilation=dilate)
        aspp_list.append(conv_aspp)
        if use_bn:
            aspp_list.append(nn.BatchNorm2d(in_channel))

    return aspp_list
