# Part of the implementation is borrowed and modified from HRNet,
# publicly available under the MIT License License at https://github.com/HRNet/HRNet-Semantic-Segmentation
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BlockModules import ASPP
from .hrnet_backnone import BatchNorm2d, HrnetBackBone, blocks_dict

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(BatchNorm2d(num_features, **kwargs), nn.ReLU())

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class SpatialGatherModule(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(
            2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)  # batch x k x c

        ocr_context = ocr_context.permute(0, 2,
                                          1).unsqueeze(3)  # batch x c x k x 1
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context,
                size=(h, w),
                mode='bilinear',
                align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(ObjectAttentionBlock):

    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(
            in_channels, key_channels, scale, bn_type=bn_type)


class SpatialOCRModule(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCRModule, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels, scale, bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(
                _in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class HrnetSuperAndOcr(HrnetBackBone):

    def __init__(self, **kwargs):
        super(HrnetSuperAndOcr, self).__init__(**kwargs)
        if 'architecture' not in kwargs:
            raise Exception('HrnetSuperAndOcr not exist architecture param!')
        self.architecture = kwargs['architecture']

        if 'class_num' not in kwargs:
            raise Exception('HrnetSuperAndOcr not exist class_num param!')
        self.class_num = kwargs['class_num']

        if 'ocr' not in kwargs:
            raise Exception('HrnetSuperAndOcr not exist ocr param!')
        ocr_mid_channels = kwargs['ocr']['mid_channels']
        ocr_key_channels = kwargs['ocr']['key_channels']
        dropout_rate = kwargs['ocr']['dropout_rate']
        scale = kwargs['ocr']['scale']

        if 'super_param' not in kwargs:
            raise Exception('HrnetSuperAndOcr not exist super_param param!')

        self.super_dict = kwargs['super_param']

        self.is_export_onnx = False
        self.is_export_full_onnx = False

        self.is_contain_tail = True if 'tail_param' in kwargs else False
        if self.is_contain_tail:
            self.stage_tail_dict = kwargs['tail_param']
            num_channels = self.stage_tail_dict['NUM_CHANNELS'][0]
            block = blocks_dict[self.stage_tail_dict['BLOCK']]
            num_blocks = self.stage_tail_dict['NUM_BLOCKS'][0]
            self.stage_tail = self._make_layer(block,
                                               self.backbone_last_inp_channels,
                                               num_channels, num_blocks)
            last_inp_channels = block.expansion * num_channels
        else:
            last_inp_channels = self.backbone_last_inp_channels

        self.is_contain_aspp = True if 'aspp' in kwargs else False

        if self.architecture == 'hrnet_super_ocr':
            self.is_ocr_first = False
            num_channels = [64, last_inp_channels]
            self.stage_super, super_stage_channels = self._make_stage(
                self.super_dict, num_channels)
            last_inp_channels = int(np.sum(super_stage_channels))

            if self.is_contain_aspp:
                aspp_param = kwargs['aspp']
                self.aspp_layer = ASPP(
                    inplanes=last_inp_channels,
                    outplanes=aspp_param['outplanes'],
                    dilations=aspp_param['dilations'],
                    drop_rate=aspp_param['drop_rate'])
                last_inp_channels = aspp_param['outplanes']

            self.aux_head = nn.Sequential(
                nn.Conv2d(
                    last_inp_channels,
                    last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    last_inp_channels,
                    self.class_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(
                    last_inp_channels,
                    ocr_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True),
            )
            self.ocr_gather_head = SpatialGatherModule(self.class_num)

            self.ocr_distri_head = SpatialOCRModule(
                in_channels=ocr_mid_channels,
                key_channels=ocr_key_channels,
                out_channels=ocr_mid_channels,
                scale=scale,
                dropout=dropout_rate,
            )

            self.cls_head = nn.Sequential(
                nn.Conv2d(
                    ocr_mid_channels,
                    ocr_mid_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    ocr_mid_channels,
                    self.class_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))
        else:
            self.is_ocr_first = True

            if self.is_contain_aspp:
                aspp_param = kwargs['aspp']
                self.aspp_layer = ASPP(
                    inplanes=last_inp_channels,
                    outplanes=aspp_param['outplanes'],
                    dilations=aspp_param['dilations'],
                    drop_rate=aspp_param['drop_rate'])
                last_inp_channels = aspp_param['outplanes']

            self.aux_head = nn.Sequential(
                nn.Conv2d(
                    last_inp_channels,
                    last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    last_inp_channels,
                    self.class_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(
                    last_inp_channels,
                    ocr_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True),
            )
            self.ocr_gather_head = SpatialGatherModule(self.class_num)

            self.ocr_distri_head = SpatialOCRModule(
                in_channels=ocr_mid_channels,
                key_channels=ocr_key_channels,
                out_channels=ocr_mid_channels,
                scale=scale,
                dropout=dropout_rate,
            )

            num_channels = [64, ocr_mid_channels]
            self.stage_super, super_stage_channels = self._make_stage(
                self.super_dict, num_channels)
            last_inp_channels = int(np.sum(super_stage_channels))

            self.cls_head = nn.Sequential(
                nn.Conv2d(
                    last_inp_channels,
                    last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    last_inp_channels,
                    self.class_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

    def forward(self, x):
        if self.is_export_onnx:
            x = x.permute(0, 3, 1, 2)
            raw_h, raw_w = x.size(2), x.size(3)
        if self.is_export_full_onnx:
            raw_h, raw_w = x.size(2), x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)  # 5, 64, 320, 320
        x_stem = self.relu(x)
        x = self.conv2(x_stem)
        x = self.bn2(x)
        x = self.relu(x)  # 5, 64, 160, 160

        x = self.layer1(x)  # 5, 256=64*4, 160, 160

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)  # [[5, 18, 160, 160],[5, 36, 80, 80]]
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(
            x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        feats = torch.cat([x[0], x1, x2, x3], 1)

        if self.is_contain_tail:
            feats = self.stage_tail(feats)

        if self.is_ocr_first:

            if self.is_contain_aspp:
                feats = self.aspp_layer(feats)
            # compute contrast feature
            out_aux = self.aux_head(feats)

            feats = self.conv3x3_ocr(feats)
            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)

            feats = [x_stem, feats]  # 320*320 2X
            x_super = self.stage_super(feats)

            xsuper_h, xsuper_w = x_super[0].size(2), x_super[0].size(3)
            x_super1 = F.interpolate(
                x_super[1],
                size=(xsuper_h, xsuper_w),
                mode='bilinear',
                align_corners=True)
            x_super = torch.cat([x_super[0], x_super1], 1)
            out = self.cls_head(x_super)

        else:
            x_super = [x_stem, feats]  # 320*320 2X, 160*160 4X
            x_super = self.stage_super(x_super)

            xsuper_h, xsuper_w = x_super[0].size(2), x_super[0].size(3)
            x_super1 = F.interpolate(
                x_super[1],
                size=(xsuper_h, xsuper_w),
                mode='bilinear',
                align_corners=True)
            x_super = torch.cat([x_super[0], x_super1], 1)

            if self.is_contain_aspp:
                x_super = self.aspp_layer(x_super)
            out_aux = self.aux_head(x_super)

            feats = self.conv3x3_ocr(x_super)
            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)

            out = self.cls_head(feats)

        if self.is_export_onnx or self.is_export_full_onnx:
            x_class = F.interpolate(
                out, size=(raw_h, raw_w), mode='bilinear', align_corners=True)
            x_class = torch.softmax(x_class, dim=1)
            _, x_class = torch.max(x_class, dim=1, keepdim=True)
            x_class = x_class.float()
            return x_class
        else:
            out_aux_seg = [
                out_aux, out
            ]  # out_aux: 5, 2, 160, 160(HRNet origin res); out: 5, 2, 320, 320(HRNet res+tail+aspp+ocr)
            return out_aux_seg


def get_seg_model(cfg, **kwargs):
    model = HrnetSuperAndOcr(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
