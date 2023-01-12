# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import functools
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.tinynas_detection.damo.base_models.core.base_ops import (
    BaseConv, DWConv)


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def multi_apply(func, *args, **kwargs):

    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def xyxy2CxCywh(xyxy, size=None):
    x1 = xyxy[..., 0]
    y1 = xyxy[..., 1]
    x2 = xyxy[..., 2]
    y2 = xyxy[..., 3]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1
    if size is not None:
        w = w.clamp(min=0, max=size[1])
        h = h.clamp(min=0, max=size[0])
    return torch.stack([cx, cy, w, h], axis=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        b, nb, ne, _ = x.size()
        x = x.reshape(b * nb * ne, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, nb, 4)
        return x


class GFocalHead_Tiny(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            stacked_convs=4,  # 4
            feat_channels=256,
            reg_max=12,
            reg_topk=4,
            reg_channels=64,
            strides=[8, 16, 32],
            add_mean=True,
            norm='gn',
            act='relu',
            start_kernel_size=3,
            conv_groups=1,
            conv_type='BaseConv',
            simOTA_cls_weight=1.0,
            simOTA_iou_weight=3.0,
            octbase=8,
            simlqe=False,
            use_lqe=True,
            **kwargs):
        self.simlqe = simlqe
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.use_lqe = use_lqe
        self.feat_channels = feat_channels if isinstance(feat_channels, list) \
            else [feat_channels] * len(self.strides)

        self.cls_out_channels = num_classes + 1  # add 1 for keep consistance with former models
        # and will be deprecated in future.
        self.stacked_convs = stacked_convs
        self.conv_groups = conv_groups
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.start_kernel_size = start_kernel_size

        self.norm = norm
        self.act = act
        self.conv_module = DWConv if conv_type == 'DWConv' else BaseConv

        if add_mean:
            self.total_dim += 1

        super(GFocalHead_Tiny, self).__init__()
        self.integral = Integral(self.reg_max)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else self.start_kernel_size
            cls_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))
            reg_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))
        if self.use_lqe:
            if not self.simlqe:
                conf_vector = [
                    nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)
                ]
            else:
                conf_vector = [
                    nn.Conv2d(4 * (self.reg_max + 1), self.reg_channels, 1)
                ]
            conf_vector += [self.relu]
            conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
            reg_conf = nn.Sequential(*conf_vector)
        else:
            reg_conf = None

        return cls_convs, reg_convs, reg_conf

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_confs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, reg_conf = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.reg_confs.append(reg_conf)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(
                self.feat_channels[i], self.cls_out_channels, 3, padding=1)
            for i in range(len(self.strides))
        ])

        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(
                self.feat_channels[i], 4 * (self.reg_max + 1), 3, padding=1)
            for i in range(len(self.strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self,
                xin,
                labels=None,
                imgs=None,
                conf_thre=0.05,
                nms_thre=0.7):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels')
                                    - 1).long())  # labels starts from 1

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                xin[i].shape[0],
                xin[i].shape[-2:],
                stride,
                dtype=torch.float32,
                device=xin[0].device) for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.reg_confs,
            self.scales,
        )
        flatten_cls_scores = torch.cat(cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(bbox_preds, dim=1)

        # calculating losses or bboxes decoded
        if self.training:
            loss = self.loss(flatten_cls_scores, flatten_bbox_preds,
                             gt_bbox_list, gt_cls_list, mlvl_priors)
            return loss
        else:
            output = self.get_bboxes(flatten_cls_scores, flatten_bbox_preds,
                                     mlvl_priors)
            return output

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg,
                       reg_conf, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(
            bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        if self.use_lqe:
            if not self.simlqe:
                prob_topk, _ = prob.topk(self.reg_topk, dim=2)

                if self.add_mean:
                    stat = torch.cat(
                        [prob_topk,
                         prob_topk.mean(dim=2, keepdim=True)],
                        dim=2)
                else:
                    stat = prob_topk

                quality_score = reg_conf(
                    stat.reshape(N, 4 * self.total_dim, H, W))
            else:
                quality_score = reg_conf(
                    bbox_pred.reshape(N, 4 * (self.reg_max + 1), H, W))

            cls_score = gfl_cls(cls_feat).sigmoid() * quality_score
        else:
            cls_score = gfl_cls(cls_feat).sigmoid()

        flatten_cls_score = cls_score.flatten(start_dim=2).transpose(1, 2)
        flatten_bbox_pred = bbox_pred.flatten(start_dim=2).transpose(1, 2)
        return flatten_cls_score, flatten_bbox_pred

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype,
                                device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype,
                                device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def get_bboxes(self,
                   cls_preds,
                   reg_preds,
                   mlvl_center_priors,
                   img_meta=None):

        dis_preds = self.integral(reg_preds) * mlvl_center_priors[..., 2, None]
        bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)

        return cls_preds[..., 0:self.num_classes], bboxes
