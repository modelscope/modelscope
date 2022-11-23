# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors, and available
# at https://github.com/tinyvision/damo-yolo.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.tinynas_detection.core.ops import ConvBNAct


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
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x


class ZeroHead(nn.Module):
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
            strides=[8, 16, 32],
            norm='gn',
            act='relu',
            nms_conf_thre=0.05,
            nms_iou_thre=0.7,
            nms=True,
            **kwargs):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.act = act
        self.strides = strides
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.strides)
        # add 1 for keep consistance with former models
        self.cls_out_channels = num_classes + 1
        self.reg_max = reg_max

        self.nms = nms
        self.nms_conf_thre = nms_conf_thre
        self.nms_iou_thre = nms_iou_thre

        self.feat_size = [torch.zeros(4) for _ in strides]

        super(ZeroHead, self).__init__()
        self.integral = Integral(self.reg_max)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            cls_convs.append(
                ConvBNAct(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=1,
                    norm='bn',
                    act=self.act))
            reg_convs.append(
                ConvBNAct(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=1,
                    norm='bn',
                    act=self.act))

        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

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

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        if self.training:
            return NotImplementedError
        else:
            return self.forward_eval(xin=xin, labels=labels, imgs=imgs)

    def forward_eval(self, xin, labels=None, imgs=None):

        # prepare priors for label assignment and bbox decode
        if self.feat_size[0] != xin[0].shape:
            mlvl_priors_list = [
                self.get_single_level_center_priors(
                    xin[i].shape[0],
                    xin[i].shape[-2:],
                    stride,
                    dtype=torch.float32,
                    device=xin[0].device)
                for i, stride in enumerate(self.strides)
            ]
            self.mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
            self.feat_size[0] = xin[0].shape

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)[:, :, :self.num_classes]
        bbox_preds = torch.cat(bbox_preds, dim=1)
        # batch bbox decode
        bbox_preds = self.integral(bbox_preds) * self.mlvl_priors[..., 2, None]
        bbox_preds = distance2bbox(self.mlvl_priors[..., :2], bbox_preds)

        res = torch.cat([bbox_preds, cls_scores[..., 0:self.num_classes]],
                        dim=-1)
        return res

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        if self.training:
            bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H,
                                                    W)
            bbox_before_softmax = bbox_before_softmax.flatten(
                start_dim=3).permute(0, 3, 1, 2)
        bbox_pred = F.softmax(
            bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)

        cls_score = gfl_cls(cls_feat).sigmoid()

        cls_score = cls_score.flatten(start_dim=2).permute(
            0, 2, 1)  # N, h*w, self.num_classes+1
        bbox_pred = bbox_pred.flatten(start_dim=3).permute(
            0, 3, 1, 2)  # N, h*w, 4, self.reg_max+1
        if self.training:
            return cls_score, bbox_pred, bbox_before_softmax
        else:
            return cls_score, bbox_pred

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
