# The implementation here is modified based on nanodet,
# originally Apache 2.0 License and publicly available at https://github.com/RangiLyu/nanodet

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from .utils import ConvModule, DepthwiseConvModule


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop('type', 'nms')
    split_thr = nms_cfg_.pop('split_thr', 10000)
    if len(boxes_for_nms) < split_thr:
        keep = nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    valid_mask = scores > score_thr

    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
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


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(self,
                 num_classes,
                 input_channel,
                 feat_channels=96,
                 stacked_convs=2,
                 kernel_size=5,
                 strides=[8, 16, 32],
                 conv_type='DWConv',
                 norm_cfg=dict(type='BN'),
                 reg_max=7,
                 activation='LeakyReLU',
                 assigner_cfg=dict(topk=13),
                 **kwargs):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == 'Conv' else DepthwiseConvModule

        self.norm_cfg = norm_cfg
        self.distribution_project = Integral(self.reg_max)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(
                self.feat_channels,
                self.num_classes + 4 * (self.reg_max + 1),
                1,
                padding=0,
            ) for _ in self.strides
        ])

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                ))
        return cls_convs

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
                feats,
                self.cls_convs,
                self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta['warp_matrix']
            if isinstance(meta['warp_matrix'], list) else meta['warp_matrix'])
        img_heights = (
            meta['img_info']['height'].cpu().numpy() if isinstance(
                meta['img_info']['height'], torch.Tensor) else
            meta['img_info']['height'])
        img_widths = (
            meta['img_info']['width'].cpu().numpy() if isinstance(
                meta['img_info']['width'], torch.Tensor) else
            meta['img_info']['width'])
        img_ids = (
            meta['img_info']['id'].cpu().numpy() if isinstance(
                meta['img_info']['id'], torch.Tensor) else
            meta['img_info']['id'])

        for result, img_width, img_height, img_id, warp_matrix in zip(
                result_list, img_widths, img_heights, img_ids, warp_matrixes):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4],
                                           np.linalg.inv(warp_matrix),
                                           img_width, img_height)
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas['img'].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [(math.ceil(input_height / stride),
                          math.ceil(input_width) / stride)
                         for stride in self.strides]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            ) for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[...,
                                                                         2,
                                                                         None]
        bboxes = distance2bbox(
            center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type='nms', iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)
