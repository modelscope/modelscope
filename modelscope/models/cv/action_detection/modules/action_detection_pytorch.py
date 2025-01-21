# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import postprocessing
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.meta_arch.fcos import FCOS, FCOSHead
from detectron2.structures import (Boxes, ImageList, Instances,
                                   pairwise_point_box_distance)
from fvcore.nn import sigmoid_focal_loss_jit
from torch.nn import functional as F

from modelscope.models.base import TorchModel
from .resnet import Bottleneck3D, ResNet3D

logger = logging.getLogger('detectron2.modelscope.' + __name__)


class ActionDetector(FCOS, TorchModel):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    @torch.no_grad()
    def load_init_backbone(self, path):
        from fvcore.common import checkpoint
        state = torch.load(path, map_location=torch.device('cpu'))
        model_state = state.pop('model')
        prefix = 'backbone.bottom_up.'
        keys = sorted(model_state.keys())
        for k in keys:
            if not k.startswith(prefix):
                model_state.pop(k)
        checkpoint._strip_prefix_if_present(model_state, prefix)
        t = self.backbone.bottom_up.load_state_dict(model_state, strict=False)
        logger.info(str(t))
        logger.info(f'Load pretrained backbone weights from {path}')

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['frames'].to(self.device) for x in batched_inputs]
        images = [x.float() / 255.0 for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def match_anchors(self, anchors: List[Boxes],
                      gt_instances: List[Instances]):
        """
        Match anchors with ground truth boxes.

        Args:
            anchors: #level boxes, from the highest resolution to lower resolution
            gt_instances: ground truth instances per image

        Returns:
            List[Tensor]:
                #image tensors, each is a vector of matched gt
                indices (or -1 for unmatched anchors) for all anchors.
        """
        num_anchors_per_level = [len(x) for x in anchors]
        anchors = Boxes.cat(anchors)  # Rx4
        anchor_centers = anchors.get_centers()  # Rx2
        anchor_sizes = anchors.tensor[:, 2] - anchors.tensor[:, 0]  # R

        lower_bound = anchor_sizes * 4
        lower_bound[:num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1]:] = float('inf')

        matched_indices = []
        for gt_per_image in gt_instances:
            if len(gt_per_image) == 0:
                matched_indices.append(
                    torch.full((len(anchors), ),
                               -1,
                               dtype=torch.int64,
                               device=anchors.tensor.device))
                continue
            gt_centers = gt_per_image.gt_boxes.get_centers()  # Nx2
            # FCOS with center sampling: anchor point must be close enough to gt center.
            center_dist = (anchor_centers[:, None, :]
                           - gt_centers[None, :, :]).abs_().max(dim=2).values
            pairwise_match = center_dist < self.center_sampling_radius * anchor_sizes[:,
                                                                                      None]
            pairwise_dist = pairwise_point_box_distance(
                anchor_centers, gt_per_image.gt_boxes)

            # The original FCOS anchor matching rule: anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # Multilevel anchor matching in FCOS: each anchor is only responsible
            # for certain scale range.
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (
                pairwise_dist < upper_bound[:, None])

            # Match the GT box with minimum area, if there are multiple GT matches
            gt_areas = gt_per_image.gt_boxes.area()  # N
            pairwise_match = pairwise_match.to(
                torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(
                dim=1)  # R, per-anchor match
            matched_idx[
                min_values < 1e-5] = -1  # Unmatched anchors are assigned -1

            matched_indices.append(matched_idx)
        return matched_indices

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas,
               gt_boxes, pred_centerness):
        """
        This method is almost identical to :meth:`RetinaNet.losses`, with an extra
        "loss_centerness" in the returned dict.
        """
        gt_labels = torch.stack(gt_labels)  # (N, R)

        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        normalizer = self._ema_update('loss_normalizer',
                                      max(num_pos_anchors, 1), 300)

        # classification and regression loss
        gt_labels_target = F.one_hot(
            gt_labels, num_classes=self.num_classes
            + 1)[:, :, :-1]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            torch.cat(pred_logits, dim=1),
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction='sum',
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            [x.tensor for x in gt_boxes],
            pos_mask,
            box_reg_loss_type='giou',
        )

        ctrness_targets = self.compute_ctrness_targets(anchors,
                                                       gt_boxes)  # NxR
        pred_centerness = torch.cat(
            pred_centerness, dim=1).squeeze(dim=2)  # NxR
        ctrness_loss = F.binary_cross_entropy_with_logits(
            pred_centerness[pos_mask],
            ctrness_targets[pos_mask],
            reduction='sum')
        return {
            'loss_fcos_cls': loss_cls / normalizer,
            'loss_fcos_loc': loss_box_reg / normalizer,
            'loss_fcos_ctr': ctrness_loss / normalizer,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS
        anchor matching rule.

        Unlike RetinaNet, there are no ignored anchors.
        """
        matched_indices = self.match_anchors(anchors, gt_instances)

        matched_labels, matched_boxes = [], []
        for gt_index, gt_per_image in zip(matched_indices, gt_instances):
            if len(gt_per_image) > 0:
                label = gt_per_image.gt_classes[gt_index.clip(min=0)]
                matched_gt_boxes = gt_per_image.gt_boxes[gt_index.clip(min=0)]
            else:
                label = gt_per_image.gt_classes.new_zeros((len(gt_index), ))
                matched_gt_boxes = Boxes(
                    gt_per_image.gt_boxes.tensor.new_zeros((len(gt_index), 4)))
            label[gt_index < 0] = self.num_classes  # background
            matched_labels.append(label)
            matched_boxes.append(matched_gt_boxes)
        return matched_labels, matched_boxes

    def compute_ctrness_targets(self, anchors, gt_boxes):  # NxR
        anchors = Boxes.cat(anchors).tensor  # Rx4
        reg_targets = [
            self.box2box_transform.get_deltas(anchors, m.tensor)
            for m in gt_boxes
        ]
        reg_targets = torch.stack(reg_targets, dim=0)  # NxRx4
        if len(reg_targets) == 0:
            # return reg_targets.new_zeros(len(reg_targets))
            return reg_targets.new_zeros(reg_targets.size()[:-1])
        left_right = reg_targets[:, :, [0, 2]]
        top_bottom = reg_targets[:, :, [1, 3]]
        ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(ctrness)


def build_action_detection_model(num_classes, device='cpu'):
    backbone = ResNet3D(
        Bottleneck3D, [3, 4, 6, 3],
        ops=['c2d', 'p3d'] * 8,
        t_stride=[1, 1, 1, 1, 1],
        num_classes=None)
    in_features = ['res3', 'res4', 'res5']
    out_channels = 512
    top_block = LastLevelP6P7(out_channels, out_channels, in_feature='p5')
    fpnbackbone = FPN(
        bottom_up=backbone,
        in_features=in_features,
        out_channels=out_channels,
        top_block=top_block,
    )
    head = FCOSHead(
        input_shape=[ShapeSpec(channels=out_channels)] * 5,
        conv_dims=[out_channels] * 2,
        num_classes=num_classes)
    model = ActionDetector(
        backbone=fpnbackbone,
        head=head,
        num_classes=num_classes,
        pixel_mean=[0, 0, 0],
        pixel_std=[0, 0, 0])
    return model
