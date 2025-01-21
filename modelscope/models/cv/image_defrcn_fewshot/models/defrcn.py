# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/modeling/meta_arch/rcnn.py

import os
from typing import Dict

import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead
from detectron2.structures import ImageList
from torch import nn

from .gdl import AffineLayer, decouple_layer
from .roi_heads import Res5ROIHeads


class DeFRCN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_resnet_backbone(
            cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self._SHAPE_ = self.backbone.output_shape()

        rpn_config = DeFRCN.from_rpn_config(cfg, self._SHAPE_)
        self.proposal_generator = RPN(**rpn_config)

        self.roi_heads = Res5ROIHeads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(
            num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(
            num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert 'instances' in batched_inputs[0]
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(
            batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get('height', image_size[0])
            width = input.get('width', image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({'instances': r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {
                k: self.affine_rpn(decouple_layer(features[k], scale))
                for k in features
            }
        proposals, proposal_losses = self.proposal_generator(
            images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {
                k: self.affine_rcnn(decouple_layer(features[k], scale))
                for k in features
            }
        results, detector_losses = self.roi_heads(images, features_de_rcnn,
                                                  proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(
                num_channels, 1, 1))
        pixel_std = (
            torch.Tensor(self.cfg.MODEL.PIXEL_STD).to(self.device).view(
                num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std

    @classmethod
    def from_rpn_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            'in_features':
            in_features,
            'min_box_size':
            cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            'nms_thresh':
            cfg.MODEL.RPN.NMS_THRESH,
            'batch_size_per_image':
            cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            'positive_fraction':
            cfg.MODEL.RPN.POSITIVE_FRACTION,
            'loss_weight': {
                'loss_rpn_cls':
                cfg.MODEL.RPN.LOSS_WEIGHT,
                'loss_rpn_loc':
                cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            'anchor_boundary_thresh':
            cfg.MODEL.RPN.BOUNDARY_THRESH,
            'box2box_transform':
            Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            'box_reg_loss_type':
            cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            'smooth_l1_beta':
            cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret['pre_nms_topk'] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
                               cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret['post_nms_topk'] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
                                cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        # ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        anchor_cfg = DefaultAnchorGenerator.from_config(
            cfg, [input_shape[f] for f in in_features])
        ret['anchor_generator'] = DefaultAnchorGenerator(**anchor_cfg)
        ret['anchor_matcher'] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS,
            cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True)
        rpn_head_cfg = {
            'in_channels':
            [s.channels for s in [input_shape[f] for f in in_features]][0],
            'num_anchors':
            ret['anchor_generator'].num_anchors[0],
            'box_dim':
            ret['anchor_generator'].box_dim
        }

        ret['head'] = StandardRPNHead(**rpn_head_cfg)
        return ret
