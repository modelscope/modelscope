# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import os.path as osp
import pickle

import torch
import torch.nn as nn
import torchvision

from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.cv.tinynas_detection.damo.base_models.backbones import \
    build_backbone
from modelscope.models.cv.tinynas_detection.damo.base_models.heads import \
    build_head
from modelscope.models.cv.tinynas_detection.damo.base_models.necks import \
    build_neck
from modelscope.outputs.cv_outputs import DetectionOutput
from .utils import parse_config


class SingleStageDetector(TorchModel):
    """
    The base class of single stage detector.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """
        init model by cfg
        """
        super().__init__(model_dir, *args, **kwargs)

        config_path = osp.join(model_dir, self.config_name)
        config = parse_config(config_path)
        self.cfg = config
        model_path = osp.join(model_dir, config.model.name)
        label_map = osp.join(model_dir, config.model.class_map)
        self.label_map = pickle.load(open(label_map, 'rb'))
        self.size_divisible = config.dataset.size_divisibility
        self.num_classes = config.model.head.num_classes
        self.conf_thre = config.model.head.nms_conf_thre
        self.nms_thre = config.model.head.nms_iou_thre

        if 'TinyNAS' in self.cfg.model.backbone.name:
            self.cfg.model.backbone.structure_file = osp.join(
                model_dir, self.cfg.model.backbone.structure_file)
        self.backbone = build_backbone(self.cfg.model.backbone)
        self.neck = build_neck(self.cfg.model.neck)
        self.head = build_head(self.cfg.model.head)
        self.head.nms = False
        self.apply(self.init_bn)

        self.load_pretrain_model(model_path)

    def load_pretrain_model(self, pretrain_model):
        ckpt = torch.load(pretrain_model, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=True)

    def init_bn(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        if self.training:
            pass
        else:
            x = self.backbone(x)
            x = self.neck(x)
            cls_scores, bbox_preds = self.head(x)
            prediction = torch.cat(
                [bbox_preds, cls_scores[..., 0:self.num_classes]], dim=-1)
            return prediction

    def postprocess(self, preds):
        bboxes, scores, labels_idx = postprocess_gfocal(
            preds, self.num_classes, self.conf_thre, self.nms_thre)
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels_idx = labels_idx.cpu().numpy()
        labels = [self.label_map[idx + 1][0]['name'] for idx in labels_idx]

        return DetectionOutput(
            boxes=bboxes,
            scores=scores,
            class_ids=labels,
        )


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
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
    num_classes = multi_scores.size(1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr  # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)  # mask->  1000*80*4, 80000*4
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0, ))

        return bboxes, scores, labels

    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]


def postprocess_gfocal(prediction, num_classes, conf_thre=0.05, nms_thre=0.7):
    assert prediction.shape[0] == 1
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        multi_bboxes = image_pred[:, :4]
        multi_scores = image_pred[:, 4:]
        detections, scores, labels = multiclass_nms(multi_bboxes, multi_scores,
                                                    conf_thre, nms_thre, 500)

    return detections, scores, labels
