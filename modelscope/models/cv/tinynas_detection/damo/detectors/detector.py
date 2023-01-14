# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.cv.tinynas_detection.damo.base_models.backbones import \
    build_backbone
from modelscope.models.cv.tinynas_detection.damo.base_models.heads import \
    build_head
from modelscope.models.cv.tinynas_detection.damo.base_models.necks import \
    build_neck
from modelscope.models.cv.tinynas_detection.damo.structures.image_list import \
    to_image_list
from modelscope.utils.logger import get_logger

logger = get_logger()


class Detector(TorchModel):

    def __init__(self, config):
        super().__init__()

        self.backbone = build_backbone(config.model.backbone)
        self.neck = build_neck(config.model.neck)
        self.head = build_head(config.model.head)

        self.config = config

    def init_bn(self, M):

        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def init_model(self):

        self.apply(self.init_bn)

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def load_pretrain_detector(self, pretrain_model):
        ckpt = torch.load(pretrain_model, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        logger.info(f'Finetune from {pretrain_model}................')
        new_state_dict = {}
        for k, v in self.state_dict().items():
            k = k.replace('module.', '')
            if 'head' in k:
                new_state_dict[k] = self.state_dict()[k]
                continue
            new_state_dict[k] = state_dict[k]

        self.load_state_dict(new_state_dict, strict=True)

    def forward(self, x, targets=None, tea=False, stu=False):
        images = to_image_list(x)
        feature_outs = self.backbone(images.tensors)  # list of tensor
        fpn_outs = self.neck(feature_outs)

        if tea:
            return fpn_outs
        else:
            outputs = self.head(
                fpn_outs,
                targets,
                imgs=images,
            )
            if stu:
                return outputs, fpn_outs
            else:
                return outputs


def build_local_model(config, device):
    model = Detector(config)
    model.init_model()
    model.to(device)

    return model


def build_ddp_model(model, local_rank):
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False)

    return model
