# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os
from typing import Any, Dict

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import CLIP, ProbingModel


@MODELS.register_module(
    Tasks.image_classification, module_name=Models.image_probing_model)
class StructuredProbingModel(TorchModel):
    """
    The implementation of 'Structured Model Probing: Empowering
        Efficient Adaptation by Structured Regularization'.
    """

    def __init__(self, model_dir, *args, **kwargs):
        """
        Initialize a probing model.
        Args:
            model_dir: model id or path
        """
        super(StructuredProbingModel, self).__init__()
        model_dir = os.path.join(model_dir, 'food101-clip-vitl14-full.pt')
        model_file = torch.load(model_dir)
        self.feature_size = model_file['meta_info']['feature_size']
        self.num_classes = model_file['meta_info']['num_classes']
        self.backbone = CLIP(
            'CLIP_ViTL14_FP16',
            use_pretrain=True,
            state_dict=model_file['backbone_model_state_dict'])
        self.probing_model = ProbingModel(self.feature_size, self.num_classes)
        self.probing_model.load_state_dict(
            model_file['probing_model_state_dict'])

    def forward(self, x):
        """
        Forward Function of SMP.
        Args:
            x: the input images (B, 3, H, W)
        """

        keys = []
        for idx in range(0, 24):
            keys.append('layer_{}_pre_attn'.format(idx))
            keys.append('layer_{}_attn'.format(idx))
            keys.append('layer_{}_mlp'.format(idx))
        keys.append('pre_logits')
        features = self.backbone(x.half())
        features_agg = []
        for i in keys:
            aggregated_feature = self.aggregate_token(features[i], 1024)
            features_agg.append(aggregated_feature)
        features_agg = torch.cat((features_agg), dim=1)
        outputs = self.probing_model(features_agg.float())
        return outputs

    def aggregate_token(self, output, target_size):
        """
        Aggregating features from tokens.
        Args:
            output: the output of intermidiant features
                from a ViT model
            target_size: target aggregated feature size
        """
        if len(output.shape) == 3:
            _, n_token, channels = output.shape
            if channels >= target_size:
                pool_size = 0
            else:
                n_groups = target_size / channels
                pool_size = int(n_token / n_groups)

            if pool_size > 0:
                output = torch.permute(output, (0, 2, 1))
                output = torch.nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_size)(
                        output)
                output = torch.flatten(output, start_dim=1)
            else:
                output = torch.mean(output, dim=1)
        output = torch.nn.functional.normalize(output, dim=1)
        return output
