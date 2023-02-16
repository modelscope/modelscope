# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import namedtuple
from math import lgamma

import torch
import torch.nn as nn
from torchvision import models

from modelscope.metainfo import Models
from modelscope.models import MODELS
from modelscope.models.base import TorchModel
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module(Tasks.image_classification, Models.content_check)
class ContentCheckBackbone(TorchModel):

    def __init__(self, *args, **kwargs):
        super(ContentCheckBackbone, self).__init__()
        cc_model = models.resnet50()
        cc_model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        self.model = cc_model

    def forward(self, img):
        x = self.model(img)
        return x

    @classmethod
    def _instantiate(cls, **kwargs):
        model_file = kwargs.get('model_name', ModelFile.TORCH_MODEL_FILE)
        ckpt_path = os.path.join(kwargs['model_dir'], model_file)
        logger.info(f'loading model from {ckpt_path}')
        model_dir = kwargs.pop('model_dir')
        model = cls(**kwargs)
        ckpt_path = os.path.join(model_dir, model_file)
        load_dict = torch.load(ckpt_path, map_location='cpu')
        new_dict = {}
        for k, v in load_dict.items():
            new_dict['model.' + k] = v
        model.load_state_dict(new_dict)
        return model
