# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.face_emotion.efficient import EfficientNet
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module(Tasks.face_emotion, module_name=Models.face_emotion)
class EfficientNetForFaceEmotion(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)
        self.model = FaceEmotionModel(
            name='efficientnet-b0', num_embed=512, num_au=12, num_emotion=7)

        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')
        pretrained_params = torch.load(
            '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location=self.device)

        state_dict = pretrained_params['model']
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state[k] = v

        self.model.load_state_dict(new_state)
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        logits_au, logits_emotion = self.model(x)
        return logits_au, logits_emotion


class FaceEmotionModel(nn.Module):

    def __init__(self,
                 name='efficientnet-b0',
                 num_embed=512,
                 num_au=12,
                 num_emotion=7):
        super(FaceEmotionModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained(
            name, weights_path=None, advprop=True)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(self.backbone._fc.weight.data.shape[1],
                               num_embed)
        self.features = nn.BatchNorm1d(num_embed)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.fc_au = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_embed, num_au),
        )
        self.fc_emotion = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_embed, num_emotion),
        )

    def feat_single_img(self, x):
        x = self.backbone.extract_features(x)
        x = self.average_pool(x)
        x = x.flatten(1)
        x = self.embed(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.feat_single_img(x)
        logits_au = self.fc_au(x)
        att_au = torch.sigmoid(logits_au).unsqueeze(-1)
        x = x.unsqueeze(1)
        emotion_vec_list = torch.matmul(att_au, x)
        emotion_vec = emotion_vec_list.sum(1)
        logits_emotion = self.fc_emotion(emotion_vec)
        return logits_au, logits_emotion
