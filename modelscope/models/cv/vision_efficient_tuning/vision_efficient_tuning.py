# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile


class VisionEfficientTuning(nn.Module):
    """ The implementation of vision efficient tuning.

    This model is constructed with the following parts:
        - 'backbone': pre-trained backbone model with parameters.
        - 'head': classification head with fine-tuning.
        - 'loss': loss function for training.
    """

    def __init__(self,
                 backbone=None,
                 head=None,
                 loss=None,
                 pretrained=True,
                 finetune=False,
                 **kwargs):
        """ Initialize a vision efficient tuning model.

        Args:
          backbone: config of backbone.
          head: config of head.
          loss: config of loss.
          pretrained: whether to load the pretrained model.
          finetune: whether to finetune the model.
        """
        from .backbone import VisionTransformerPETL
        from .head import ClassifierHead

        super(VisionEfficientTuning, self).__init__()

        if backbone and 'type' in backbone:
            backbone.pop('type')
            self.backbone = VisionTransformerPETL(**backbone)
        else:
            self.backbone = None

        # TODO Use a more elegant method to build the model.
        if head and 'type' in head:
            head.pop('type')
            self.head = ClassifierHead(**head)
        else:
            self.head = None

        if loss and 'type' in loss:
            self.loss = getattr(torch.nn, loss['type'])()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.CLASSES = kwargs.pop('CLASSES', None)
        self.pretrained_cfg = kwargs.pop('pretrained_cfg', None)

        if pretrained:
            assert 'model_dir' in kwargs, 'pretrained model dir is missing.'
            model_path = os.path.join(kwargs['model_dir'],
                                      ModelFile.TORCH_MODEL_FILE)
            model_dict = torch.load(model_path, map_location='cpu')

            if self.backbone is None and 'backbone_cfg' in model_dict:
                model_dict['backbone_cfg'].pop('type')
                self.backbone = VisionTransformerPETL(
                    **model_dict['backbone_cfg'])
            if self.head is None and 'head_cfg' in model_dict:
                model_dict['head_cfg'].pop('type')
                self.head = ClassifierHead(**model_dict['head_cfg'])

            if 'backbone_weight' in model_dict:
                backbone_weight = model_dict['backbone_weight']
                if finetune and self.pretrained_cfg and 'unload_part' in self.pretrained_cfg \
                   and 'backbone' in self.pretrained_cfg['unload_part']:
                    backbone_weight = self.filter_weight(
                        backbone_weight,
                        self.pretrained_cfg['unload_part']['backbone'])
                self.backbone.load_state_dict(backbone_weight, strict=False)

            if 'head_weight' in model_dict:
                head_weight = model_dict['head_weight']
                if finetune and self.pretrained_cfg and 'unload_part' in self.pretrained_cfg \
                   and 'head' in self.pretrained_cfg['unload_part']:
                    head_weight = self.filter_weight(
                        head_weight,
                        self.pretrained_cfg['unload_part']['head'])
                self.head.load_state_dict(head_weight, strict=False)

            self.CLASSES = model_dict[
                'CLASSES'] if 'CLASSES' in model_dict else self.CLASSES

    def filter_weight(self, weights, unload_part=[]):
        """ Filter parameters that the model does not need to load.

        Args:
          weights: the parameters of the model.
          unload_part: the config of unloading parameters.
        """
        ret_dict = {}
        for key, value in weights.items():
            flag = sum([p in key for p in unload_part]) > 0
            if not flag:
                ret_dict[key] = value
        return ret_dict

    def forward(self, imgs, labels=None, **kwargs):
        """ Dynamic forward function of vision efficient tuning.

        Args:
            imgs: (B, 3, H, W).
            labels: (B), when training stage.
        """
        return self.forward_train(imgs, labels, **kwargs) \
            if self.training else self.forward_test(imgs, labels, **kwargs)

    def forward_train(self, imgs, labels=None):
        """ Dynamic forward function of training stage.

        Args:
            imgs: (B, 3, H, W).
            labels: (B), when training stage.
        """
        output = OrderedDict()

        backbone_output = self.backbone(imgs)
        head_output = self.head(backbone_output)
        loss = self.loss(head_output, labels)

        output = {OutputKeys.LOSS: loss}
        return output

    def forward_test(self, imgs, labels=None):
        """ Dynamic forward function of testing stage.

        Args:
            imgs: (B, 3, H, W).
            labels: (B), when training stage.
        """
        output = OrderedDict()
        backbone_output = self.backbone(imgs)
        head_output = self.head(backbone_output)

        scores = F.softmax(head_output, dim=1)
        preds = scores.topk(1, 1, True, True)[-1].squeeze(-1)

        output = {OutputKeys.SCORES: scores, OutputKeys.LABELS: preds}
        return output
