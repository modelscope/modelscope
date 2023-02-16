# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.vision_efficient_tuning, module_name=Models.vision_efficient_tuning)
class VisionEfficientTuningModel(TorchModel):
    """ The implementation of vision efficient tuning.

    This model is constructed with the following parts:
        - 'backbone': pre-trained backbone model with parameters.
        - 'head': classification head with fine-tuning.
    """

    def __init__(self, model_dir: str, **kwargs):
        """ Initialize a vision efficient tuning model.

        Args:
          model_dir: model id or path, where model_dir/pytorch_model.pt contains:
                    - 'backbone_cfg': config of backbone.
                    - 'backbone_weight': parameters of backbone.
                    - 'head_cfg': config of head.
                    - 'head_weight': parameters of head.
                    - 'CLASSES': list of label name.
        """

        from .backbone import VisionTransformerPETL
        from .head import ClassifierHead
        super().__init__(model_dir)

        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        model_dict = torch.load(model_path)

        backbone_cfg = model_dict['backbone_cfg']
        if 'type' in backbone_cfg:
            backbone_cfg.pop('type')
        self.backbone_model = VisionTransformerPETL(**backbone_cfg)
        self.backbone_model.load_state_dict(
            model_dict['backbone_weight'], strict=True)

        head_cfg = model_dict['head_cfg']
        if 'type' in head_cfg:
            head_cfg.pop('type')
        self.head_model = ClassifierHead(**head_cfg)
        self.head_model.load_state_dict(model_dict['head_weight'], strict=True)

        self.CLASSES = model_dict['CLASSES']

    def forward(self, inputs):
        """ Dynamic forward function of vision efficient tuning.

        Args:
          inputs: the input images (B, 3, H, W).
        """

        backbone_output = self.backbone_model(inputs)
        head_output = self.head_model(backbone_output)
        return head_output
