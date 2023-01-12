# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .models.defaults_config import _C
from .models.defrcn import DeFRCN
from .utils.requirements_check import requires_version

logger = get_logger()
__all__ = ['DeFRCNForFewShot']


@MODELS.register_module(
    Tasks.image_fewshot_detection, module_name=Models.defrcn)
class DeFRCNForFewShot(TorchModel):
    """ Few-shot object detection model DeFRCN. The model requires detectron2-0.3 and pytorch-1.11.
        Model config params mainly from detectron2, you can use detectron2 config file to initialize model.
        Detail configs can be visited on detectron2.config.defaults and .models.defaults_config.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the few-shot defrcn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        requires_version()

        super().__init__(model_dir, *args, **kwargs)

        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        if 'config_path' in kwargs:
            self.config.merge_from_dict(
                {'model.config_path': kwargs['config_path']})

        self.model_cfg = _C.clone()
        self.model_cfg.merge_from_file(
            os.path.join(model_dir, self.config.model.config_path))

        if 'model_weights' in kwargs:
            self.model_cfg.merge_from_list(
                ['MODEL.WEIGHTS', kwargs['model_weights']])

        self.model_cfg.freeze()

        self.model = DeFRCN(self.model_cfg)

    def forward(self, inputs) -> Any:
        """return the result by the model

        Args:
            inputs (list): the preprocessed data

        Returns:
            Any: results
        """
        if self.training:
            return self.model.forward(inputs)
        else:
            return self.model.inference(inputs)

    def inference(self, input: Dict[str, Any]) -> Any:
        with torch.no_grad():
            results = self.model([input])
        return results[0] if len(results) > 0 else None

    def get_model_cfg(self):
        return self.model_cfg
