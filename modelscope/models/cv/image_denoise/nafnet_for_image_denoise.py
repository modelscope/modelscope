# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy
from typing import Any, Dict, Union

import numpy as np
import torch.cuda
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .nafnet.NAFNet_arch import NAFNet, PSNRLoss

logger = get_logger()
__all__ = ['NAFNetForImageDenoise']


@MODELS.register_module(Tasks.image_denoising, module_name=Models.nafnet)
class NAFNetForImageDenoise(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image denoise model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = NAFNet(**self.config.model.network_g)
        self.loss = PSNRLoss()
        self.model = self._load_pretrained(self.model, model_path)

    def _load_pretrained(self,
                         net,
                         load_path,
                         strict=True,
                         param_key='params'):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info(
                    f'Loading: {param_key} does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].'
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
        logger.info('load model done.')
        return net

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        preds = self.model(input)
        return {'loss': self.loss(preds, target)}

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'outputs': self.model(input).clamp(0, 1)}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        preds = self.model(input)
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target, 1, 0))

        preds = [(pred.data * 255.).squeeze(0).permute(
            1, 2, 0).cpu().numpy().astype(np.uint8) for pred in preds]
        targets = [(target.data * 255.).squeeze(0).permute(
            1, 2, 0).cpu().numpy().astype(np.uint8) for target in targets]

        return {'pred': preds, 'target': targets}

    def forward(self, inputs: Dict[str,
                                   Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        if self.training:
            return self._train_forward(**inputs)
        elif 'target' in inputs:
            return self._evaluate_postprocess(**inputs)
        else:
            return self._inference_forward(**inputs)
