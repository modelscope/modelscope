# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from copy import deepcopy
from typing import Dict, Union

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .csrnet import CSRNet, L1Loss

logger = get_logger()

__all__ = ['ImageColorEnhance']


@MODELS.register_module(
    Tasks.image_color_enhancement, module_name=Models.csrnet)
class ImageColorEnhance(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image color enhance model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        self.loss = L1Loss()
        self.model = CSRNet()
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.model = self.model.to(self._device)

        self.model = self.load_pretrained(self.model, model_path)

        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def load_pretrained(self, net, load_path, strict=True, param_key='params'):
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

    def _evaluate_postprocess(self, src: Tensor,
                              target: Tensor) -> Dict[str, list]:
        preds = self.model(src)
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target, 1, 0))

        preds = [(pred.data * 255.).squeeze(0).type(torch.uint8).permute(
            1, 2, 0).cpu().numpy() for pred in preds]
        targets = [(target.data * 255.).squeeze(0).type(torch.uint8).permute(
            1, 2, 0).cpu().numpy() for target in targets]

        return {'pred': preds, 'target': targets}

    def _train_forward(self, src: Tensor, target: Tensor) -> Dict[str, Tensor]:
        preds = self.model(src)
        return {'loss': self.loss(preds, target)}

    def _inference_forward(self, src: Tensor) -> Dict[str, Tensor]:
        return {'outputs': self.model(src).clamp(0, 1)}

    def forward(self, input: Dict[str,
                                  Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Union[list, Tensor]]: results
        """
        for key, value in input.items():
            input[key] = input[key].to(self._device)
        if self.training:
            return self._train_forward(**input)
        elif 'target' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self._inference_forward(**input)
