# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy
from typing import Any, Dict, Union

import torch.cuda
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_deinterlace.deinterlace_arch import \
    DeinterlaceNet
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
__all__ = ['UNetForVideoDeinterlace']


def convert(param):
    return {
        k.replace('module.', ''): v
        for k, v in param.items() if 'module.' in k
    }


@MODELS.register_module(
    Tasks.video_deinterlace, module_name=Models.video_deinterlace)
class UNetForVideoDeinterlace(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video deinterlace model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.model_dir = model_dir

        frenet_path = os.path.join(model_dir, 'deinterlace_fre.pth')
        enhnet_path = os.path.join(model_dir, 'deinterlace_mf.pth')

        self.model = DeinterlaceNet()
        self._load_pretrained(frenet_path, enhnet_path)

    def _load_pretrained(self, frenet_path, enhnet_path):
        state_dict_frenet = torch.load(frenet_path, map_location=self._device)
        state_dict_enhnet = torch.load(enhnet_path, map_location=self._device)

        self.model.frenet.load_state_dict(state_dict_frenet, strict=True)
        self.model.enhnet.load_state_dict(state_dict_enhnet, strict=True)
        logger.info('load model done.')

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'output': self.model(input)}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        preds = self.model(input)
        del input
        torch.cuda.empty_cache()
        return {'pred': preds, 'target': target}

    def forward(self, inputs: Dict[str,
                                   Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """

        if 'target' in inputs:
            return self._evaluate_postprocess(**inputs)
        else:
            return self._inference_forward(**inputs)
