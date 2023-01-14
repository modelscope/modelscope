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
# from modelscope.models.cv.video_super_resolution.common import charbonnier_loss
from modelscope.models.cv.video_frame_interpolation.VFINet_arch import VFINet
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
__all__ = ['VFINetForVideoFrameInterpolation']


def convert(param):
    return {
        k.replace('module.', ''): v
        for k, v in param.items() if 'module.' in k
    }


@MODELS.register_module(
    Tasks.video_frame_interpolation,
    module_name=Models.video_frame_interpolation)
class VFINetForVideoFrameInterpolation(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video frame-interpolation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        flownet_path = os.path.join(model_dir, 'raft-sintel.pt')
        internet_path = os.path.join(model_dir, 'interpnet.pt')

        self.model = VFINet(self.config.model.network, Ds_flag=True)
        self._load_pretrained(flownet_path, internet_path)

    def _load_pretrained(self, flownet_path, internet_path):
        state_dict_flownet = torch.load(
            flownet_path, map_location=self._device)
        state_dict_internet = torch.load(
            internet_path, map_location=self._device)

        self.model.flownet.load_state_dict(
            convert(state_dict_flownet), strict=True)
        self.model.internet.load_state_dict(
            convert(state_dict_internet), strict=True)
        self.model.internet_Ds.load_state_dict(
            convert(state_dict_internet), strict=True)
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
