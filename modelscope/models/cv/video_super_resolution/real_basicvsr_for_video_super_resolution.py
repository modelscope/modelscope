# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch.cuda
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_super_resolution.common import charbonnier_loss
from modelscope.models.cv.video_super_resolution.real_basicvsr_net import \
    RealBasicVSRNet
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
__all__ = ['RealBasicVSRNetForVideoSR']


@MODELS.register_module(
    Tasks.video_super_resolution, module_name=Models.real_basicvsr)
class RealBasicVSRNetForVideoSR(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video super-resolution model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = RealBasicVSRNet(**self.config.model.generator)
        self.loss = charbonnier_loss
        self.model = self._load_pretrained(self.model, model_path)
        self.max_seq_len = 7

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        preds, lqs = self.model(input, return_lqs=True)

        n, t, c, h, w = target.size()
        target_clean = target.view(-1, c, h, w)
        target_clean = F.interpolate(
            target_clean, scale_factor=0.25, mode='area')
        target_clean = target_clean.view(n, t, c, h // 4, w // 4)

        losses = dict()
        losses['loss_pix'] = self.loss(preds, target)
        losses['loss_clean'] = self.loss(lqs, target_clean)
        return losses

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'output': self.model(input).clamp(0, 1)}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        device = input.device
        input = input.cpu()
        torch.cuda.empty_cache()
        with torch.cuda.amp.autocast():
            outputs = []
            for i in range(0, input.size(1), self.max_seq_len):
                imgs = input[:, i:i + self.max_seq_len, :, :, :]
                imgs = imgs.to(device)
                outputs.append(self.model(imgs).float().cpu())
            preds = torch.cat(outputs, dim=1).squeeze(0)  # (t, c, h, w)
            torch.cuda.empty_cache()
        preds = list(torch.split(preds.clamp(0, 1), 1, 0))  # [(c, h, w), ...]
        targets = list(torch.split(target.clamp(0, 1), 1,
                                   0))  # [(t, c, h, w), ...]

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
