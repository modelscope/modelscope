# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch.cuda

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_denoise.nafnet.NAFNet_arch import (NAFNet,
                                                                   PSNRLoss)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
__all__ = ['NAFNetForImageDeblur']


@MODELS.register_module(Tasks.image_deblurring, module_name=Models.nafnet)
class NAFNetForImageDeblur(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image deblur model from the `model_dir` path.

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

    def crop_process(self, input):
        output = torch.zeros_like(input)  # [1, C, H, W]
        # determine crop_h and crop_w
        ih, iw = input.shape[-2:]
        crop_rows, crop_cols = max(ih // 512, 1), max(iw // 512, 1)
        overlap = 16

        step_h, step_w = ih // crop_rows, iw // crop_cols
        for y in range(crop_rows):
            for x in range(crop_cols):
                crop_y = step_h * y
                crop_x = step_w * x

                crop_h = step_h if y < crop_rows - 1 else ih - crop_y
                crop_w = step_w if x < crop_cols - 1 else iw - crop_x

                crop_frames = input[:, :,
                                    max(0, crop_y - overlap
                                        ):min(crop_y + crop_h + overlap, ih),
                                    max(0, crop_x - overlap
                                        ):min(crop_x + crop_w
                                              + overlap, iw)].contiguous()
                h_start = overlap if max(0, crop_y - overlap) > 0 else 0
                w_start = overlap if max(0, crop_x - overlap) > 0 else 0
                h_end = h_start + crop_h if min(crop_y + crop_h
                                                + overlap, ih) < ih else ih
                w_end = w_start + crop_w if min(crop_x + crop_w
                                                + overlap, iw) < iw else iw

                output[:, :, crop_y:crop_y + crop_h,
                       crop_x:crop_x + crop_w] = self.model(
                           crop_frames)[:, :, h_start:h_end,
                                        w_start:w_end].clamp(0, 1)
        return output

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        preds = self.model(input)
        return {'loss': self.loss(preds, target)}

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'outputs': self.crop_process(input).cpu()}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        preds = self.crop_process(input).cpu()
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target.cpu(), 1, 0))

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
