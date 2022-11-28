# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.cv.image_color_enhance import ImageColorEnhance
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (ImageColorEnhanceFinetunePreprocessor,
                                      LoadImage)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_color_enhancement, module_name=Pipelines.image_color_enhance)
class ImageColorEnhancePipeline(Pipeline):

    def __init__(self,
                 model: Union[ImageColorEnhance, str],
                 preprocessor: Optional[
                     ImageColorEnhanceFinetunePreprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a image color enhance pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        result = {'src': img.unsqueeze(0).to(self._device)}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return super().forward(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_img = (inputs['outputs'].squeeze(0) * 255.).type(
            torch.uint8).cpu().permute(1, 2, 0).numpy()[:, :, ::-1]
        return {OutputKeys.OUTPUT_IMG: output_img}
