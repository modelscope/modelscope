from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.cv.image_color_enhance.image_color_enhance import \
    ImageColorEnhance
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input
from modelscope.preprocessors import (ImageColorEnhanceFinetunePreprocessor,
                                      load_image)
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from ..base import Pipeline
from ..builder import PIPELINES

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_color_enhance, module_name=Pipelines.image_color_enhance)
class ImageColorEnhancePipeline(Pipeline):

    def __init__(self,
                 model: Union[ImageColorEnhance, str],
                 preprocessor: Optional[
                     ImageColorEnhanceFinetunePreprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        model = model if isinstance(
            model, ImageColorEnhance) else Model.from_pretrained(model)
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            img = load_image(input)
        elif isinstance(input, PIL.Image.Image):
            img = input.convert('RGB')
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')

        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        result = {'src': img.unsqueeze(0).to(self._device)}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return super().forward(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_img = (inputs['outputs'].squeeze(0) * 255.).type(
            torch.uint8).cpu().permute(1, 2, 0).numpy()
        return {OutputKeys.OUTPUT_IMG: output_img}
