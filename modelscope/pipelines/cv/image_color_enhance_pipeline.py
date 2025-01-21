# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (ImageColorEnhanceFinetunePreprocessor,
                                      LoadImage)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_color_enhancement,
    module_name=Pipelines.adaint_image_color_enhance)
@PIPELINES.register_module(
    Tasks.image_color_enhancement,
    module_name=Pipelines.deeplpf_image_color_enhance)
@PIPELINES.register_module(
    Tasks.image_color_enhancement, module_name=Pipelines.image_color_enhance)
class ImageColorEnhancePipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, 'AdaIntImageColorEnhance',
                              'DeepLPFImageColorEnhance', 'ImageColorEnhance',
                              str],
                 preprocessor: Optional[
                     ImageColorEnhanceFinetunePreprocessor] = None,
                 **kwargs):
        """The inference pipeline for image color enhance.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Example:
            >>> import cv2
            >>> from modelscope.outputs import OutputKeys
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope.utils.constant import Tasks

            >>> img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_color_enhance.png'
                image_color_enhance = pipeline(Tasks.image_color_enhancement,
                    model='damo/cv_deeplpfnet_image-color-enhance-models')
                result = image_color_enhance(img)
            >>> cv2.imwrite('enhanced_result.png', result[OutputKeys.OUTPUT_IMG])
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
