# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_deblur import NAFNetForImageDeblur
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import ImageDeblurPreprocessor, LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ImageDeblurPipeline']


@PIPELINES.register_module(
    Tasks.image_deblurring, module_name=Pipelines.image_deblur)
class ImageDeblurPipeline(Pipeline):
    """

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> from modelscope.outputs import OutputKeys
    >>> import cv2
    >>>
    >>> img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/blurry.jpg'
    >>> image_deblur_pipeline = pipeline(Tasks.image_deblurring, 'damo/cv_nafnet_image-deblur_gopro')
    >>> result = image_deblur_pipeline(img)[OutputKeys.OUTPUT_IMG]
    >>> cv2.imwrite('result.png', result)
    """

    def __init__(self,
                 model: Union[NAFNetForImageDeblur, str],
                 preprocessor: Optional[ImageDeblurPreprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a cv image deblur pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        self.config = self.model.config

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        logger.info('load image denoise model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        result = {'img': img.unsqueeze(0).to(self._device)}
        return result

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
                       crop_x:crop_x + crop_w] = self.model._inference_forward(
                           crop_frames)['outputs'][:, :, h_start:h_end,
                                                   w_start:w_end]
        return output

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        def set_phase(model, is_train):
            if is_train:
                model.train()
            else:
                model.eval()

        is_train = False
        set_phase(self.model, is_train)
        with torch.no_grad():
            output = self.crop_process(input['img'])  # output Tensor

        return {'output_tensor': output}

    def postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        output_img = (input['output_tensor'].squeeze(0) * 255).cpu().permute(
            1, 2, 0).numpy().astype('uint8')
        return {OutputKeys.OUTPUT_IMG: output_img[:, :, ::-1]}
