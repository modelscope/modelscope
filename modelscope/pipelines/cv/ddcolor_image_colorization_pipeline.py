# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_colorization import DDColorForImageColorization
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_colorization, module_name=Pipelines.ddcolor_image_colorization)
class DDColorImageColorizationPipeline(Pipeline):
    """ DDColor Image Colorization Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> colorizer = pipeline('image-colorization', 'damo/cv_ddcolor_image-colorization')
    >>> colorizer("data/test/images/audrey_hepburn.jpg")
       {'output_img': array([[[198, 199, 193],
         [198, 199, 193],
         [197, 199, 195],
         ...,
         [197, 213, 206],
         [197, 213, 206],
         [197, 213, 207]],

        [[198, 199, 193],
         [198, 199, 193],
         [197, 199, 195],
         ...,
         [196, 212, 205],
         [196, 212, 205],
         [196, 212, 206]],

        [[198, 199, 193],
         [198, 199, 193],
         [197, 199, 195],
         ...,
         [193, 209, 202],
         [193, 209, 202],
         [193, 209, 203]],

        ...,

        [[ 56,  72, 103],
         [ 56,  72, 103],
         [ 56,  72, 102],
         ...,
         [233, 231, 232],
         [233, 231, 232],
         [233, 231, 232]],

        [[ 51,  62,  91],
         [ 52,  63,  92],
         [ 52,  64,  92],
         ...,
         [233, 232, 231],
         [233, 232, 231],
         [232, 232, 229]],

        [[ 60,  72, 101],
         [ 59,  71, 100],
         [ 57,  70,  99],
         ...,
         [233, 232, 231],
         [233, 232, 231],
         [232, 232, 229]]], dtype=uint8)}
    """

    def __init__(self, model: Union[DDColorForImageColorization, str],
                 **kwargs):
        """
        use `model` to create an image colorization pipeline for prediction

        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.model.eval()
        self.input_size = 512
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        # self.model = DDColorForImageColorization(
        #     model_dir=model,
        #     encoder_name='convnext-l',
        #     input_size=[self.input_size, self.input_size],
        # ).to(self.device)

        # model_path = f'{model}/{ModelFile.TORCH_MODEL_FILE}'
        # logger.info(f'loading model from {model_path}')
        # self.model.load_state_dict(
        #     torch.load(model_path, map_location=torch.device('cpu'))['params'],
        #     strict=True)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        """preprocess the input image, extract L-channel and convert it back to RGB

        Args:
            inputs: an input image from file or url

        Returns:
            Dict[str, Any]: the pre-processed image
        """
        img = LoadImage.convert_to_ndarray(input)
        self.height, self.width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        self.orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose(
            (2, 0, 1))).float()
        tensor_gray_rgb = tensor_gray_rgb.unsqueeze(0).to(self.device)

        result = {'img': tensor_gray_rgb}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """call model to output the predictions and concatenate it with the original L-channel

        Args:
            inputs: input image tensor

        Returns:
            Dict[str, Any]: the result image
        """

        output_ab = self.model(input).cpu()

        output_ab_resize = F.interpolate(
            output_ab, size=(self.height, self.width))
        output_ab_resize = output_ab_resize[0].float().numpy().transpose(
            1, 2, 0)
        out_lab = np.concatenate((self.orig_l, output_ab_resize), axis=-1)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        output_img = (out_bgr * 255.0).round().astype(np.uint8)

        return {OutputKeys.OUTPUT_IMG: output_img}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
