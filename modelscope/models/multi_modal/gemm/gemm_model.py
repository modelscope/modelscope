# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
""" Generative Multimodal Model Wrapper."""
import os.path as osp
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.gemm.gemm_base import GEMMModel
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['GEMMForMultiModalEmbedding']


@MODELS.register_module(
    Tasks.generative_multi_modal_embedding, module_name=Models.gemm)
class GEMMForMultiModalEmbedding(TorchModel):
    """ Generative multi-modal model for multi-modal embedding
    Inputs could be image or text or both of them.
    Outputs could be features of input image or text,
    image caption could also be produced when image is available.
    """

    def __init__(self, model_dir, device_id=0, *args, **kwargs):
        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)
        self.gemm_model = GEMMModel(model_dir=model_dir)
        pretrained_params = torch.load('{}/{}'.format(
            model_dir, ModelFile.TORCH_MODEL_BIN_FILE))
        self.gemm_model.load_state_dict(pretrained_params)
        self.gemm_model.eval()
        self.device_id = device_id
        if self.device_id >= 0 and torch.cuda.is_available():
            self.gemm_model.to('cuda:{}'.format(self.device_id))
            logger.info('Use GPU: {}'.format(self.device_id))
        else:
            self.device_id = -1
            logger.info('Use CPU for inference')
        self.img_preprocessor = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def parse_image(self, input_img):
        if input_img is None:
            return None
        input_img = LoadImage.convert_to_img(input_img)
        img_tensor = self.img_preprocessor(input_img)[None, ...]
        if self.device_id >= 0:
            img_tensor = img_tensor.to('cuda:{}'.format(self.device_id))
        return img_tensor

    def parse_text(self, text_str):
        if text_str is None or len(text_str) == 0:
            return None
        if isinstance(text_str, str):
            text_ids_tensor = self.gemm_model.tokenize(text_str)
        else:
            raise TypeError(f'text should be str, but got {type(text_str)}')
        if self.device_id >= 0:
            text_ids_tensor = text_ids_tensor.to('cuda:{}'.format(
                self.device_id))
        return text_ids_tensor.view(1, -1)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        image_input = input.get('image', input.get('img', None))
        text_input = input.get('text', input.get('txt', None))
        captioning_input = input.get('captioning', None)
        image = self.parse_image(image_input)
        text = self.parse_text(text_input)
        captioning = captioning_input is True or text_input == ''
        out = self.gemm_model(image, text, captioning)
        output = {
            OutputKeys.IMG_EMBEDDING: out.get('image_feature', None),
            OutputKeys.TEXT_EMBEDDING: out.get('text_feature', None),
            OutputKeys.CAPTION: out.get('caption', None)
        }
        return output
