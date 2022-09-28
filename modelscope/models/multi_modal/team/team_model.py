# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tokenizers import BertWordPieceTokenizer
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .utils import TEAM, BertWrapper, CLIPVisionWrapper, CrossLayer

logger = get_logger()

__all__ = ['TEAMForMultiModalSimilarity']


@MODELS.register_module(Tasks.multi_modal_similarity, module_name=Models.team)
class TEAMForMultiModalSimilarity(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):
        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        text_model = BertWrapper(
            config_json='{}/text_config.json'.format(model_dir),
            feat_dim=768,
            token_dim=1024)
        text_model.bert.cls = None
        image_model = CLIPVisionWrapper()

        self.model = TEAM(
            text_model,
            image_model,
            pretrained='{}/{}'.format(model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE))
        self.model.eval()

        self.device_id = device_id
        if self.device_id >= 0 and torch.cuda.is_available():
            self.model.to('cuda:{}'.format(self.device_id))
            logger.info('Use GPU: {}'.format(self.device_id))
        else:
            self.device_id = -1
            logger.info('Use CPU for inference')

        self.text_tokenizer = BertWordPieceTokenizer(
            '{}/{}'.format(model_dir, ModelFile.VOCAB_FILE), lowercase=False)
        self.text_tokenizer.enable_truncation(max_length=30)

        norm_op = Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
        self.img_preprocessor = Compose([
            Resize((224, 224), interpolation=Image.BICUBIC),
            ToTensor(), norm_op
        ])

    def tokenize_text(self, text_str):
        tokens = self.text_tokenizer.encode(text_str)
        max_tokens = 30
        text_ids_tensor = torch.zeros((1, max_tokens)).long()
        text_mask_tensor = torch.zeros((1, max_tokens))
        text_ids, text_mask = tokens.ids, tokens.attention_mask
        text_ids_tensor[0, 0:len(text_ids)] = torch.tensor(text_ids)
        text_mask_tensor[0, 0:len(text_mask)] = torch.tensor(text_mask)
        return text_ids_tensor, text_mask_tensor

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            if 'img' in input and input['img'] is not None:
                input_img = input['img']
                input_img = LoadImage.convert_to_img(input_img)
                img_tensor = self.img_preprocessor(input_img)[None, ...]

                if self.device_id >= 0:
                    img_tensor = img_tensor.to('cuda:{}'.format(
                        self.device_id))
                _, _, image_feature, image_tensors = self.model.get_feature(
                    None, None, img_tensor)
                image_feature = image_feature.cpu().numpy()
            else:
                image_feature, image_tensors = None, None

            if 'text' in input and input['text'] is not None:
                text_str = input['text']
                if isinstance(text_str, str):
                    text_ids_tensor, text_mask_tensor = self.tokenize_text(
                        text_str)
                else:
                    raise TypeError(
                        f'text should be str, but got {type(text_str)}')

                if self.device_id >= 0:
                    text_ids_tensor = text_ids_tensor.to('cuda:{}'.format(
                        self.device_id))
                    text_mask_tensor = text_mask_tensor.to('cuda:{}'.format(
                        self.device_id))
                text_feature, text_tensors, _, _ = self.model.get_feature(
                    text_ids_tensor, text_mask_tensor, None)
                text_feature = text_feature.cpu().numpy()
            else:
                text_tensors, text_mask_tensor = None, None

            if text_tensors is not None and text_mask_tensor is not None and image_tensors is not None:
                score = self.model.get_cross_score(text_tensors,
                                                   text_mask_tensor,
                                                   image_tensors)[0].item()
            else:
                score = None
            output = {
                OutputKeys.IMG_EMBEDDING: image_feature,
                OutputKeys.TEXT_EMBEDDING: text_feature,
                OutputKeys.SCORES: score
            }
            return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
