from typing import Any, Dict

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tokenizers import BertWordPieceTokenizer
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.clip.clip_bert import TextTransformer
from modelscope.models.multi_modal.clip.clip_vit import VisionTransformer
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['CLIPForMultiModalEmbedding']


class CLIPModel(nn.Module):

    def __init__(self, model_dir):
        super(CLIPModel, self).__init__()
        # including vision config and text config
        model_config = json.load(
            open('{}/encoder_config.json'.format(model_dir)))

        # vision encoder
        vision_config = model_config['vision_config']
        self.img_size = vision_config['input_resolution']
        self.vision_encoder = VisionTransformer(
            input_resolution=self.img_size,
            patch_size=vision_config['patch_size'],
            width=vision_config['width'],
            layers=vision_config['layers'],
            heads=vision_config['heads'],
            output_dim=vision_config['feat_dim'])

        # text encoder
        text_config = model_config['text_config']
        self.text_encoder = TextTransformer(
            text_config['bert_config'], feat_dim=text_config['feat_dim'])

    def forward(self, input_data, input_type):
        if input_type == 'img':
            img_embedding = self.vision_encoder(input_data)
            img_embedding = F.normalize(img_embedding, p=2.0, dim=1)
            return img_embedding
        elif input_type == 'text':
            text_ids_tensor, text_mask_tensor = input_data
            text_embedding = self.text_encoder(text_ids_tensor,
                                               text_mask_tensor)
            text_embedding = F.normalize(text_embedding, p=2.0, dim=1)
            return text_embedding
        else:
            raise ValueError('Unknown input type')


@MODELS.register_module(Tasks.multi_modal_embedding, module_name=Models.clip)
class CLIPForMultiModalEmbedding(Model):

    def __init__(self, model_dir, device_id=-1):
        super().__init__(model_dir=model_dir, device_id=device_id)
        self.clip_model = CLIPModel(model_dir=model_dir)
        pretrained_params = torch.load(
            '{}/pytorch_model.bin'.format(model_dir), 'cpu')
        self.clip_model.load_state_dict(pretrained_params)
        self.clip_model.eval()

        self.device_id = device_id
        if self.device_id >= 0:
            self.clip_model.to('cuda:{}'.format(self.device_id))
            logger.info('Use GPU: {}'.format(self.device_id))
        else:
            logger.info('Use CPU for inference')

        # image preprocessor
        norm_op = Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
        self.img_preprocessor = Compose([
            Resize((self.clip_model.img_size, self.clip_model.img_size),
                   interpolation=Image.BICUBIC),
            ToTensor(), norm_op
        ])

        # text tokenizer
        vocab_path = '{}/vocab.txt'.format(model_dir)
        self.text_tokenizer = BertWordPieceTokenizer(
            vocab_path, lowercase=False)
        self.text_tokenizer.enable_truncation(max_length=30)

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
        from modelscope.outputs import OutputKeys
        output = {
            OutputKeys.IMG_EMBEDDING: None,
            OutputKeys.TEXT_EMBEDDING: None
        }
        if 'img' in input and input['img'] is not None:
            input_img = input['img']
            if isinstance(input_img, Image.Image):
                img_tensor = self.img_preprocessor(input_img)[None, ...]
            elif isinstance(input_img, np.ndarray):
                if len(input_img.shape) == 2:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
                input_img = input_img[:, :, ::-1]  # in rgb order
                input_img = Image.fromarray(
                    input_img.astype('uint8')).convert('RGB')
                img_tensor = self.img_preprocessor(input_img)[None, ...]
            else:
                raise TypeError(
                    f'img should be either PIL.Image or np.array, but got {type(input_img)}'
                )

            if self.device_id >= 0:
                img_tensor = img_tensor.to('cuda:{}'.format(self.device_id))

            img_embedding = self.clip_model(
                input_data=img_tensor, input_type='img')
            from modelscope.outputs import OutputKeys
            output[OutputKeys.IMG_EMBEDDING] = img_embedding.data.cpu().numpy()

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

            text_embedding = self.clip_model(
                input_data=(text_ids_tensor, text_mask_tensor),
                input_type='text')
            output['text_embedding'] = text_embedding.data.cpu().numpy()

        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
