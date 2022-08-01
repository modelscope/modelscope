from typing import Any, Dict

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tokenizers import BertWordPieceTokenizer
from torch.distributed.nn.functional import \
    all_gather as all_gather_with_backprop
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.clip.clip_bert import TextTransformer
from modelscope.models.multi_modal.clip.clip_vit import VisionTransformer
from modelscope.utils.constant import ModeKeys, Tasks
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
            output_dim=vision_config['feat_dim'],
            use_grad_ckp=True)

        # text encoder
        text_config = model_config['text_config']
        self.text_encoder = TextTransformer(
            text_config['bert_config'], feat_dim=text_config['feat_dim'])

        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6)

    def contrastive_loss(self, logits, dim):
        neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
        return -neg_ce.mean()

    def clip_loss(self, t2i_sim, i2t_sim, img_idx=None, all_img_idx=None):
        if img_idx is not None and all_img_idx is not None:
            with torch.no_grad():
                false_neg_indicator = (
                    img_idx[:, None] == all_img_idx[None, :])
                false_neg_indicator.fill_diagonal_(False)
            t2i_sim.masked_fill_(false_neg_indicator, float('-inf'))
            i2t_sim.masked_fill_(false_neg_indicator, float('-inf'))
            caption_loss = self.contrastive_loss(t2i_sim, dim=1)
            image_loss = self.contrastive_loss(i2t_sim, dim=1)
        else:
            caption_loss = self.contrastive_loss(t2i_sim, dim=1)
            image_loss = self.contrastive_loss(i2t_sim, dim=1)
        return (caption_loss + image_loss) / 2.0

    def get_loss(self, img_tensor, text_ids_tensor, text_masks_tensor,
                 img_id_list):
        img_feat = self.forward(img_tensor, input_type='img')
        text_feat = self.forward((text_ids_tensor, text_masks_tensor),
                                 input_type='text')

        global_img_feat = torch.cat(all_gather_with_backprop(img_feat), dim=0)
        global_text_feat = torch.cat(
            all_gather_with_backprop(text_feat), dim=0)
        global_img_id_list = torch.cat(
            all_gather_with_backprop(img_id_list), dim=0)

        t2i_sim_mat = text_feat @ global_img_feat.t()
        i2t_sim_mat = img_feat @ global_text_feat.t()

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        t2i_sim_mat_logits = t2i_sim_mat * logit_scale
        i2t_sim_mat_logits = i2t_sim_mat * logit_scale

        loss = self.clip_loss(
            t2i_sim_mat_logits,
            i2t_sim_mat_logits,
            img_idx=img_id_list,
            all_img_idx=global_img_id_list)

        return loss

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
        elif input_type == ModeKeys.TRAIN:
            return self.get_loss(*input_data)
        else:
            raise ValueError('Unknown input type')


@MODELS.register_module(Tasks.multi_modal_embedding, module_name=Models.clip)
class CLIPForMultiModalEmbedding(TorchModel):

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
