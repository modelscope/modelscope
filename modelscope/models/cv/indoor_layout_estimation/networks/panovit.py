# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone, modality
from .utils import visualize_a_data


class PanoVIT(nn.Module):

    def __init__(self,
                 emb_dim=256,
                 input_hw=None,
                 input_norm='imagenet',
                 pretrain='',
                 backbone_config={'module': 'Resnet'},
                 transformer_config={'module': 'ViT'},
                 modalities_config={}):
        super(PanoVIT, self).__init__()
        self.input_hw = input_hw
        if input_norm == 'imagenet':
            self.register_buffer(
                'x_mean',
                torch.FloatTensor(
                    np.array([0.485, 0.456, 0.406])[None, :, None, None]))
            self.register_buffer(
                'x_std',
                torch.FloatTensor(
                    np.array([0.229, 0.224, 0.225])[None, :, None, None]))
        else:
            raise NotImplementedError

        Encoder = getattr(backbone, backbone_config['module'])
        Encoder_kwargs = backbone_config.get('kwargs', {})
        self.encoder = Encoder(**Encoder_kwargs)

        Transformer = getattr(backbone, transformer_config['module'])
        Transformer_kwargs = transformer_config.get('kwargs', {})
        self.transformer = Transformer(**Transformer_kwargs)
        self.transformer_config = transformer_config['module']
        self.transformer_Fourier = transformer_config['kwargs']['fourier']

        self.modalities = nn.ModuleList([
            getattr(modality, key)(emb_dim, **config)
            for key, config in modalities_config.items()
        ])

    def extract_feat(self, x):
        img = x[:, 0:3, :, :]
        if self.input_hw:
            img = F.interpolate(
                img, size=self.input_hw, mode='bilinear', align_corners=False)

        img = (img - self.x_mean) / self.x_std
        if self.transformer_Fourier == 'fourier_res':
            img = torch.cat((img, x[:, 3:, :, :]), dim=1)
            res_f = self.encoder(img)
        elif self.transformer_Fourier == 'fourier_trans':
            res_f = self.encoder(img)
            img = torch.cat((img, x[:, 3:, :, :]), dim=1)
        else:
            res_f = self.encoder(img)

        if self.transformer_config == 'ViTHorizonPryImage':

            feat = self.transformer(img, res_f)
        else:
            feat = self.transformer(x)
        return feat

    def call_modality(self, method, *feed_args, **feed_kwargs):
        output_dict = {}
        for m in self.modalities:
            curr_dict = getattr(m, method)(*feed_args, **feed_kwargs)
            assert len(output_dict.keys() & curr_dict.keys()
                       ) == 0, 'Key collision for different modalities'
            output_dict.update(curr_dict)
        return output_dict

    def forward(self, x):
        feat = self.extract_feat(x)
        results = self.call_modality('forward', feat)
        return torch.cat((results['pred_bon'], results['pred_cor']), dim=1)

    def infer(self, x):
        feat = self.extract_feat(x)
        result = self.call_modality('infer', feat)
        result['image'] = x
        return result

    def postprocess(self, image, y_bon, y_cor):
        vis_layout = visualize_a_data(image, y_bon, y_cor)
        return (vis_layout[:, :, (2, 1, 0)])
