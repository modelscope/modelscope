# The implementation here is modified based on ddpm-segmentation,
# originally Apache 2.0 License and publicly avaialbe at https://github.com/yandex-research/ddpm-segmentation
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.distributions import Categorical

from .data_util import get_class_names, get_palette
from .utils import colorize_mask, oht_to_scalar


# Adopted from https://github.com/nv-tlabs/datasetGAN/train_interpreter.py
class pixel_classifier(nn.Module):

    def __init__(self, category, dim, **kwargs):
        super(pixel_classifier, self).__init__()
        category_cfg = kwargs.get(category, None)
        assert category_cfg is not None
        class_num = category_cfg['number_class']

        dim = dim[-1]

        if class_num < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim,
                          128), nn.ReLU(), nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32), nn.ReLU(), nn.BatchNorm1d(num_features=32),
                nn.Linear(32, class_num))
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim,
                          256), nn.ReLU(), nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128), nn.ReLU(),
                nn.BatchNorm1d(num_features=128), nn.Linear(128, class_num))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)


def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            models[MODEL_NUMBER].to(features.device)
            preds = models[MODEL_NUMBER](features)
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][-int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


def load_ensemble(model_num=1,
                  model_path='',
                  device='cpu',
                  category='ffhq_34',
                  dim=[256, 256, 8448],
                  **kwargs):
    models = []

    for i in range(model_num):
        per_model_path = os.path.join(model_path, f'model_{i}.pth')
        state_dict = torch.load(
            per_model_path, map_location='cpu')['model_state_dict']
        new_state_dict = {
            k.replace('module.', ''): v
            for k, v in state_dict.items()
        }
        model = pixel_classifier(category, dim, **kwargs)
        model.load_state_dict(new_state_dict)
        models.append(model.eval())
    return models


def save_predictions(preds, category='ffhq_34'):
    palette = get_palette(category)

    masks = []
    out_imgs = []
    for i, pred in enumerate(preds['pred']):

        pred = np.squeeze(pred)
        masks.append(pred)

        out_img = colorize_mask(pred, palette)
        out_img = Image.fromarray(out_img)
        out_imgs.append(out_img)
    return masks, out_imgs
