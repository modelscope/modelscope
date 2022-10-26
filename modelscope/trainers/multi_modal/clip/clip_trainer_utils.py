# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random

import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from modelscope.utils.constant import ModeKeys

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                           p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])


class ImageWithCaptionDataset(Dataset):

    def __init__(self, json_file, img_dir, phase):
        self.annotations = json.load(open(json_file))
        self.img_dir = img_dir
        if phase == ModeKeys.TRAIN:
            self.transform = train_transform
        elif phase == ModeKeys.EVAL:
            self.transform = val_transform

        self.img_name2img_id = {}
        for anno_dict in self.annotations:
            img_name = anno_dict['image']
            if img_name not in self.img_name2img_id:
                self.img_name2img_id[img_name] = len(self.img_name2img_id)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        anno_dict = self.annotations[index]

        img_path = os.path.join(self.img_dir, anno_dict['image'])
        img_pil = Image.open(img_path).convert('RGB')
        img_th = self.transform(img_pil)
        img_id = self.img_name2img_id[anno_dict['image']]

        text_str = random.choice(anno_dict['caption'])

        return img_th, text_str, img_id


def get_params_groups(ddp_model, weight_decay):
    decay = []
    no_decay = []
    for name, param in ddp_model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    params_groups = [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }]
    return params_groups


def get_optimizer(ddp_model):
    from torch.optim import AdamW
    lr_init = 1e-5
    betas = [0.9, 0.999]
    weight_decay = 0.02
    params_groups = get_params_groups(ddp_model, weight_decay=weight_decay)
    return AdamW(
        params_groups, lr=lr_init, betas=betas, weight_decay=weight_decay)
