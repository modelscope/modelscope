# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import AdamW

from modelscope.utils.logger import get_logger

logger = get_logger()

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


def train_mapping(examples):
    examples['pixel_values'] = [
        train_transforms(Image.open(image).convert('RGB'))
        for image in examples['image:FILE']
    ]
    examples['labels'] = [label for label in examples['label:LABEL']]
    return examples


def val_mapping(examples):
    examples['pixel_values'] = [
        val_transforms(Image.open(image).convert('RGB'))
        for image in examples['image:FILE']
    ]
    examples['labels'] = [label for label in examples['label:LABEL']]
    return examples


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example['pixel_values']))
        labels.append(example['labels'])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {'pixel_values': pixel_values, 'labels': labels}


def get_params_groups(ddp_model, lr):
    large_lr_params = []
    small_lr_params = []
    for name, param in ddp_model.named_parameters():
        if not param.requires_grad:
            continue

        if 'encoder' in name:
            small_lr_params.append(param)
        elif 'classifier' in name:
            large_lr_params.append(param)
        else:
            logger.info('skip param: {}'.format(name))

    params_groups = [{
        'params': small_lr_params,
        'lr': lr / 10.0
    }, {
        'params': large_lr_params,
        'lr': lr
    }]
    return params_groups


def get_optimizer(ddp_model):
    lr_init = 1e-3
    betas = [0.9, 0.999]
    weight_decay = 0.02
    params_groups = get_params_groups(ddp_model, lr=lr_init)
    return AdamW(
        params_groups, lr=lr_init, betas=betas, weight_decay=weight_decay)
