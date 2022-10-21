# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unicodedata
from typing import Any, Dict, Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from modelscope.preprocessors.image import load_image
from .base import OfaBasePreprocessor

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def ocr_resize(img, patch_image_size, is_document=False):
    img = img.convert('RGB')
    width, height = img.size

    if is_document:
        new_height, new_width = 64, 1920
    else:
        if width >= height:
            new_width = max(64, patch_image_size)
            new_height = max(64, int(patch_image_size * (height / width)))
            top = (patch_image_size - new_height) // 2
            bottom = patch_image_size - new_height - top
            left, right = 0, 0
        else:
            new_height = max(64, patch_image_size)
            new_width = max(64, int(patch_image_size * (width / height)))
            left = (patch_image_size - new_width) // 2
            right = patch_image_size - new_width - left
            top, bottom = 0, 0

    img_new = F.resize(
        img,
        (new_height, new_width),
        interpolation=InterpolationMode.BICUBIC,
    )

    if is_document:
        img_split = transforms.ToTensor()(img_new).chunk(4, dim=-1)
        img_new = transforms.ToPILImage()(torch.cat(img_split, dim=-2))
        new_width, new_height = img_new.size
        top = (patch_image_size - new_height) // 2
        bottom = patch_image_size - new_height - top
        left, right = 0, 0

    img_new = F.pad(
        img_new, padding=[left, top, right, bottom], padding_mode='edge')
    assert img_new.size == (patch_image_size, patch_image_size)

    return img_new


class OfaOcrRecognitionPreprocessor(OfaBasePreprocessor):

    def __init__(self, cfg, model_dir):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path
        """
        super(OfaOcrRecognitionPreprocessor, self).__init__(cfg, model_dir)
        # Initialize transform
        if self.cfg.model.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: ocr_resize(
                image,
                self.cfg.model.patch_image_size,
                is_document=self.cfg.model.is_document),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image'] if isinstance(
            data['image'], Image.Image) else load_image(data['image'])
        patch_image = self.patch_resize_transform(image)
        prompt = self.cfg.model.get('prompt', '图片上的文字是什么?')
        inputs = self.get_inputs(prompt)

        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True])
        }
        return sample
