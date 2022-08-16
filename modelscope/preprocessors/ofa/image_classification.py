# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from .base import OfaBasePreprocessor


class OfaImageClassificationPreprocessor(OfaBasePreprocessor):

    def __init__(self, cfg, model_dir):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path
        """
        super(OfaImageClassificationPreprocessor,
              self).__init__(cfg, model_dir)
        # Initialize transform
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize((self.patch_image_size, self.patch_image_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image'] if isinstance(
            data['image'], Image.Image) else load_image(data['image'])
        patch_image = self.patch_resize_transform(image)
        prompt = self.cfg.model.get('prompt', ' what does the image describe?')
        inputs = self.get_inputs(prompt)
        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True])
        }
        return sample
