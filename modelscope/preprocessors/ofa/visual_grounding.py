# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaVisualGroundingPreprocessor(OfaBasePreprocessor):

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaVisualGroundingPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)
        # Initialize transform
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize(
                (self.patch_image_size, self.patch_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image'] if isinstance(
            data['image'], Image.Image) else load_image(data['image'])
        w, h = image.size
        patch_image = self.patch_resize_transform(image)
        w_resize_ratio = torch.tensor(self.patch_image_size / w)
        h_resize_ratio = torch.tensor(self.patch_image_size / h)
        src_caption = self.pre_caption(data['text'], self.max_src_length)
        prompt = self.cfg.model.get(
            'prompt', ' which region does the text " {} " describe?')
        text = prompt.format(src_caption)
        src_item = self.get_inputs(text)
        sample = {
            'source': src_item,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'w_resize_ratio': w_resize_ratio,
            'h_resize_ratio': h_resize_ratio,
        }
        return sample
