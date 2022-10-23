# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from torchvision import transforms

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaImageCaptioningPreprocessor(OfaBasePreprocessor):

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
        super(OfaImageCaptioningPreprocessor,
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
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = self._build_infer_sample(data)
        target = data[self.column_map['text']]
        target = target.translate(self.transtab).strip()
        target_token_list = target.strip().split()
        target = ' '.join(target_token_list[:self.max_tgt_length])
        sample['target'] = self.tokenize_text(target, add_bos=False)
        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, sample['target'][:-1]])
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        prompt = self.cfg.model.get('prompt', ' what does the image describe?')
        inputs = self.tokenize_text(prompt)
        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True])
        }
        if self.column_map['text'] in data:
            sample['label'] = data[self.column_map['text']]
        return sample
