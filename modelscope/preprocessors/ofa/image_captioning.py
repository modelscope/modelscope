# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from torchvision import transforms

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaImageCaptioningPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for image captioning task.
    """

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
        r"""
        Building training samples.

        step 1. Preprocess the data using the logic of `_build_infer_sample`
            and make sure the label data in the result.
        step 2. Preprocess the label data. Contains:
            - remove tokens within `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~` and tripe
            - tokenize the label as `target` value without `bos` token.
            - add `bos` token and remove `eos` token of `target` as `prev_output_tokens`.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `image`, `prompt`
                and `label`, `image` refers the image input data, `prompt` refers the text
                input data the `label` is the supervised data for training.
        Return:
            A dict object, contains source, image, mask, label, target tokens,
            and previous output tokens data.
        """
        sample = self._build_infer_sample(data)
        target = sample['label']
        target = target.translate(self.transtab).strip()
        target_token_list = target.strip().split()
        target = ' '.join(target_token_list[:self.max_tgt_length])
        sample['target'] = self.tokenize_text(target, add_bos=False)
        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, sample['target'][:-1]])
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Building inference samples.

        step 1. Get the pillow image.
        step 2. Do some transforms for the pillow image as the image input,
            such as resize, normalize, to tensor etc.
        step 3. Tokenize the prompt as text input.
        step 4. Determine Whether or not to add labels to the sample.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `image` and `prompt`,
                the former refers the image input data, and the later refers the text input data.
        Return:
            A dict object, contains source, image, mask and label data.
        """
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        prompt = self.cfg.model.get('prompt', ' what does the image describe?')
        inputs = self.tokenize_text(prompt)
        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True])
        }
        if 'text' in self.column_map and self.column_map['text'] in data:
            sample['label'] = data[self.column_map['text']]
        return sample
