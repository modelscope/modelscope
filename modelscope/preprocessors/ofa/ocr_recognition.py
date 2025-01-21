# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
import unicodedata2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from zhconv import convert

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


def ocr_resize(img, patch_image_size, is_document=False):
    r"""
    Image resize function for OCR tasks.
    """
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
    r"""
    OFA preprocessor for OCR recognition tasks.
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
        super(OfaOcrRecognitionPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)

        self.patch_resize_transform = transforms.Compose([
            lambda image: ocr_resize(
                image,
                self.patch_image_size,
                is_document=self.cfg.model.get('is_document', False)),
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
            - do tripe to the label value.
            - tokenize the label as `target` value without `bos` token.
            - add `bos` token and remove `eos` token of `target` as `prev_output_tokens`.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `image`, `prompt` and `label`,
                the former refers the image input data, and the later refers the text input data
                the `label` is the supervised data for training.
        Return:
            A dict object, contains source, image, mask, label, target tokens,
            and previous output tokens data.
        """
        sample = self._build_infer_sample(data)
        target = sample['label']
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
            A dict object, contains source, image, image patch mask and label data.
        """
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        prompt = self.cfg.model.get('prompt', '图片上的文字是什么?')
        inputs = self.tokenize_text(prompt)

        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True])
        }
        if 'text' in self.column_map and self.column_map['text'] in data:
            target = data[self.column_map['text']]
            sample['label'] = unicodedata2.normalize(
                'NFKC', convert(target, 'zh-hans'))
        return sample
