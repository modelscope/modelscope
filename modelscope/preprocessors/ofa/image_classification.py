# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
from typing import Any, Dict

import torch
from PIL import Image, ImageFile
from timm.data import create_transform
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor
from .utils.vision_helper import RandomAugment

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


class OfaImageClassificationPreprocessor(OfaBasePreprocessor):

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
        super(OfaImageClassificationPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)
        # Initialize transform
        if self.mode != ModeKeys.TRAIN:
            self.patch_resize_transform = transforms.Compose([
                lambda image: image.convert('RGB'),
                transforms.Resize(
                    (self.patch_image_size, self.patch_image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.patch_resize_transform = create_transform(
                input_size=self.patch_image_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                mean=self.mean,
                std=self.std)
            self.patch_resize_transform = transforms.Compose(
                functools.reduce(lambda x, y: x + y, [
                    [
                        lambda image: image.convert('RGB'),
                    ],
                    self.patch_resize_transform.transforms[:2],
                    [self.patch_resize_transform.transforms[2]],
                    [
                        RandomAugment(
                            2,
                            7,
                            isPIL=True,
                            augs=[
                                'Identity', 'AutoContrast', 'Equalize',
                                'Brightness', 'Sharpness', 'ShearX', 'ShearY',
                                'TranslateX', 'TranslateY', 'Rotate'
                            ]),
                    ],
                    self.patch_resize_transform.transforms[3:],
                ]))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = self._build_infer_sample(data)
        target = ' {}'.format(data[self.column_map['text']])
        sample['ref_dict'] = {data[self.column_map['text']]: 1.0}
        sample['target'] = self.tokenize_text(target, add_bos=False)
        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, sample['target']])

        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(sample['prev_output_tokens']),
                                           len(self.tgt_dict))).bool()
            for i in range(len(sample['prev_output_tokens'])):
                constraint_prefix_token = sample[
                    'prev_output_tokens'][:i + 1].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(
                    constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            sample['constraint_mask'] = constraint_mask

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
        if 'text' in self.column_map and self.column_map['text'] in data:
            sample['label'] = data[self.column_map['text']]
        return sample
