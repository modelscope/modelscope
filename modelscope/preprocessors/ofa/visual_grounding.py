# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
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

        if self.mode == ModeKeys.TRAIN:
            # for positioning
            self.positioning_transform = transforms.Compose([
                transforms.RandomResize([self.patch_image_size],
                                        max_size=self.patch_image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_image_size=self.max_image_size)
            ])
        else:
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
        image = self.get_img_pil(data[self.column_map['image']])
        w, h = image.size
        boxes_target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'size': torch.tensor([h, w])
        }
        x0, y0, x1, y1 = data[self.column_map['region_coord']].strip().split(
            ',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target['boxes'] = torch.tensor(
            [[float(x0), float(y0), float(x1),
              float(y1)]])
        boxes_target['labels'] = np.array([0])
        area = [(float(x1) - float(x0)) * (float(y1) - float(y0))]
        boxes_target['area'] = torch.tensor(area)

        patch_image, patch_boxes = self.positioning_transform(
            image, boxes_target)
        resize_h, resize_w = patch_boxes['size'][0], patch_boxes['size'][1]
        quant_x0 = '<bin_{}>'.format(
            int((patch_boxes['boxes'][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = '<bin_{}>'.format(
            int((patch_boxes['boxes'][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = '<bin_{}>'.format(
            int((patch_boxes['boxes'][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = '<bin_{}>'.format(
            int((patch_boxes['boxes'][0][3] * (self.num_bins - 1)).round()))
        region_coord = '{} {} {} {}'.format(quant_x0, quant_y0, quant_x1,
                                            quant_y1)
        src_caption = self.pre_caption(data[self.column_map['text']],
                                       self.max_src_length)
        prompt = self.cfg.model.get(
            'prompt', ' which region does the text " {} " describe?')
        text = prompt.format(src_caption)
        src_item = self.tokenize_text(text)
        target_item = self.tokenize_text(
            region_coord, add_bos=False)  # !!! use_bpe=False
        prev_output_item = torch.cat([self.bos_item, target_item[:-1]])

        sample = {
            'source': src_item,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'target': target_item,
            'prev_output_tokens': prev_output_item,
            'w_resize_ratio': resize_w / w,
            'h_resize_ratio': resize_h / h,
            'region_coord': region
        }
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = self.get_img_pil(data[self.column_map['image']])
        w, h = image.size
        patch_image = self.patch_resize_transform(image)
        w_resize_ratio = torch.tensor(self.patch_image_size / w)
        h_resize_ratio = torch.tensor(self.patch_image_size / h)
        src_caption = self.pre_caption(data[self.column_map['text']],
                                       self.max_src_length)
        prompt = self.cfg.model.get(
            'prompt', ' which region does the text " {} " describe?')
        text = prompt.format(src_caption)
        src_item = self.tokenize_text(text)
        sample = {
            'source': src_item,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'w_resize_ratio': w_resize_ratio,
            'h_resize_ratio': h_resize_ratio,
        }
        return sample
