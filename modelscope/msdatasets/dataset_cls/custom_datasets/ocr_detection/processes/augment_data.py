import math

import cv2
import imgaug
import numpy as np

from ..augmenter import AugmenterBuilder
from .data_process import DataProcess


class AugmentData(DataProcess):

    def __init__(self, cfg):
        self.augmenter_args = cfg.augmenter_args
        self.keep_ratio = cfg.keep_ratio
        self.only_resize = cfg.only_resize
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, c = image.shape
        resize_shape = self.augmenter_args[0][1]

        new_height_pad = resize_shape['height']
        new_width_pad = resize_shape['width']
        if self.keep_ratio:
            if origin_height > origin_width:
                new_height = new_height_pad
                new_width = int(
                    math.ceil(new_height / origin_height * origin_width / 32)
                    * 32)
            else:
                new_width = new_width_pad
                new_height = int(
                    math.ceil(new_width / origin_width * origin_height / 32)
                    * 32)
            image = cv2.resize(image, (new_width, new_height))

        else:
            image = cv2.resize(image, (new_width_pad, new_height_pad))

        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape
        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True
        else:
            data['is_training'] = False
        return data


class AugmentDetectionData(AugmentData):

    def may_augment_annotation(self, aug: imgaug.augmenters.Augmenter, data,
                               shape):
        if aug is None:
            return data

        line_polys = []
        keypoints = []
        texts = []
        new_polys = []
        for line in data['lines']:
            texts.append(line['text'])
            new_poly = []
            for p in line['poly']:
                new_poly.append((p[0], p[1]))
                keypoints.append(imgaug.Keypoint(p[0], p[1]))
            new_polys.append(new_poly)
        if not self.only_resize:
            keypoints = aug.augment_keypoints(
                [imgaug.KeypointsOnImage(keypoints=keypoints,
                                         shape=shape)])[0].keypoints
            new_polys = np.array([[p.x, p.y]
                                  for p in keypoints]).reshape([-1, 4, 2])
        for i in range(len(texts)):
            poly = new_polys[i]
            line_polys.append({
                'points': poly,
                'ignore': texts[i] == '###',
                'text': texts[i]
            })

        data['polys'] = line_polys
