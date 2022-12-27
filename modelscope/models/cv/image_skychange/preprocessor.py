# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import pdb
from typing import Any, Dict, Union

import cv2
import json
import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Fields, ModeKeys

_cv2_pad_to_str = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT,
}


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.image_sky_change_preprocessor)
class ImageSkyChangePreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str = None,
                 mode: str = ModeKeys.INFERENCE,
                 coarse_model_width=640,
                 coarse_model_height=640,
                 refine_model_width=1280,
                 refine_model_height=1280,
                 mean_vec=[0.485, 0.456, 0.406],
                 std_vec=[0.229, 0.224, 0.225],
                 *args,
                 **kwargs):
        """
        Args:
            model_dir (str): model directory to initialize some resource.
            mode: The mode for the preprocessor.
            coarse_model_width: required width of input tensor of coarse model.
            coarse_model_height: required height of input tensor of coarse model.
            refine_model_width: required width of input tensor of refine model.
            refine_model_height: required height of input tensor of refine model.
            mean_vec: mean of dataset(for transforms.Normalize), default is mean of Imagenet dataset.
            std_vec: standard deviation of dataset(for transforms.Normalize), default is std of Imagenet dataset.
        """
        super().__init__(mode)

        # set preprocessor info
        self.coarse_input_size = [coarse_model_width, coarse_model_height]
        self.refine_input_size = [refine_model_width, refine_model_height]
        self.normalize = transforms.Normalize(mean=mean_vec, std=std_vec)

    def __call__(self, data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """process the raw input data
        Args:
            data (dict): data dict containing following info:
                sky_image, scene_image
                example:
                    ```python
                    {
                        "sky_image": "xxx.jpg" # sky_image path(str)
                        "scene_image": "xxx.jpg", # scene_image path(str)
                    }
                    ```
        Returns:
            Dict[str, Any]: the preprocessed data
            {
                "sky_image": the preprocessed sky image(origin size)
                "sky_image_refine": the preprocessed resized sky image
                "scene_image": the preprocessed scene image(origin size)
                "scene_image_refine": the preprocessed resized scene image
                "img_metas": informations of preprocessed images, e.g. origin shape, pad information, resized shape.
            }
        """
        if 'sky_image' not in data.keys():
            raise Exception('sky_image not in input data')
        if 'scene_image' not in data.keys():
            raise Exception('scene_image not in input data')
        if isinstance(data['sky_image'], str):
            sky_image = LoadImage.convert_to_ndarray(data['sky_image'])
            sky_image = sky_image.astype(np.uint8)  # RGB
            sky_image = cv2.cvtColor(sky_image, cv2.COLOR_RGB2BGR)  # BGR
            if sky_image is not None:
                sky_image = self.check_image(sky_image)
            else:
                raise Exception('sky_image is None')
        else:
            raise Exception('sky_image(path of sky image) is not valid')
        if isinstance(data['scene_image'], str):
            scene_image = LoadImage.convert_to_ndarray(data['scene_image'])
            scene_image = scene_image.astype(np.uint8)  # RGB
            scene_image = cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR)  # BGR
            if scene_image is not None:
                scene_image = self.check_image(scene_image)
            else:
                raise Exception('scene_image is None')
        else:
            raise Exception('scene_image(path of scene image) is not valid')
        data = {}
        sky_image_refine, sky_img_metas = self.process_single_img(sky_image)
        scene_image_refine, scene_img_metas = self.process_single_img(
            scene_image)
        data['sky_image'] = sky_image
        data['sky_image_refine'] = sky_image_refine
        data['scene_image'] = scene_image
        data['scene_image_refine'] = scene_image_refine
        data['img_metas'] = {
            'sky_img_metas': sky_img_metas,
            'scene_img_metas': scene_img_metas,
            'input_size': {
                'coarse_input_size': self.coarse_input_size,
                'refine_input_size': self.refine_input_size
            }
        }
        return data

    def process_single_img(self, img):
        img_metas = {}
        img_metas['ori_shape'] = img.shape[0:2]  # img: (origin_h, origin_w, 3)
        img, pad_direction = get_refine_input(img, self.refine_input_size)
        img = image_transform(
            img, self.normalize)  # torch.Size([3, refine_net_h, refine_net_w])
        img = img.unsqueeze(0)
        img_metas['pad_direction'] = pad_direction
        img_metas['refine_shape'] = img.shape[
            2:]  # torch.Size([1, 3, refine_net_h, refine_net_w])
        return img, img_metas

    def check_image(self, input_img):
        whole_temp_shape = input_img.shape
        if len(whole_temp_shape) == 2:
            input_img = np.stack([input_img, input_img, input_img], axis=2)
        elif whole_temp_shape[2] == 1:
            input_img = np.concatenate([input_img, input_img, input_img],
                                       axis=2)
        elif whole_temp_shape[2] == 4:
            input_img = input_img[:, :,
                                  0:3] * 1.0 * input_img[:, :,
                                                         3:4] * 1.0 / 255.0
        return input_img


def get_refine_input(mat, refine_input_size):
    # maxDimMatch: resize
    mat = max_dim_match(mat, refine_input_size)
    # pad image to refine net input size
    mat, pad_direction = center_pad_image_withwh(mat, refine_input_size, 0)
    return mat, pad_direction


def max_dim_match(image, refine_model_size):
    h, w, c = np.shape(image)
    resize_w, resize_h = refine_model_size
    if h != resize_h or w != resize_w:
        h_scale = float(resize_h) / h
        w_scale = float(resize_w) / w
        resize_scale = min(w_scale, h_scale)
        new_h = int(h * resize_scale + 0.5)
        new_w = int(w * resize_scale + 0.5)
        image = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def center_pad_image_withwh(image,
                            crop_size,
                            padvalue,
                            padding_mode='constant'):
    pad_image = image
    h, w = image.shape[0], image.shape[1]
    pad_h = max(crop_size[1] - h, 0)
    pad_w = max(crop_size[0] - w, 0)
    pad_direction = (0, 0, 0, 0)
    if pad_h > 0 or pad_w > 0:
        half_w = int(pad_w / 2 + 0.5)
        half_h = int(pad_h / 2 + 0.5)
        pad_direction = (half_w, half_h, pad_w - half_w, pad_h - half_h)
        pad_image = pad(
            image, pad_direction, padvalue, padding_mode=padding_mode)
    return pad_image, pad_direction


def pad(img, padding, fill=0, padding_mode='constant'):
    if not is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(
            type(img)))
    if not isinstance(padding,
                      (numbers.Number, tuple, list)) or len(padding) != 4:
        raise TypeError('Got inappropriate padding arg')

    pad_left = padding[0]
    pad_top = padding[1]
    pad_right = padding[2]
    pad_bottom = padding[3]

    shape_len = len(img.shape)
    if shape_len == 2:
        return cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=_cv2_pad_to_str[padding_mode],
            value=fill,
        )
    elif shape_len == 3 and img.shape[2] == 1:
        return cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=_cv2_pad_to_str[padding_mode],
            value=fill,
        )[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=_cv2_pad_to_str[padding_mode],
            value=fill,
        )


def image_transform(img, normalize):
    img = img[:, :, ::-1]  # BGR-->RGB to pil format
    img = img.transpose((2, 0, 1))  # h,w,c --> c,h,w
    img = img.astype(np.float32) / 255
    img = normalize(torch.from_numpy(img.copy()))
    return img


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
