# Part of the implementation is borrowed and modified from ControlNet,
# publicly available at https://github.com/lllyasviel/ControlNet

import math
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Preprocessors
from modelscope.models.cv.controllable_image_generation.annotator.annotator import (
    CannyDetector, HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector,
    SegformerDetector, nms)
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, Fields, Invoke,
                                       ModeKeys, Tasks)
from modelscope.utils.type_assert import type_assert


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def build_detector(control_type, model_path, device):
    if control_type == 'scribble':
        detector = None
    elif control_type == 'canny':
        detector = CannyDetector()
    elif control_type == 'hough':
        detector = MLSDdetector(model_path, device)
    elif control_type == 'hed':
        detector = HEDdetector(model_path, device)
    elif control_type == 'depth':
        detector = MidasDetector(model_path, device)
    elif control_type == 'normal':
        detector = MidasDetector(model_path, device)
    elif control_type == 'pose':
        detector = OpenposeDetector(model_path, device)
    elif control_type == 'seg':
        detector = SegformerDetector(model_path, device)
    elif control_type == 'fake_scribble':
        detector = HEDdetector(model_path, device)
    else:
        detector = HEDdetector(model_path, device)
    return detector


def get_detected_map(detector, control_type, img, **kwargs):
    if control_type == 'scribble':
        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255
    elif control_type == 'canny':
        detected_map = detector(img, kwargs['low_threshold'],
                                kwargs['high_threshold'])
        detected_map = HWC3(detected_map)
    elif control_type == 'hough':
        detected_map = detector(img, kwargs['value_threshold'],
                                kwargs['distance_threshold'])
        detected_map = HWC3(detected_map)
    elif control_type == 'hed':
        detected_map = detector(img)
        detected_map = HWC3(detected_map)
    elif control_type == 'depth':
        H, W, C = img.shape
        det_img = resize_image(img, 384)
        detected_map, _ = detector(det_img)
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    elif control_type == 'normal':
        H, W, C = img.shape
        det_img = resize_image(img, 384)
        _, detected_map = detector(det_img, bg_th=kwargs['bg_threshold'])
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = detected_map[:, :, ::-1]
    elif control_type == 'pose':
        detected_map, _ = detector(img)
        detected_map = HWC3(detected_map)
    elif control_type == 'seg':
        detected_map = detector(img)
    elif control_type == 'fake_scribble':
        detected_map = detector(img)
        detected_map = HWC3(detected_map)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

    return detected_map


@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.controllable_image_generation_preprocessor)
class ControllableImageGenerationPreprocessor(Preprocessor):

    def __init__(self, mode=ModeKeys.INFERENCE, *args, **kwargs):
        super().__init__(mode=ModeKeys.INFERENCE, *args, **kwargs)
        self.detector = build_detector(
            kwargs.get('control_type', 'hed'), kwargs.get('model_path', None),
            kwargs.get('device', 'cuda'))

    @type_assert(object, object)
    def __call__(self, data: input, **kwargs) -> Dict[str, Any]:
        image_resolution = data.get('image_resolution',
                                    kwargs['image_resolution'])
        image = np.array(load_image(data['image']))
        image = resize_image(HWC3(image), image_resolution)
        print(f'Test with image resolution: {image_resolution}')

        is_cat_img = kwargs.get('is_cat_img', True)

        if 'prompt' in data.keys():
            model_prompt = data['prompt']
        else:
            # for demo_service
            model_prompt = kwargs.get('prompt', '')
        print(f'Test with prompt: {model_prompt}')

        control_type = kwargs.get('control_type', 'hed')
        print(f'Test with input type: {control_type}')

        save_memory = kwargs.get('save_memory', False)

        # generate detected_map
        if control_type == 'scribble':
            detected_map = get_detected_map(self.detector, control_type, image)
        elif control_type == 'canny':
            low_threshold = kwargs['modelsetting'].canny.low_threshold
            high_threshold = kwargs['modelsetting'].canny.high_threshold
            detected_map = get_detected_map(
                self.detector,
                control_type,
                image,
                low_threshold=low_threshold,
                high_threshold=high_threshold)
        elif control_type == 'hough':
            value_threshold = kwargs['modelsetting'].hough.value_threshold
            distance_threshold = kwargs[
                'modelsetting'].hough.distance_threshold
            detected_map = get_detected_map(
                self.detector,
                control_type,
                image,
                value_threshold=value_threshold,
                distance_threshold=distance_threshold)
        elif control_type in ['hed', 'depth', 'pose', 'seg', 'fake_scribble']:
            detected_map = get_detected_map(self.detector, control_type, image)
        elif control_type == 'normal':
            bg_threshold = kwargs['modelsetting'].normal.bg_threshold
            detected_map = get_detected_map(
                self.detector, control_type, image, bg_threshold=bg_threshold)
        else:
            detected_map = get_detected_map(
                self.detector, control_type='hed', img=image)

        input_dict = {
            'image': image,
            'prompt': model_prompt,
            'detected_map': detected_map,
            'save_memory': save_memory,
            'is_cat_img': is_cat_img
        }

        for k in data.keys():
            if k not in input_dict.keys():
                input_dict[k] = data[k]

        return input_dict
