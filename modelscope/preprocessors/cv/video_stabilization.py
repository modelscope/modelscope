# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import cv2
import numpy as np
import torch
import torch.nn as nn

from modelscope.preprocessors.cv import VideoReader


def stabilization_preprocessor(input, cfg):
    video_reader = VideoReader(input)
    inputs = []
    for frame in video_reader:
        inputs.append(np.flip(frame, axis=2))
    fps = video_reader.fps
    w = video_reader.width
    h = video_reader.height
    rgb_images = []
    images = []
    ori_images = []
    for i, frame in enumerate(inputs):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = image * (1. / 255.)
        image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        images.append(image.reshape(1, 1, cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH))
        rgb_image = cv2.resize(frame, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        rgb_images.append(
            np.expand_dims(np.transpose(rgb_image, (2, 0, 1)), 0))
        ori_images.append(np.expand_dims(np.transpose(frame, (2, 0, 1)), 0))
    x = np.concatenate(images, 1).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    x_rgb = np.concatenate(rgb_images, 0).astype(np.float32)
    x_rgb = torch.from_numpy(x_rgb).unsqueeze(0)

    return {
        'ori_images': ori_images,
        'x': x,
        'x_rgb': x_rgb,
        'fps': fps,
        'width': w,
        'height': h
    }
