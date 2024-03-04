# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math
import random
from ast import Global
from pickle import GLOBAL

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .generators.extraction_distribution_model_flow25 import \
    Generator as Generator

tv_version = int(torchvision.__version__.split('.')[1])
if tv_version > 8:
    from torchvision.transforms.functional import InterpolationMode
    resize_method = InterpolationMode.BICUBIC
    resize_nearest = InterpolationMode.NEAREST
else:
    resize_method = Image.BICUBIC
    resize_nearest = Image.NEAREST

logger = get_logger()


def get_random_params(size, scale_param, use_flip=False):
    w, h = size
    scale = random.random() * scale_param

    if use_flip:
        use_flip = random.random() > 0.9

    new_w = int(w * (1.0 + scale))
    new_h = int(h * (1.0 + scale))
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))
    return {
        'crop_param': (x, y, w, h),
        'scale_size': (new_h, new_w),
        'use_flip': use_flip
    }


def get_transform(param, method=resize_method, normalize=True, toTensor=True):
    transform_list = []
    if 'scale_size' in param and param['scale_size'] is not None:
        osize = param['scale_size']
        transform_list.append(transforms.Resize(osize, interpolation=method))

    if 'crop_param' in param and param['crop_param'] is not None:
        transform_list.append(
            transforms.Lambda(lambda img: __crop(img, param['crop_param'])))

    if param['use_flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img)))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return transforms.Compose(transform_list)


def __crop(img, pos):
    x1, y1, tw, th = pos
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img):
    return F.hflip(img)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def load_checkpoint(model, checkpoint_path, device):
    params = torch.load(checkpoint_path, map_location=device)
    if 'target_image_renderer.weight' in params['net_G_ema'].keys():
        params['net_G_ema'].pop('target_image_renderer.weight')
    model.load_state_dict(params['net_G_ema'])
    model.to(device)
    model.eval()
    return model


@MODELS.register_module(
    Tasks.human_image_generation, module_name=Models.human_image_generation)
class FreqHPTForHumanImageGeneration(TorchModel):
    """initialize the human image generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
    """

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

        size = 512
        semantic_dim = 20
        channels = {
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 128,
            512: 64,
            1024: 32
        }
        num_labels = {16: 16, 32: 32, 64: 64, 128: 64, 256: 64, 512: False}
        match_kernels = {16: False, 32: 3, 64: 3, 128: 3, 256: 3, 512: False}
        wavelet_down_levels = {16: False, 32: 1, 64: 2, 128: 3, 256: 3, 512: 3}
        self.model = Generator(
            size,
            semantic_dim,
            channels,
            num_labels,
            match_kernels,
            wavelet_down_levels=wavelet_down_levels)
        self.model = load_checkpoint(
            self.model, model_dir + '/' + ModelFile.TORCH_MODEL_BIN_FILE,
            self.device)

    def forward(self, x, y, z):
        pred_result = self.model(x, y, z)
        return pred_result


def trans_keypoints(keypoints, param, img_size, offset=None):
    missing_keypoint_index = keypoints == -1

    # crop the white line in the original dataset
    if not offset == 40:
        keypoints[:, 0] = (keypoints[:, 0] - 40)

    # resize the dataset
    img_h, img_w = img_size
    scale_w = 1.0 / 176.0 * img_w
    scale_h = 1.0 / 256.0 * img_h

    if 'scale_size' in param and param['scale_size'] is not None:
        new_h, new_w = param['scale_size']
        scale_w = scale_w / img_w * new_w
        scale_h = scale_h / img_h * new_h

    if 'crop_param' in param and param['crop_param'] is not None:
        w, h, _, _ = param['crop_param']
    else:
        w, h = 0, 0

    keypoints[:, 0] = keypoints[:, 0] * scale_w - w
    keypoints[:, 1] = keypoints[:, 1] * scale_h - h

    normalized_kp = keypoints.copy()
    normalized_kp[:, 0] = (normalized_kp[:, 0]) / img_w * 2 - 1
    normalized_kp[:, 1] = (normalized_kp[:, 1]) / img_h * 2 - 1
    normalized_kp[missing_keypoint_index] = -1

    keypoints[missing_keypoint_index] = -1
    return keypoints, normalized_kp


def get_label_tensor(path, img, param):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15],
               [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
              [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
              [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
              [255, 0, 170], [255, 0, 85]]
    canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
    keypoint = np.loadtxt(path)
    keypoint, normalized_kp = trans_keypoints(keypoint, param, img.shape[1:])
    stickwidth = 4
    for i in range(18):
        x, y = keypoint[i, 0:2]
        if x == -1 or y == -1:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    joints = []
    for i in range(17):
        Y = keypoint[np.array(limbSeq[i]) - 1, 0]
        X = keypoint[np.array(limbSeq[i]) - 1, 1]
        cur_canvas = canvas.copy()
        if -1 in Y or -1 in X:
            joints.append(np.zeros_like(cur_canvas[:, :, 0]))
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
            360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        joint = np.zeros_like(cur_canvas[:, :, 0])
        cv2.fillConvexPoly(joint, polygon, 255)
        joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
        joints.append(joint)
    pose = F.to_tensor(
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

    tensors_dist = 0
    e = 1
    for i in range(len(joints)):
        im_dist = cv2.distanceTransform(255 - joints[i], cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
        tensor_dist = F.to_tensor(Image.fromarray(im_dist))
        tensors_dist = tensor_dist if e == 1 else torch.cat(
            [tensors_dist, tensor_dist])
        e += 1

    label_tensor = torch.cat((pose, tensors_dist), dim=0)
    return label_tensor, normalized_kp


def get_image_tensor(path):
    img = Image.open(path)
    param = get_random_params(img.size, 0)
    trans = get_transform(param, normalize=True, toTensor=True)
    img = trans(img)
    return img, param


def infer(genmodel, image_path, target_label_path, device):
    ref_tensor, param = get_image_tensor(image_path)
    target_label_tensor, target_kp = get_label_tensor(target_label_path,
                                                      ref_tensor, param)

    ref_tensor = ref_tensor.unsqueeze(0).to(device)
    target_label_tensor = target_label_tensor.unsqueeze(0).to(device)
    target_kp = torch.from_numpy(target_kp).unsqueeze(0).to(device)
    output_dict = genmodel(ref_tensor, target_label_tensor, target_kp)
    output_image = output_dict['fake_image'][0]

    output_image = output_image.clamp_(-1, 1)
    image = (output_image + 1) * 0.5
    image = image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    image = np.uint8(image)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr
