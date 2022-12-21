# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import os
import pdb
from collections import deque

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

torch.backends.cudnn.enabled = True

IMAGE_MAX_DIM = 3000
IMAGE_MIN_DIM = 50
IMAGE_MAX_RATIO = 10.0
IMAGE_BLENDER_MASK_RESIZE_SCALE = 10.0
IMAGE_BLENDER_INNER_RECT_MAX_DIM = 256
IMAGE_BLENDER_DILATE_KERNEL_SIZE = 7
IMAGE_BLENDER_VALID_MASK_THRESHOLD = 100
IMAGE_BLENDER_MIN_VALID_SKY_AREA = 100
IMAGE_BLENDER_MIN_RESIZE_DIM = 10
IMAGE_BLENDER_BLUR_KERNEL_SIZE = 5


def extract_sky_image(in_sky_image, in_sky_mask):
    scale = 1.0
    resize_mask = in_sky_mask.copy()

    rows, cols = resize_mask.shape[0:2]
    # src size: (512, 640), target size: (256,256), then scale to size (256, 320)
    if (rows > IMAGE_BLENDER_INNER_RECT_MAX_DIM
            or cols > IMAGE_BLENDER_INNER_RECT_MAX_DIM):
        height_scale = IMAGE_BLENDER_INNER_RECT_MAX_DIM / float(rows)
        width_scale = IMAGE_BLENDER_INNER_RECT_MAX_DIM / float(cols)
        scale = height_scale if height_scale > width_scale else width_scale
        new_size = (max(int(cols * scale), 1), max(int(rows * scale),
                                                   1))  # w, h
        resize_mask = cv2.resize(resize_mask, new_size, cv2.INTER_LINEAR)

    kernelSize = max(3, int(scale * IMAGE_BLENDER_DILATE_KERNEL_SIZE + 0.5))

    element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (kernelSize, kernelSize))
    resize_mask = cv2.morphologyEx(resize_mask, cv2.MORPH_CLOSE, element)

    max_inner_rect, area = get_max_inner_rect(
        resize_mask, IMAGE_BLENDER_VALID_MASK_THRESHOLD, True)

    if area < IMAGE_BLENDER_MIN_VALID_SKY_AREA:
        raise Exception(
            '[extractSkyImage]failed!! Valid sky region is too small')

    scale = 1.0 / scale
    # max_inner_rect: left top(x,y), right bottome(x,y); raw_inner_rect:left top x,y,w(of bbox),h(of bbox)
    raw_inner_rect = scale_rect(max_inner_rect, in_sky_mask, scale)
    out_sky_image = in_sky_image[raw_inner_rect[1]:raw_inner_rect[1]
                                 + raw_inner_rect[3] + 1,
                                 raw_inner_rect[0]:raw_inner_rect[0]
                                 + raw_inner_rect[2] + 1, ].copy()
    return out_sky_image


def blend(scene_image, scene_mask, sky_image, sky_mask, inBlendLevelNum=10):
    if torch.cuda.is_available():
        scene_image = scene_image.cpu().numpy()
        sky_image = sky_image.cpu().numpy()
    else:
        scene_image = scene_image.numpy()
        sky_image = sky_image.numpy()
    sky_image_h, sky_image_w = sky_image.shape[0:2]
    sky_mask_h, sky_mask_w = sky_mask.shape[0:2]

    scene_image_h, scene_image_w = scene_image.shape[0:2]
    scene_mask_h, scene_mask_w = scene_mask.shape[0:2]

    if sky_image_h != sky_mask_h or sky_image_w != sky_mask_w:
        raise Exception(
            '[blend]failed!! sky_image shape not equal with sky_image_mask shape'
        )

    if scene_image_h != scene_mask_h or scene_image_w != scene_mask_w:
        raise Exception(
            '[blend]failed!! scene_image shape not equal with scene_image_mask shape'
        )

    valid_sky_image = extract_sky_image(sky_image, sky_mask)
    out_blend_image = blend_merge(scene_image, scene_mask, valid_sky_image,
                                  inBlendLevelNum)
    return out_blend_image


def get_max_inner_rect(in_image_mask, in_alpha_threshold, is_bigger_valid):
    res = 0
    row, col = in_image_mask.shape[0:2]
    i0, j0, i1, j1 = 0, 0, 0, 0
    height = [0] * (col + 1)

    for i in range(0, row):
        s = deque()
        for j in range(0, col + 1):
            if j < col:
                if is_bigger_valid:
                    height[j] = (
                        height[j]
                        + 1 if in_image_mask[i, j] > in_alpha_threshold else 0)
                else:
                    height[j] = (
                        height[j] + 1
                        if in_image_mask[i, j] <= in_alpha_threshold else 0)

            while len(s) != 0 and height[s[-1]] >= height[j]:
                cur = s[-1]
                s.pop()
                _h = height[cur]
                _w = j if len(s) == 0 else j - s[-1] - 1
                curArea = _h * _w
                if curArea > res:
                    res = curArea
                    i1 = i
                    i0 = i1 - _h + 1
                    j1 = j - 1
                    j0 = j1 - _w + 1
            s.append(j)

    out_rect = (
        j0,
        i0,
        j1 - j0 + 1,
        i1 - i0 + 1,
    )
    return out_rect, res


def scale_rect(in_rect, in_image_size, in_scale):
    tlX = int(in_rect[0] * in_scale + 0.5)
    tlY = int(in_rect[1] * in_scale + 0.5)
    in_image_size_h, in_image_size_w = in_image_size.shape[0:2]
    brX = min(int(in_rect[2] * in_scale + 0.5), in_image_size_w)
    brY = min(int(in_rect[3] * in_scale + 0.5), in_image_size_h)
    out_rect = (tlX, tlY, brX - tlX, brY - tlY)
    return out_rect


def get_fast_valid_rect(in_mask, in_threshold=0):
    # mask: np.array [0~1]
    in_mask = in_mask > in_threshold
    locations = cv2.findNonZero(in_mask.astype(np.uint8))
    output_rect = cv2.boundingRect(locations)  # x,y,w,h
    return output_rect


def min_size_match(in_image, in_min_size, type=cv2.INTER_LINEAR):
    resize_image = in_image.copy()
    width, height = in_min_size
    resize_img_height, resize_img_width = in_image.shape[0:2]
    height_scale = height / resize_img_height
    widht_scale = width / resize_img_width
    scale = height_scale if height_scale > widht_scale else widht_scale
    new_size = (
        max(int(resize_img_width * scale + 0.5), 1),
        max(int(resize_img_height * scale + 0.5), 1),
    )

    resize_image = cv2.resize(resize_image, new_size, 0, 0, type)
    return resize_image


def center_crop(in_image, in_size):
    in_size_w, in_size_h = in_size
    in_image_h, in_image_w = in_image.shape[0:2]

    half_height = (in_image_h - in_size_h) // 2
    half_width = (in_image_w - in_size_w) // 2

    out_crop_image = in_image.copy()
    out_crop_image = out_crop_image[half_height:half_height + in_size_h,
                                    half_width:half_width + in_size_w]
    return out_crop_image


def safe_roi_pad(in_pad_image, in_rect, out_base_image):
    in_rect_x, in_rect_y, in_rect_w, in_rect_h = in_rect

    if in_rect_x < 0 or in_rect_y < 0 or in_rect_w <= 0 or in_rect_h <= 0:
        raise Exception('[safe_roi_pad] Failed!! x,y,w,h of rect are illegal')

    if in_rect_w != in_pad_image.shape[1] or in_rect_h != in_pad_image.shape[0]:
        raise Exception('[safe_roi_pad] Failed!!')

    if (in_rect_x + in_rect_w > out_base_image.shape[1]
            or in_rect_y + in_rect_h > out_base_image.shape[0]):
        raise Exception('[safe_roi_pad] Failed!!')

    out_base_image[in_rect_y:in_rect_y + in_rect_h,
                   in_rect_x:in_rect_x + in_rect_w] = in_pad_image


def merge_image(in_base_image, in_merge_image, in_merge_mask, in_point):
    if in_merge_image.shape[0:2] != in_merge_mask.shape[0:2]:
        raise Exception(
            '[merge_image] Failed!! in_merge_image.shape != in_merge_mask.shape!!'
        )

    in_point_x, in_point_y = in_point
    in_merge_image_rows, in_merge_image_cols = in_merge_image.shape[0:2]
    in_base_image_rows, in_base_image_cols = in_base_image.shape[0:2]

    if (in_point_x + in_merge_image_cols > in_base_image_cols
            or in_point_y + in_merge_image_rows > in_base_image_rows):
        raise Exception(
            '[merge_image] Failed!! merge_image:image rect not in image')

    base_roi_image = in_base_image[in_point_y:in_point_y + in_merge_image_rows,
                                   in_point_x:in_point_x
                                   + in_merge_image_cols, ]

    merge_image = in_merge_image.copy()
    merge_alpha = in_merge_mask.copy()
    base_roi_image = np.float32(base_roi_image)
    merge_alpha = np.repeat(merge_alpha[:, :, np.newaxis], 3, axis=2)
    merge_alpha = merge_alpha / 255.0

    base_roi_image = (
        1 - merge_alpha) * base_roi_image + merge_alpha * merge_image
    base_roi_image = np.clip(base_roi_image, 0, 255)
    base_roi_image = base_roi_image.astype('uint8')

    roi_rect = (in_point_x, in_point_y, in_merge_image_cols,
                in_merge_image_rows)
    safe_roi_pad(base_roi_image, roi_rect, in_base_image)
    return in_base_image


def blend_merge(in_scene_image,
                in_scene_mask,
                in_valid_sky_image,
                inBlendLevelNum=5):
    scene_sky_rect = get_fast_valid_rect(in_scene_mask, 1)
    area = scene_sky_rect[2] * scene_sky_rect[3]

    if area < IMAGE_BLENDER_MIN_VALID_SKY_AREA:
        raise Exception(
            '[blend_merge] Failed!! Scene Image Valid sky region is too small')

    valid_sky_image = min_size_match(in_valid_sky_image, scene_sky_rect[2:])
    valid_sky_image = center_crop(valid_sky_image, scene_sky_rect[2:])

    # resizeSceneMask
    sky_size = (
        max(
            int(in_scene_mask.shape[1] * IMAGE_BLENDER_MASK_RESIZE_SCALE
                + 0.5),
            IMAGE_BLENDER_MIN_RESIZE_DIM,
        ),
        max(
            int(in_scene_mask.shape[0] * IMAGE_BLENDER_MASK_RESIZE_SCALE
                + 0.5),
            IMAGE_BLENDER_MIN_RESIZE_DIM,
        ),
    )

    resize_scene_mask = cv2.resize(in_scene_mask, sky_size, cv2.INTER_LINEAR)
    resize_scene_mask = cv2.blur(
        resize_scene_mask,
        (IMAGE_BLENDER_BLUR_KERNEL_SIZE, IMAGE_BLENDER_BLUR_KERNEL_SIZE),
    )

    element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (IMAGE_BLENDER_BLUR_KERNEL_SIZE, IMAGE_BLENDER_BLUR_KERNEL_SIZE))
    sky_mask = cv2.dilate(resize_scene_mask, element)  # enlarge sky region
    scene_mask = cv2.erode(resize_scene_mask, element)  # enlarge scene region
    scene_mask = 255 - scene_mask

    sky_mask = cv2.resize(sky_mask, in_scene_mask.shape[0:2][::-1])
    scene_mask = cv2.resize(scene_mask, in_scene_mask.shape[0:2][::-1])

    x, y, w, h = scene_sky_rect
    valid_sky_mask = sky_mask[y:y + h, x:x + w]

    pano_sky_image = in_scene_image.copy()

    pano_sky_image = merge_image(pano_sky_image, valid_sky_image,
                                 valid_sky_mask, scene_sky_rect[0:2])
    blend_images = []
    blend_images.append(in_scene_image)
    blend_images.append(pano_sky_image)

    blend_masks = []
    blend_masks.append(scene_mask.astype(np.uint8))
    blend_masks.append(sky_mask.astype(np.uint8))

    panorama_rect = (0, 0, in_scene_image.shape[1], in_scene_image.shape[0])

    blender = cv2.detail_MultiBandBlender(1, inBlendLevelNum)
    blender.prepare(panorama_rect)

    for i in range(0, len(blend_images)):
        blender.feed(blend_images[i], blend_masks[i], (0, 0))
    pano_mask = (
        np.ones(
            (in_scene_image.shape[1], in_scene_image.shape[0]), dtype='uint8')
        * 255)
    out_blend_image = np.zeros_like(in_scene_image)
    result = blender.blend(out_blend_image, pano_mask)
    return result[0]
