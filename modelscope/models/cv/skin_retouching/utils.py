# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'gen_diffuse_mask', 'get_crop_bbox', 'get_roi_without_padding',
    'patch_aggregation_overlap', 'patch_partition_overlap', 'preprocess_roi',
    'resize_on_long_side', 'roi_to_tensor', 'smooth_border_mg', 'whiten_img'
]


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(
            img, (int(src_width * scale), long_side),
            interpolation=cv2.INTER_LINEAR)
    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(
            img, (long_side, int(src_height * scale)),
            interpolation=cv2.INTER_LINEAR)

    return _img, scale


def get_crop_bbox(detecting_results):
    boxes = []
    for anno in detecting_results:
        if anno['score'] == -1:
            break
        boxes.append({
            'x1': anno['bbox'][0],
            'y1': anno['bbox'][1],
            'x2': anno['bbox'][2],
            'y2': anno['bbox'][3]
        })
    face_count = len(boxes)

    suitable_bboxes = []
    for i in range(face_count):
        face_bbox = boxes[i]

        face_bbox_width = abs(face_bbox['x2'] - face_bbox['x1'])
        face_bbox_height = abs(face_bbox['y2'] - face_bbox['y1'])

        face_bbox_center = ((face_bbox['x1'] + face_bbox['x2']) / 2,
                            (face_bbox['y1'] + face_bbox['y2']) / 2)

        square_bbox_length = face_bbox_height if face_bbox_height > face_bbox_width else face_bbox_width
        enlarge_ratio = 1.5
        square_bbox_length = int(enlarge_ratio * square_bbox_length)

        sideScale = 1

        square_bbox = {
            'x1':
            int(face_bbox_center[0] - sideScale * square_bbox_length / 2),
            'x2':
            int(face_bbox_center[0] + sideScale * square_bbox_length / 2),
            'y1':
            int(face_bbox_center[1] - sideScale * square_bbox_length / 2),
            'y2': int(face_bbox_center[1] + sideScale * square_bbox_length / 2)
        }

        suitable_bboxes.append(square_bbox)

    return suitable_bboxes


def get_roi_without_padding(img, bbox):
    crop_t = max(bbox['y1'], 0)
    crop_b = min(bbox['y2'], img.shape[0])
    crop_l = max(bbox['x1'], 0)
    crop_r = min(bbox['x2'], img.shape[1])
    roi = img[crop_t:crop_b, crop_l:crop_r]
    return roi, 0, [crop_t, crop_b, crop_l, crop_r]


def roi_to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))[None, ...]

    return img


def preprocess_roi(img):
    img = img.float() / 255.0
    img = (img - 0.5) * 2

    return img


def patch_partition_overlap(image, p1, p2, padding=32):

    B, C, H, W = image.size()
    h, w = H // p1, W // p2
    image = F.pad(
        image,
        pad=(padding, padding, padding, padding, 0, 0),
        mode='constant',
        value=0)

    patch_list = []
    for i in range(h):
        for j in range(w):
            patch = image[:, :, p1 * i:p1 * (i + 1) + padding * 2,
                          p2 * j:p2 * (j + 1) + padding * 2]
            patch_list.append(patch)

    output = torch.cat(
        patch_list, dim=0)  # (b h w) c (p1 + 2 * padding) (p2 + 2 * padding)
    return output


def patch_aggregation_overlap(image, h, w, padding=32):

    image = image[:, :, padding:-padding, padding:-padding]

    output = rearrange(image, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=h, w=w)

    return output


def smooth_border_mg(diffuse_mask, mg):
    mg = mg - 0.5
    diffuse_mask = F.interpolate(
        diffuse_mask, mg.shape[:2], mode='bilinear')[0].permute(1, 2, 0)
    mg = mg * diffuse_mask
    mg = mg + 0.5
    return mg


def whiten_img(image, skin_mask, whitening_degree, flag_bigKernal=False):
    """
    image: rgb
    """
    dilate_kernalsize = 30
    if flag_bigKernal:
        dilate_kernalsize = 80
    new_kernel1 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernalsize, dilate_kernalsize))
    new_kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernalsize, dilate_kernalsize))
    if len(skin_mask.shape) == 3:
        skin_mask = skin_mask[:, :, -1]
    skin_mask = cv2.dilate(skin_mask, new_kernel1, 1)
    skin_mask = cv2.erode(skin_mask, new_kernel2, 1)
    skin_mask = cv2.blur(skin_mask, (20, 20)) / 255.0
    skin_mask = skin_mask.squeeze()
    skin_mask = torch.from_numpy(skin_mask).to(image.device)
    skin_mask = torch.stack([skin_mask, skin_mask, skin_mask], dim=0)[None,
                                                                      ...]
    skin_mask[:, 1:, :, :] *= 0.75

    whiten_mg = skin_mask * 0.2 * whitening_degree + 0.5
    assert len(whiten_mg.shape) == 4
    whiten_mg = F.interpolate(
        whiten_mg, image.shape[:2], mode='bilinear')[0].permute(1, 2,
                                                                0).half()
    output_pred = image.half()
    output_pred = output_pred / 255.0
    output_pred = (
        -2 * whiten_mg + 1
    ) * output_pred * output_pred + 2 * whiten_mg * output_pred  # value: 0~1
    output_pred = output_pred * 255.0
    output_pred = output_pred.byte()

    output_pred = output_pred.cpu().numpy()
    return output_pred


def gen_diffuse_mask(out_channels=3):
    mask_size = 500
    diffuse_with = 20
    a = np.ones(shape=(mask_size, mask_size), dtype=np.float32)

    for i in range(mask_size):
        for j in range(mask_size):
            if i >= diffuse_with and i <= (
                    mask_size - diffuse_with) and j >= diffuse_with and j <= (
                        mask_size - diffuse_with):
                a[i, j] = 1.0
            elif i <= diffuse_with:
                a[i, j] = i * 1.0 / diffuse_with
            elif i > (mask_size - diffuse_with):
                a[i, j] = (mask_size - i) * 1.0 / diffuse_with

    for i in range(mask_size):
        for j in range(mask_size):
            if j <= diffuse_with:
                a[i, j] = min(a[i, j], j * 1.0 / diffuse_with)
            elif j > (mask_size - diffuse_with):
                a[i, j] = min(a[i, j], (mask_size - j) * 1.0 / diffuse_with)
    a = np.dstack([a] * out_channels)
    return a


def pad_to_size(
    target_size: Tuple[int, int],
    image: np.array,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    """Pads the image on the sides to the target_size

    Args:
        target_size: (target_height, target_width)
        image:
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f'Target width should bigger than image_width'
                         f'We got {target_width} {image_width}')

    if target_height < image_height:
        raise ValueError(f'Target height should bigger than image_height'
                         f'We got {target_height} {image_height}')

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        'pads': (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        'image':
        cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad,
                           cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result['bboxes'] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result['keypoints'] = keypoints

    return result


def unpad_from_size(
    pads: Tuple[int, int, int, int],
    image: Optional[np.array] = None,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crops patch from the center so that sides are equal to pads.

    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns: cropped image

    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result['image'] = image[y_min_pad:height - y_max_pad,
                                x_min_pad:width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result['bboxes'] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result['keypoints'] = keypoints

    return result
