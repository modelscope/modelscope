# Copyright © Alibaba, Inc. and its affiliates.

import math
import random

import cv2
import numpy as np
import torch

from modelscope.models.cv.tinynas_detection.damo.structures.bounding_box import \
    BoxList
from modelscope.models.cv.tinynas_detection.damo.utils import adjust_box_anns


def xyn2xy(x, scale, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = scale * x[:, 0] + padw  # top left x
    y[:, 1] = scale * x[:, 1] + padh  # top left y
    return y


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([
            np.interp(x, xp, s[:, i]) for i in range(2)
        ]).reshape(2, -1).T  # segment xy
    return segments


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint,
    # i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(),
                     y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            'Affine params should be either a sequence containing two values\
                          or single float values. Got {}'.format(value))


def box_candidates(box1,
                   box2,
                   wh_thr=2,
                   ar_thr=20,
                   area_thr=0.1,
                   eps=1e-16):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    valid_w = w2 > wh_thr
    valid_h = h2 > wh_thr
    valid_ar = (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
    return valid_w & valid_h & valid_ar


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear,
                         translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = get_aug_params(scale, center=1.0)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi
                       / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi
                       / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(
        0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1,
      2] = random.uniform(0.5 - translate, 0.5
                          + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def random_affine(
        img,
        targets=(),
        segments=None,
        target_size=(640, 640),
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
):
    M, scale = get_transform_matrix(img.shape[:2], target_size, degrees,
                                    scales, shear, translate)

    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(
            img, M[:2], dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if (n and len(segments) == 0) or (len(segments) != len(targets)):
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, target_size[0])
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, target_size[1])

    else:
        segments = resample_segments(segments)  # upsample
        new = np.zeros((len(targets), 4))
        assert len(segments) <= len(targets)
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = xy[:, :2]  # perspective rescale or affine
            # clip
            new[i] = segment2box(xy, target_size[0], target_size[1])

    # filter candidates
    i = box_candidates(
        box1=targets[:, 0:4].T * scale, box2=new.T, area_thr=0.1)
    targets = targets[i]
    targets[:, 0:4] = new[i]

    return img, targets


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h,
                          input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w,
                                     input_w * 2), min(input_h * 2,
                                                       yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicWrapper(torch.utils.data.dataset.Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self,
                 dataset,
                 img_size,
                 mosaic_prob=1.0,
                 mixup_prob=1.0,
                 transforms=None,
                 degrees=10.0,
                 translate=0.1,
                 mosaic_scale=(0.1, 2.0),
                 mixup_scale=(0.5, 1.5),
                 shear=2.0,
                 *args):
        super().__init__()
        self._dataset = dataset
        self.input_dim = img_size
        self._transforms = transforms
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            enable_mosaic_mixup = inp[0]
            idx = inp[1]
        else:
            enable_mosaic_mixup = False
            idx = inp
        img, labels, segments, img_id = self._dataset.pull_item(idx)

        if enable_mosaic_mixup:
            if random.random() < self.mosaic_prob:
                mosaic_labels = []
                mosaic_segments = []
                input_h, input_w = self.input_dim[0], self.input_dim[1]

                yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

                # 3 additional image indices
                indices = [idx] + [
                    random.randint(0,
                                   len(self._dataset) - 1) for _ in range(3)
                ]

                for i_mosaic, index in enumerate(indices):
                    img, _labels, _segments, img_id = self._dataset.pull_item(
                        index)
                    h0, w0 = img.shape[:2]  # orig hw
                    scale = min(1. * input_h / h0, 1. * input_w / w0)
                    img = cv2.resize(
                        img, (int(w0 * scale), int(h0 * scale)),
                        interpolation=cv2.INTER_LINEAR)
                    # generate output mosaic image
                    (h, w, c) = img.shape[:3]
                    if i_mosaic == 0:
                        mosaic_img = np.full((input_h * 2, input_w * 2, c),
                                             114,
                                             dtype=np.uint8)  # pad 114

                    (l_x1, l_y1, l_x2,
                     l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                         mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w)

                    mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2,
                                                           s_x1:s_x2]
                    padw, padh = l_x1 - s_x1, l_y1 - s_y1

                    labels = _labels.copy()
                    # Normalized xywh to pixel xyxy format
                    if _labels.size > 0:
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2] + padw
                        labels[:, 3] = scale * _labels[:, 3] + padh
                    segments = [
                        xyn2xy(x, scale, padw, padh) for x in _segments
                    ]
                    mosaic_segments.extend(segments)
                    mosaic_labels.append(labels)

                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(
                        mosaic_labels[:, 0],
                        0,
                        2 * input_w,
                        out=mosaic_labels[:, 0])
                    np.clip(
                        mosaic_labels[:, 1],
                        0,
                        2 * input_h,
                        out=mosaic_labels[:, 1])
                    np.clip(
                        mosaic_labels[:, 2],
                        0,
                        2 * input_w,
                        out=mosaic_labels[:, 2])
                    np.clip(
                        mosaic_labels[:, 3],
                        0,
                        2 * input_h,
                        out=mosaic_labels[:, 3])

                if len(mosaic_segments):
                    assert input_w == input_h
                    for x in mosaic_segments:
                        np.clip(
                            x, 0, 2 * input_w,
                            out=x)  # clip when using random_perspective()

                img, labels = random_affine(
                    mosaic_img,
                    mosaic_labels,
                    mosaic_segments,
                    target_size=(input_w, input_h),
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (not len(labels) == 0 and random.random() < self.mixup_prob):
                img, labels = self.mixup(img, labels, self.input_dim)

            # transfer labels to BoxList
            h_tmp, w_tmp = img.shape[:2]
            boxes = np.array([label[:4] for label in labels])
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            valid_idx = areas > 4

            target = BoxList(boxes[valid_idx], (w_tmp, h_tmp), mode='xyxy')

            classes = [label[4] for label in labels]
            classes = torch.tensor(classes)[valid_idx]
            target.add_field('labels', classes.long())

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return img, target, img_id

        else:
            return self._dataset.__getitem__(idx)

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3),
                             dtype=np.uint8) * 114  # pad 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114  # pad 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0],
                             input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio),
             int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[:int(img.shape[0]
                    * cp_scale_ratio), :int(img.shape[1]
                                            * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor),
             int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3),
            dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(),
                                              cp_scale_ratio, 0, 0, origin_w,
                                              origin_h)
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
            np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def get_img_info(self, index):
        return self._dataset.get_img_info(index)
