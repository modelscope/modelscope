# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

__all__ = ['make_irregular_mask', 'make_rectangle_mask', 'make_uncrop']


def make_irregular_mask(w,
                        h,
                        max_angle=4,
                        max_length=200,
                        max_width=100,
                        min_strokes=1,
                        max_strokes=5,
                        mode='line'):
    # initialize mask
    assert mode in ['line', 'circle', 'square']
    mask = np.zeros((h, w), np.float32)

    # draw strokes
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        x1 = np.random.randint(w)
        y1 = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_length)
            radius = 5 + np.random.randint(max_width)
            x2 = np.clip((x1 + length * np.sin(angle)).astype(np.int32), 0, w)
            y2 = np.clip((y1 + length * np.cos(angle)).astype(np.int32), 0, h)
            if mode == 'line':
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, radius)
            elif mode == 'circle':
                cv2.circle(
                    mask, (x1, y1), radius=radius, color=1.0, thickness=-1)
            elif mode == 'square':
                radius = radius // 2
                mask[y1 - radius:y1 + radius, x1 - radius:x1 + radius] = 1
            x1, y1 = x2, y2
    return mask


def make_rectangle_mask(w,
                        h,
                        margin=10,
                        min_size=30,
                        max_size=150,
                        min_strokes=1,
                        max_strokes=4):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # draw rectangles
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        box_w = np.random.randint(min_size, max_size)
        box_h = np.random.randint(min_size, max_size)
        x1 = np.random.randint(margin, w - margin - box_w + 1)
        y1 = np.random.randint(margin, h - margin - box_h + 1)
        mask[y1:y1 + box_h, x1:x1 + box_w] = 1
    return mask


def make_uncrop(w, h):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # randomly halve the image
    side = np.random.choice([0, 1, 2, 3])
    if side == 0:
        mask[:h // 2, :] = 1
    elif side == 1:
        mask[h // 2:, :] = 1
    elif side == 2:
        mask[:, :w // 2] = 1
    elif side == 3:
        mask[:, w // 2:] = 1
    return mask
