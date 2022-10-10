# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFile

from .face import FaceDetector

ImageFile.LOAD_TRUNCATED_IMAGES = True


def adjust_bx_v2(box, w, h):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    box_w = x2 - x1
    box_h = y2 - y1
    delta = abs(box_w - box_h)
    if box_w > box_h:
        if y1 >= delta:
            y1 = y1 - delta
        else:
            delta_y1 = y1
            y1 = 0
            delta_y2 = delta - delta_y1
            y2 = y2 + delta_y2 if y2 < h - delta_y2 else h - 1
    else:
        if x1 >= delta / 2 and x2 <= w - delta / 2:
            x1 = x1 - delta / 2
            x2 = x2 + delta / 2
        elif x1 < delta / 2 and x2 <= w - delta / 2:
            delta_x1 = x1
            x1 = 0
            delta_x2 = delta - delta_x1
            x2 = x2 + delta_x2 if x2 < w - delta_x2 else w - 1
        elif x1 >= delta / 2 and x2 > w - delta / 2:
            delta_x2 = w - x2
            x2 = w - 1
            delta_x1 = delta - x1
            x1 = x1 - delta_x1 if x1 >= delta_x1 else 0

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return [x1, y1, x2, y2]


def face_detection_PIL_v2(image, face_model):
    crop_size = 112
    face_detector = FaceDetector(face_model)
    img = np.array(image)
    h, w = img.shape[0:2]
    bxs, conf = face_detector.do_detect(img)
    bx = bxs[0]
    bx = adjust_bx_v2(bx, w, h)
    x1, y1, x2, y2 = bx
    image = img[y1:y2, x1:x2, :]
    img = Image.fromarray(image)
    img = img.resize((crop_size, crop_size))
    bx = tuple(bx)
    return img, bx
