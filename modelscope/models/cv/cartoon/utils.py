# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random

import cv2
import numpy as np
import tensorflow as tf


def resize_size(image, size=720):
    h, w, c = np.shape(image)
    if min(h, w) > size:
        if h > w:
            h, w = int(size * h / w), size
        else:
            h, w = size, int(size * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image


def padTo16x(image):
    h, w, c = np.shape(image)
    if h % 16 == 0 and w % 16 == 0:
        return image, h, w
    nh, nw = (h // 16 + 1) * 16, (w // 16 + 1) * 16
    img_new = np.ones((nh, nw, 3), np.uint8) * 255
    img_new[:h, :w, :] = image

    return img_new, h, w


def get_f5p(landmarks, np_img):
    eye_left = find_pupil(landmarks[36:41], np_img)
    eye_right = find_pupil(landmarks[42:47], np_img)
    if eye_left is None or eye_right is None:
        print('cannot find 5 points with find_puil, used mean instead.!')
        eye_left = landmarks[36:41].mean(axis=0)
        eye_right = landmarks[42:47].mean(axis=0)
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    f5p = [[eye_left[0], eye_left[1]], [eye_right[0], eye_right[1]],
           [nose[0], nose[1]], [mouth_left[0], mouth_left[1]],
           [mouth_right[0], mouth_right[1]]]
    return f5p


def find_pupil(landmarks, np_img):
    h, w, _ = np_img.shape
    xmax = int(landmarks[:, 0].max())
    xmin = int(landmarks[:, 0].min())
    ymax = int(landmarks[:, 1].max())
    ymin = int(landmarks[:, 1].min())

    if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > h or xmax > w:
        return None
    eye_img_bgr = np_img[ymin:ymax, xmin:xmax, :]
    eye_img = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.equalizeHist(eye_img)
    n_marks = landmarks - np.array([xmin, ymin]).reshape([1, 2])
    eye_mask = cv2.fillConvexPoly(
        np.zeros_like(eye_img), n_marks.astype(np.int32), 1)
    ret, thresh = cv2.threshold(eye_img, 100, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = (1 - thresh / 255.) * eye_mask
    cnt = 0
    xm = []
    ym = []
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i, j] > 0.5:
                xm.append(j)
                ym.append(i)
                cnt += 1
    if cnt != 0:
        xm.sort()
        ym.sort()
        xm = xm[cnt // 2]
        ym = ym[cnt // 2]
    else:
        xm = thresh.shape[1] / 2
        ym = thresh.shape[0] / 2

    return xm + xmin, ym + ymin


def next_batch(filename_list, batch_size, fineSize=256):
    idx = np.arange(0, len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        h, w, c = image.shape
        rw = random.randint(0, w - fineSize)
        rh = random.randint(0, h - fineSize)
        image = image[rh:rh + fineSize, rw:rw + fineSize, :]
        image = image.astype(np.float32) / 127.5 - 1
        batch_data.append(image)

    return np.asarray(batch_data)


def read_image(image_path, IMAGE_SIZE=256):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image[..., ::-1]
    # image = image / 127.5 - 1
    image = (image - 0.5) * 2

    return image


def load_data(photo_list):
    photo = read_image(photo_list)
    return photo


def tf_data_loader(image_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_list))
    dataset = dataset.shuffle(len(image_list))
    dataset = dataset.map(load_data, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def write_batch_image(image, save_dir, name, n):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k] + 1) * 127.5
            image[k] = np.clip(image[k], 0, 255)
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image.astype(np.uint8))


def grid_batch_image(image, n):
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k] + 1) * 127.5
            image[k] = np.clip(image[k], 0, 255)
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    return fused_image


def all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG':
                L.append(os.path.join(root, file))
    return L
