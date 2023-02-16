# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2

AL = 1
pas = 50
highpas = 250
index = 550


def gene_mask(f, angle, horizon):
    alpha = AL
    an = angle / 360 * (2 * np.pi)
    W, H = f.shape[:2]
    X = np.arange(0, W) - 255
    Y = np.arange(0, H) - 512
    x, y = np.meshgrid(Y, X)
    dis = np.sqrt(x**2 + y**2)
    dis[dis < pas] = 0
    dis[dis >= highpas] = 0
    dis[dis != 0] = 1

    maskband = np.zeros(f.shape)
    maskband[:, index:index + 200] = 1

    angles = np.abs(np.arctan(y / (x + 0.00001)))
    angles[angles > an] = 1
    angles[angles < an] = 0
    if not horizon:
        mask = angles
    else:
        mask = 1 - angles
    return alpha * mask * maskband


def normal(rgb_recons, edge=False):
    rgb_recons = rgb_recons.astype(int)
    rgb_recons = (rgb_recons - np.min(rgb_recons)) / (
        np.max(rgb_recons) - np.min(rgb_recons)) * 255
    return rgb_recons


def fourier_gray(img):
    img = Image.fromarray(img.astype(np.uint8)).convert('L')
    img = np.array(img)
    im = img
    x = im * 1
    y = fft2(x, axes=(0, 1))
    shift2center = np.fft.fftshift(y)

    mask = np.zeros(y.shape)
    mask_angle = gene_mask(mask, 25, horizon=False)
    mask = mask_angle
    crop1 = shift2center * mask
    iresult = np.fft.ifftshift(crop1)
    recons = np.abs(ifft2(iresult, axes=(0, 1)))
    rgb_recons = recons

    mask = np.zeros(y.shape)
    mask_angle = gene_mask(mask, 20, horizon=True)
    mask = mask_angle
    crop1 = shift2center * mask
    iresult = np.fft.ifftshift(crop1)
    recons = np.abs(ifft2(iresult, axes=(0, 1)))
    rgb_reconsH = recons

    rgb_recons = normal(rgb_recons, True)

    rgb_reconsH = normal(rgb_reconsH, True)

    rgb_reconsA = rgb_reconsH * 0

    x = np.concatenate((rgb_recons[:, :, None], rgb_reconsH[:, :, None],
                        rgb_reconsA[:, :, None]), 2)

    return x


def fourier(img):
    rgb_recons = np.zeros(img.shape)
    index = 520
    for k in range(3):
        im = img[:, :, k:k + 1]
        x = im * 1
        y = fft2(x)
        shift2center = np.fft.fftshift(y)

        mask = np.zeros(y.shape)
        mask[:, index:index + 200, :] = 1

        crop1 = shift2center * mask
        iresult = np.fft.ifftshift(crop1)
        recons = np.abs(ifft2(iresult))
        rgb_recons[:, :, k:k + 1] = recons

    rgb_recons = (rgb_recons - np.min(rgb_recons)) / (
        np.max(rgb_recons) - np.min(rgb_recons))
    return rgb_recons
