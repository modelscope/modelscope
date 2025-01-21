# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
import random
from datetime import datetime

import numpy as np
import scipy
import scipy.stats as stats
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.linalg import orth
from torchvision.utils import make_grid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__all__ = ['degradation_bsrgan_light', 'degradation_bsrgan']


# get uint8 image of size HxWxn_channles (RGB)
def imread_uint(path, n_channels=3):
    # input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def uint162single(img):
    return np.float32(img / 65535.)


def single2uint16(img):
    return np.uint16((img.clip(0, 1) * 65535.).round())


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                          [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [
                              -222.921, 135.576, -276.836
                          ]  # noqa
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


# PSNR
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# SSIM
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1)  # noqa
                * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)  # noqa
                                         *  # noqa
                                         (sigma1_sq + sigma2_sq + C2))  # noqa
    return ssim_map.mean()


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5 * absx3 + 2.5 * absx2 - 4*absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))  # noqa


def calculate_weights_indices(in_length, out_length, scale, kernel,
                              kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# imresize for tensor image [0, 1]
def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: pytorch tensor, CHW or HW [0,1]
    # output: CHW or HW [0,1] w/o round
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(0)
    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W
                                                                   * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(
                0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :,
                                       idx:idx + kernel_width].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()
    return out_2


# imresize for numpy image [0, 1]
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W
                                                                   * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :,
                                     j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width,
                                       j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()


def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor
    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)
    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = stats.multivariate_normal.pdf([cx, cy],
                                                    mean=mean,
                                                    cov=cov)

    k = k / np.sum(k)
    return k


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1, c, 1, 1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(
        x, k, bias=None, stride=1, padding=0, groups=n * c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x


def gen_kernel(
        k_size=np.array([15, 15]),
        scale_factor=np.array([4, 4]),
        min_var=0.6,
        max_var=10.,
        noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5 * (scale_factor - 1)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(
        np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor
    Return:
        bicubicly downsampled LR image
    '''
    x = imresize_np(x, scale=1 / sf)
    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling
    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor
    Return:
        downsampled LR image
    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):
    ''' bicubic downsampling + blur
    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor
    Return:
        downsampled LR image
    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling
    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor
    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    st = 0
    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur_1(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf

    wd2 = wd2 / 4
    wd = wd / 4

    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_Gaussian(
            ksize=random.randint(2, 11) + 3,
            theta=random.random() * np.pi,
            l1=l1,
            l2=l2)
    else:
        k = fspecial('gaussian',
                     random.randint(2, 4) + 3, wd * random.random())
    img = ndimage.filters.convolve(
        img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(
        img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])),
        interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:
        img = img + np.random.normal(0, noise_level / 255.0, img.shape).astype(
            np.float32)
    elif rnum < 0.4:
        img = img + np.random.normal(0, noise_level / 255.0,
                                     (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img = img + np.random.multivariate_normal([0, 0, 0], np.abs(
            L**2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255.0,
                                      img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255.0,
                                      (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal(
            [0, 0, 0], np.abs(L**2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2 * random.random() + 2.0)
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(
            np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(80, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(
        '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h - lq_patchsize)
    rnd_w = random.randint(0, w - lq_patchsize)
    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf,
            rnd_w_H:rnd_w_H + lq_patchsize * sf, :]
    return lq, hq


def degradation_bsrgan_light(image, sf=4, isp_model=None):
    """
    This is the variant of the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    sf: scale factor
    isp_model: camera ISP model
    Returns
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    image = uint2single(image)
    _, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25

    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]
    h, w = image.shape[:2]

    if sf == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            image = cv2.resize(
                image,
                (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])),
                interpolation=random.choice([1, 2, 3]))
        else:
            image = imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[
            idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0:
            image = add_blur_1(image, sf=sf)
        elif i == 2:
            a, b = image.shape[1], image.shape[0]
            # downsample2
            if random.random() < 0.8:
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(
                    image, (int(1 / sf1 * image.shape[1]),
                            int(1 / sf1 * image.shape[0])),
                    interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                image = ndimage.filters.convolve(
                    image, np.expand_dims(k_shifted, axis=2), mode='mirror')
                image = image[0::sf, 0::sf, ...]
            image = np.clip(image, 0.0, 1.0)
        elif i == 3:
            # downsample3
            image = cv2.resize(
                image, (int(1 / sf * a), int(1 / sf * b)),
                interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)
        elif i == 4:
            # add Gaussian noise
            image = add_Gaussian_noise(image, noise_level1=1, noise_level2=2)
        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                image = add_JPEG_noise(image)

    # add final JPEG compression noise
    image = add_JPEG_noise(image)
    image = single2uint(image)
    return image


def add_blur_2(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf
    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_Gaussian(
            ksize=2 * random.randint(2, 11) + 3,
            theta=random.random() * np.pi,
            l1=l1,
            l2=l2)
    else:
        k = fspecial('gaussian', 2 * random.randint(2, 11) + 3,
                     wd * random.random())
    img = ndimage.filters.convolve(
        img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def degradation_bsrgan(image, sf=4, isp_model=None):
    """
    This is the variant of the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    sf: scale factor
    isp_model: camera ISP model
    Returns
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    image = uint2single(image)
    _, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25

    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]
    h, w = image.shape[:2]

    if sf == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            image = cv2.resize(
                image,
                (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])),
                interpolation=random.choice([1, 2, 3]))
        else:
            image = imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[
            idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0:
            image = add_blur_2(image, sf=sf)
        elif i == 1:
            image = add_blur_2(image, sf=sf)
        elif i == 2:
            a, b = image.shape[1], image.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(
                    image, (int(1 / sf1 * image.shape[1]),
                            int(1 / sf1 * image.shape[0])),
                    interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                image = ndimage.filters.convolve(
                    image, np.expand_dims(k_shifted, axis=2), mode='mirror')
                image = image[0::sf, 0::sf, ...]
            image = np.clip(image, 0.0, 1.0)
        elif i == 3:
            # downsample3
            image = cv2.resize(
                image, (int(1 / sf * a), int(1 / sf * b)),
                interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)
        elif i == 4:
            # add Gaussian noise
            image = add_Gaussian_noise(image, noise_level1=2, noise_level2=25)
        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                image = add_JPEG_noise(image)

    # add final JPEG compression noise
    image = add_JPEG_noise(image)
    image = single2uint(image)
    return image
