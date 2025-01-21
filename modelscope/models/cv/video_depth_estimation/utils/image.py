# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as funct
from PIL import Image

from modelscope.models.cv.video_depth_estimation.utils.misc import same_shape


def parse_video(video_file, save_root, sample_rate=10):
    os.makedirs(save_root, exist_ok=True)

    cap = cv2.VideoCapture(video_file)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print('Error opening video stream or file')
    count = 0
    sample_count = 0

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if count % sample_rate == 0:
                save_path = os.path.join(save_root,
                                         f'{sample_count}'.zfill(6) + '.jpg')
                cv2.imwrite(save_path, img)
                sample_count += 1
            count += 1
        else:
            break
    print(
        f'video total frames num: {count},  sampled frames num:{sample_count}')


def get_intrinsics(image_shape_raw, image_shape, data_type):
    if data_type == 'kitti':
        intr = np.array([
            7.215376999999999725e+02, 0.000000000000000000e+00,
            6.095593000000000075e+02, 0.000000000000000000e+00,
            7.215376999999999725e+02, 1.728540000000000134e+02,
            0.000000000000000000e+00, 0.000000000000000000e+00,
            1.000000000000000000e+00
        ]).reshape(3, 3)
    elif data_type == 'indoor':
        intr = np.array([
            1170.187988, 0.000000, 647.750000, 0.000000, 1170.187988,
            483.750000, 0.000000, 0.000000, 1.000000
        ]).reshape(3, 3)
    else:
        # print("fake intrinsics")
        w, h = image_shape_raw
        fx = w * 1.2
        fy = w * 1.2
        cx = w / 2.0
        cy = h / 2.0
        intr = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    orig_w, orig_h = image_shape_raw
    out_h, out_w = image_shape

    # Scale intrinsics
    intr[0] *= out_w / orig_w
    intr[1] *= out_h / orig_h

    return intr


def load_image(path):
    """
    Read an image using PIL

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : PIL.Image
        Loaded image
    """
    return Image.open(path)


def write_image(filename, image):
    """
    Write an image to file.

    Parameters
    ----------
    filename : str
        File where image will be saved
    image : np.array [H,W,3]
        RGB image
    """
    cv2.imwrite(filename, image[:, :, ::-1])


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def flip_lr_intr(intr, width):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert intr.shape[1:] == (3, 3)
    # trans = torch.eye(3, dtype=intr.dtype, device=intr.device)
    # trans[0, 0] = -1
    # intr_trans = torch.matmul(trans.unsqueeze(0), intr)
    intr[:, 0, 0] = -1 * intr[:, 0, 0]
    intr[:, 0, 2] = width - intr[:, 0, 2]
    return intr


def flip_model(model, image, flip):
    """
    Flip input image and flip output inverse depth map

    Parameters
    ----------
    model : nn.Module
        Module to be used
    image : torch.Tensor [B,3,H,W]
        Input image
    flip : bool
        True if the flip is happening

    Returns
    -------
    inv_depths : list of torch.Tensor [B,1,H,W]
        List of predicted inverse depth maps
    """
    if flip:
        return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
    else:
        return model(image)


def flip_mf_model(model, image, ref_imgs, intrinsics, flip):
    """
    Flip input image and flip output inverse depth map

    Parameters
    ----------
    model : nn.Module
        Module to be used
    image : torch.Tensor [B,3,H,W]
        Input image
    flip : bool
        True if the flip is happening

    Returns
    -------
    inv_depths : list of torch.Tensor [B,1,H,W]
        List of predicted inverse depth maps
    """
    if flip:
        if ref_imgs is not None:
            return model(
                flip_lr(image), [flip_lr(img) for img in ref_imgs], intrinsics)
        else:
            return model(flip_lr(image), None, intrinsics)
    else:
        return model(image, ref_imgs, intrinsics)


########################################################################################################################


def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    return image[:, :, :-1, :] - image[:, :, 1:, :]


########################################################################################################################


def interpolate_image(image, shape, mode='bilinear', align_corners=True):
    """
    Interpolate an image to a different resolution

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Image to be interpolated
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    image : torch.Tensor [B,?,H,W]
        Interpolated image
    """
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # If the shapes are the same, do nothing
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        # Interpolate image to match the shape
        return funct.interpolate(
            image, size=shape, mode=mode, align_corners=align_corners)


def interpolate_scales(images,
                       shape=None,
                       mode='bilinear',
                       align_corners=False):
    """
    Interpolate list of images to the same shape

    Parameters
    ----------
    images : list of torch.Tensor [B,?,?,?]
        Images to be interpolated, with different resolutions
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,H,W]
        Interpolated images, with the same resolution
    """
    # If no shape is provided, interpolate to highest resolution
    if shape is None:
        shape = images[0].shape
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # Interpolate all images
    return [
        funct.interpolate(
            image, shape, mode=mode, align_corners=align_corners)
        for image in images
    ]


def match_scales(image,
                 targets,
                 num_scales,
                 mode='bilinear',
                 align_corners=True):
    """
    Interpolate one image to produce a list of images with the same shape as targets

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Input image
    targets : list of torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(
                interpolate_image(
                    image,
                    target_shape,
                    mode=mode,
                    align_corners=align_corners))
    # Return scaled images
    return images


########################################################################################################################


@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid


########################################################################################################################
