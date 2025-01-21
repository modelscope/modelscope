# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/color_augs.py
# Copyright © Alibaba, Inc. and its affiliates.

import random

import torch
import torch.nn.functional as F

from .gaussian_maps import _merge_gaussian

_MAX_LEVEL = 10.0


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 1.0.
    """

    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp

    # Extrapolate:
    #
    # We need to clip and then cast.
    return torch.clamp(temp, 0.0, 1.0)


def solarize(image, threshold=0.5):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return torch.where(image <= threshold, image, 1.0 - image)


def solarize_add(image, addition=0, threshold=0.5):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = image + addition
    added_image = torch.clamp(added_image, 0.0, 1.0)
    return torch.where(image <= threshold, added_image, image)


def rgb2gray(rgb):
    gray = rgb[0] * 0.2989 + rgb[1] * 0.5870 + rgb[2] * 0.1140
    gray = gray.unsqueeze(0).repeat((3, 1, 1))
    return gray


def color(img, factor):
    """Equivalent of PIL Color."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    degenerate = rgb2gray(img)
    return blend(degenerate, img, factor)


def contrast(img, factor):
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb2gray(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    return blend(mean, img, max(factor, 1e-6))


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = torch.zeros(image.shape)
    return blend(degenerate, image, factor)


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    if image.shape[0] == 0 or image.shape[1] == 0:
        return image
    channels = image.shape[0]
    kernel = torch.Tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]]).reshape(
        1, 1, 3, 3) / 13.0
    kernel = kernel.repeat((3, 1, 1, 1))
    image_newaxis = image.unsqueeze(0)
    image_pad = F.pad(image_newaxis, (1, 1, 1, 1), mode='reflect')
    degenerate = F.conv2d(image_pad, weight=kernel, groups=channels).squeeze(0)
    return blend(degenerate, image, factor)


def equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/
    autoaugment.py#L352"""
    image = image * 255

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[c, :, :]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(
                build_lut(histo, step), 0,
                im.flatten().long())
            result = result.reshape_as(im)

        return result  # .type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 0) / 255.0
    return image


def autocontrast(image):

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        lo = torch.min(image)
        hi = torch.max(image)

        # Scale the image, making the lowest value 0 and the highest value 1.
        def scale_values(im):
            scale = 1.0 / (hi - lo)
            offset = -lo * scale
            im = im * scale + offset
            im = torch.clamp(im, 0.0, 1.0)
            return im

        if hi > lo:
            result = scale_values(image)
        else:
            result = image

        return result

    # Assumes RGB for now. Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[0, :, :])
    s2 = scale_channel(image[1, :, :])
    s3 = scale_channel(image[2, :, :])
    image = torch.stack([s1, s2, s3], 0)
    return image


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    image *= 255
    image = image.long()
    shift = bits  # 8 - bits
    image_rightshift = image >> shift
    image_leftshift = image_rightshift << shift
    image_leftshift = image_leftshift.float() / 255.0
    return image_leftshift


def _color_aug_func(img, img_aug, target, scale_ratios_splits,
                    box_sample_probs):
    scale_ratios, scale_splits = scale_ratios_splits
    boxes = [
        bbox for i, bbox in enumerate(target.bbox)
        if random.random() < box_sample_probs[i]
    ]
    img_aug = _merge_gaussian(img, img_aug, boxes, scale_ratios, scale_splits)
    return img_aug


color_aug_func = {
    'AutoContrast':
    lambda x, level, target,
    scale_ratios_splits, box_sample_probs: _color_aug_func(
        x, autocontrast(x), target, scale_ratios_splits, box_sample_probs),
    'Equalize':
    lambda x, leve, target,
    scale_ratios_splits, box_sample_probs: _color_aug_func(
        x, equalize(x), target, scale_ratios_splits, box_sample_probs),
    'SolarizeAdd':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, solarize_add(x, level / _MAX_LEVEL * 0.4296875), target,
                    scale_ratios_splits, box_sample_probs),
    'Color':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, color(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Contrast':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, contrast(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Brightness':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, brightness(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Sharpness':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, sharpness(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
}
