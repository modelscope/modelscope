# The code is modified based on BasicSR metrics:
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py
from typing import Dict

import cv2
import numpy as np

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.image_denoise_metric)
class ImageDenoiseMetric(Metric):
    """The metric computation class for image denoise classes.
    """
    pred_name = 'pred'
    label_name = 'target'

    def __init__(self):
        self.preds = []
        self.labels = []

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs[ImageDenoiseMetric.label_name]
        eval_results = outputs[ImageDenoiseMetric.pred_name]
        self.preds.append(
            torch_nested_numpify(torch_nested_detach(eval_results)))
        self.labels.append(
            torch_nested_numpify(torch_nested_detach(ground_truths)))

    def evaluate(self):
        psnr_list, ssim_list = [], []
        for (pred, label) in zip(self.preds, self.labels):
            psnr_list.append(calculate_psnr(label[0], pred[0], crop_border=0))
            ssim_list.append(calculate_ssim(label[0], pred[0], crop_border=0))
        return {
            MetricKeys.PSNR: np.mean(psnr_list),
            MetricKeys.SSIM: np.mean(ssim_list)
        }


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def calculate_psnr(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (
        f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


def calculate_ssim(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate SSIM (structural similarity).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (
        f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5,
                                        5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    tmp1 = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    tmp2 = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = tmp1 / tmp2
    return ssim_map.mean()
