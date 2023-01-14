# ------------------------------------------------------------------------
# Copyright (c) Alibaba, Inc. and its affiliates.
# ------------------------------------------------------------------------
import math
from math import exp
from typing import Dict

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group,
    module_name=Metrics.video_frame_interpolation_metric)
class VideoFrameInterpolationMetric(Metric):
    """The metric computation class for video frame interpolation,
    which will return PSNR, SSIM and LPIPS.
    """
    pred_name = 'pred'
    label_name = 'target'

    def __init__(self):
        super(VideoFrameInterpolationMetric, self).__init__()
        self.preds = []
        self.labels = []
        self.loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs[VideoFrameInterpolationMetric.label_name]
        eval_results = outputs[VideoFrameInterpolationMetric.pred_name]
        self.preds.append(eval_results)
        self.labels.append(ground_truths)

    def evaluate(self):
        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for (pred, label) in zip(self.preds, self.labels):
                # norm to 0-1
                height, width = label.size(2), label.size(3)
                pred = pred[:, :, 0:height, 0:width]

                psnr_list.append(calculate_psnr(label, pred))
                ssim_list.append(calculate_ssim(label, pred))
                lpips_list.append(
                    calculate_lpips(label, pred, self.loss_fn_alex))

        return {
            MetricKeys.PSNR: np.mean(psnr_list),
            MetricKeys.SSIM: np.mean(ssim_list),
            MetricKeys.LPIPS: np.mean(lpips_list)
        }

    def merge(self, other: 'VideoFrameInterpolationMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return self.preds, self.labels

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.labels = state


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window_3d(window_size, channel=1, device=None):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size,
                               window_size).contiguous().to(device)
    return window


def calculate_psnr(img1, img2):
    psnr = -10 * math.log10(
        torch.mean((img1[0] - img2[0]) * (img1[0] - img2[0])).cpu().data)
    return psnr


def calculate_ssim(img1,
                   img2,
                   window_size=11,
                   window=None,
                   size_average=True,
                   full=False,
                   val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(
            real_size, channel=1, device=img1.device).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(
        F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'),
        window,
        padding=padd,
        groups=1)
    mu2 = F.conv3d(
        F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'),
        window,
        padding=padd,
        groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(
        F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'),
        window,
        padding=padd,
        groups=1) - mu1_sq
    sigma2_sq = F.conv3d(
        F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'),
        window,
        padding=padd,
        groups=1) - mu2_sq
    sigma12 = F.conv3d(
        F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'),
        window,
        padding=padd,
        groups=1) - mu1_mu2

    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret.cpu()


def calculate_lpips(img1, img2, loss_fn_alex):
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    d = loss_fn_alex(img1, img2)
    return d.cpu().item()
