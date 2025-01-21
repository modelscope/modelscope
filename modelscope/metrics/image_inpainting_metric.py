"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from modelscope.metainfo import Metrics
from modelscope.models.cv.image_inpainting.modules.inception import InceptionV3
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .base import Metric
from .builder import METRICS, MetricKeys


def fid_calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(activations_pred, activations_target, eps=1e-6):
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
            - 2 * tr_covmean)


class FIDScore(torch.nn.Module):

    def __init__(self, dims=2048, eps=1e-6):
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)

        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())

    def get_value(self):
        activations_pred, activations_target = (self.activations_pred,
                                                self.activations_target)
        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()

        total_distance = calculate_frechet_distance(
            activations_pred, activations_target, eps=self.eps)

        self.reset()
        return total_distance

    def reset(self):
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, \
                'We should not have got here, because Inception always scales inputs to 299x299'
        activations = activations.squeeze(-1).squeeze(-1)
        return activations


class SSIM(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window',
                             self._create_window(window_size, self.channel))

    def forward(self, img1, img2):
        assert len(img1.shape) == 4

        channel = img1.size()[1]

        if channel == self.channel and self.window.data.type(
        ) == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel,
                          self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - (window_size // 2))**2 / float(2 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size,
                                 window_size).contiguous()

    def _ssim(self,
              img1,
              img2,
              window,
              window_size,
              channel,
              size_average=True):
        mu1 = F.conv2d(
            img1, window, padding=(window_size // 2), groups=channel)
        mu2 = F.conv2d(
            img2, window, padding=(window_size // 2), groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=(window_size // 2),
            groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=(window_size // 2),
            groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=(window_size // 2),
            groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.image_inpainting_metric)
class ImageInpaintingMetric(Metric):
    """The metric computation class for image inpainting classes.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.SSIM = SSIM(window_size=11, size_average=False).eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.FID = FIDScore().to(device)

    def add(self, outputs: Dict, inputs: Dict):
        pred = outputs['inpainted']
        target = inputs['image']
        self.preds.append(torch_nested_detach(pred))
        self.targets.append(torch_nested_detach(target))

    def evaluate(self):
        ssim_list = []
        for (pred, target) in zip(self.preds, self.targets):
            ssim_list.append(self.SSIM(pred, target))
            self.FID(pred, target)
        ssim_list = torch_nested_numpify(ssim_list)
        fid = self.FID.get_value()
        return {MetricKeys.SSIM: np.mean(ssim_list), MetricKeys.FID: fid}

    def merge(self, other: 'ImageInpaintingMetric'):
        self.preds.extend(other.preds)
        self.targets.extend(other.targets)

    def __getstate__(self):
        return self.preds, self.targets

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.targets = state
