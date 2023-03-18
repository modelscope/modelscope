# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from copy import deepcopy
from typing import Dict, Union

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ddcolor import DDColor
from .loss import L1Loss

logger = get_logger()

__all__ = ['DDColorForImageColorization']


def tensor_lab2rgb(labs, illuminant='D65', observer='2'):
    """
    Args:
        lab    : (B, C, H, W)
    Returns:
        tuple   : (C, H, W)
    """
    illuminants = \
        {'A': {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         'D55': {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         'D65': {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         'D75': {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         'E': {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}
    rgb_from_xyz = np.array([[3.240481340, -0.96925495, 0.055646640],
                             [-1.53715152, 1.875990000, -0.20404134],
                             [-0.49853633, 0.041555930, 1.057311070]])
    B, C, H, W = labs.shape
    arrs = labs.permute(
        (0, 2, 3, 1)).contiguous()  # (B, 3, H, W) -> (B, H, W, 3)
    L, a, b = arrs[:, :, :, 0:1], arrs[:, :, :, 1:2], arrs[:, :, :, 2:]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    invalid = z.data < 0
    z[invalid] = 0
    xyz = torch.cat([x, y, z], dim=3)
    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.) / 7.787
    xyz_ref_white = illuminants[illuminant][observer]
    for i in range(C):
        mask_xyz[:, :, :, i] = mask_xyz[:, :, :, i] * xyz_ref_white[i]

    rgb_trans = torch.mm(
        mask_xyz.view(-1, 3),
        torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(B, H, W, C)
    rgb = rgb_trans.permute((0, 3, 1, 2)).contiguous()
    mask = rgb.data > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92
    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb


@MODELS.register_module(Tasks.image_colorization, module_name=Models.ddcolor)
class DDColorForImageColorization(TorchModel):
    """DDColor model for Image Colorization:
        Colorize an image using unet with dual decoders,
        while the image decoder restores the spatial resolution,
        and the color decoder learn adaptive color queries.
    """

    def __init__(self,
                 model_dir,
                 encoder_name='convnext-l',
                 input_size=(512, 512),
                 num_queries=100,
                 *args,
                 **kwargs):
        """initialize the image colorization model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            encoder_name (str): the encoder name.
            input_size (tuple): size of the model input image.
            num_queries (int): number of decoder queries
        """
        super().__init__(model_dir, *args, **kwargs)

        self.model = DDColor(encoder_name, input_size, num_queries)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = self._load_pretrained(self.model, model_path)
        self.loss = L1Loss(loss_weight=0.1)

    def _load_pretrained(self,
                         net,
                         load_path,
                         strict=True,
                         param_key='params'):
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info(
                    f'Loading: {param_key} does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].'
        )
        # remove unnecessary 'module.' or 'model.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
            elif k.startswith('model.'):
                load_net[k[6:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
        logger.info('load model done.')
        return net

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        preds = self.model(input)
        return {'loss': self.loss(preds, target)}

    def _evaluate_postprocess(self, input: Tensor, target: Tensor,
                              img_l: Tensor,
                              gt_rgb: Tensor) -> Dict[str, list]:
        preds = self.model(input)  # (n, 2, h, w)

        preds_lab = torch.cat((img_l, preds), 1)  # (n, 3, h, w)
        preds_rgb = tensor_lab2rgb(preds_lab)

        # preds = list(torch.split(preds_rgb, 1, 0))
        # targets = list(torch.split(gt_rgb, 1, 0))
        preds = preds_rgb
        targets = gt_rgb

        return {'preds': preds, 'targets': targets}

    def forward(self, input: Dict[str,
                                  Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result of the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        if self.training:
            return self._train_forward(**input)
        elif 'target' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self.model(**input)
