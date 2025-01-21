# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):

    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


@MODELS.register_module(
    Tasks.image_classification, module_name=Models.easyrobust_model)
class EasyRobustModel(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        import easyrobust.models
        from timm.models import create_model
        from mmcls.datasets import ImageNet
        import modelscope.models.cv.image_classification.backbones
        from modelscope.utils.hub import read_config

        super().__init__(model_dir)

        self.config_type = 'ms_config'
        self.CLASSES = ImageNet.CLASSES
        cfg = read_config(model_dir)
        cfg.model.mm_model.pretrained = None
        self.cls_model = create_model(
            cfg.model.mm_model['type'], pretrained=False, num_classes=1000)

        model_pth_path = os.path.join(model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        normalize = NormalizeByChannelMeanStd(
            mean=cfg.model.mm_model['mean'], std=cfg.model.mm_model['std'])
        checkpoint = torch.load(model_pth_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if '0.mean' in state_dict.keys() and '0.std' in state_dict.keys():
            self.cls_model = nn.Sequential(normalize, self.cls_model)
            self.cls_model.load_state_dict(state_dict)
        else:
            self.cls_model.load_state_dict(state_dict)
            self.cls_model = nn.Sequential(normalize, self.cls_model)

        self.cfg = cfg
        self.ms_model_dir = model_dir

    def forward(self, inputs):
        logits = self.cls_model(inputs['img'])
        score = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return score
