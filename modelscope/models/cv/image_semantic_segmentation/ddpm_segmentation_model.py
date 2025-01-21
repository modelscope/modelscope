# The implementation here is modified based on ddpm-segmentation,
# originally Apache 2.0 License and publicly available at https://github.com/yandex-research/ddpm-segmentation
# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import torch
from ddpm_guided_diffusion.script_util import model_and_diffusion_defaults

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ddpm_seg.feature_extractors import (collect_features,
                                          create_feature_extractor)
from .ddpm_seg.pixel_classifier import (load_ensemble, pixel_classifier,
                                        predict_labels, save_predictions)

logger = get_logger()


@MODELS.register_module(Tasks.semantic_segmentation, module_name=Models.ddpm)
class DDPMSegmentationModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)
        self.cfg.model.mlp.category = self.cfg.pipeline.category

        self.cfg.model.ddpm.model_path = osp.join(model_dir,
                                                  ModelFile.TORCH_MODEL_FILE)
        default_ddpm_args = model_and_diffusion_defaults()
        default_ddpm_args.update(self.cfg.model.ddpm)
        self.feature_extractor = create_feature_extractor(**default_ddpm_args)

        self.cfg.model.mlp.model_path = osp.join(model_dir,
                                                 self.cfg.model.mlp.category)
        self.is_ensemble = kwargs.get('is_pipeline', True)
        if self.is_ensemble:
            logger.info('Load ensemble mlp ......')
            self.seg_model = load_ensemble(**self.cfg.model.mlp)
        else:
            logger.info('Load single mlp ......')
            self.seg_model = pixel_classifier(**self.cfg.model.mlp)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        re = self.inference(inputs)
        return re

    def inference(self, batch, seed=0):
        img = batch['input_img']
        img = img[None]
        w, h, c = self.cfg.model.mlp.dim

        if self.cfg.model.ddpm.share_noise:
            rnd_gen = torch.Generator().manual_seed(seed)
            noise = torch.randn(1, 3, w, h, generator=rnd_gen)
            noise = noise.to(img.device)
        else:
            noise = None

        features = self.feature_extractor(img, noise=noise)
        features = collect_features(self.cfg.model, features)

        x = features.view(c, -1).permute(1, 0)

        pred, _ = predict_labels(self.seg_model, x, size=(w, h))

        return {'pred': [pred.numpy()]}

    def postprocess(self, inputs: Dict[str, Any], **kwargs):
        category = self.cfg.model.mlp.category
        mask, out_img = save_predictions(inputs, category)
        return mask, out_img
