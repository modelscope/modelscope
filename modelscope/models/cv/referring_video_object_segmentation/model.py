# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .utils import (MTTR, A2DSentencesPostProcess, ReferYoutubeVOSPostProcess,
                    nested_tensor_from_videos_list)

logger = get_logger()


@MODELS.register_module(
    Tasks.referring_video_object_segmentation,
    module_name=Models.referring_video_object_segmentation)
class ReferringVideoObjectSegmentation(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)
        self.model = MTTR(**self.cfg.model)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in params_dict.keys():
            params_dict = params_dict['model_state_dict']
        self.model.load_state_dict(params_dict, strict=True)

        dataset_name = self.cfg.pipeline.dataset_name
        if dataset_name == 'a2d_sentences' or dataset_name == 'jhmdb_sentences':
            self.postprocessor = A2DSentencesPostProcess()
        elif dataset_name == 'ref_youtube_vos':
            self.postprocessor = ReferYoutubeVOSPostProcess()
        else:
            assert False, f'postprocessing for dataset: {dataset_name} is not supported'

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return inputs

    def inference(self, **kwargs):
        window = kwargs['window']
        text_query = kwargs['text_query']
        video_metadata = kwargs['metadata']

        window = nested_tensor_from_videos_list([window])
        valid_indices = torch.arange(len(window.tensors))
        if self._device_name == 'gpu':
            valid_indices = valid_indices.cuda()
        outputs = self.model(window, valid_indices, [text_query])
        window_masks = self.postprocessor(
            outputs, [video_metadata],
            window.tensors.shape[-2:])[0]['pred_masks']
        return window_masks

    def postprocess(self, inputs: Dict[str, Any], **kwargs):
        return inputs
