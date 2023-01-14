# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from .config.default import get_cfg_defaults
from .loftr_quadtree.loftr import LoFTR
from .utils.misc import lower_config


@MODELS.register_module(
    Tasks.image_matching, module_name=Models.quadtree_attention_image_matching)
class QuadTreeAttentionForImageMatching(TorchModel):
    '''
    Image matching with quadtree attention. This model is trained on outdoor images.
    For more details, please refer to https://arxiv.org/abs/2201.02767
    '''

    def __init__(self, model_dir: str, model_type='outdoor', **kwargs):
        '''
        Args:
            model_dir: model directory
            model_type: model type, 'outdoor' or 'indoor'. Only support outdoor model for modelscope.
        '''
        assert model_type == 'outdoor', 'Only support outdoor model for modelscope'
        # Note: for indoor model, max_image_size should be 640 because scannet training image size is 640,
        # and currently, this model is overfited on scannet. For outdoor model, larger image size will be better

        super().__init__(model_dir, **kwargs)
        config = get_cfg_defaults()
        _config = lower_config(config)

        matcher = LoFTR(config=_config['loftr'])
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        state_dict = torch.load(
            str(model_path), map_location='cpu')['state_dict']

        matcher.load_state_dict(state_dict, strict=True)
        self.matcher = matcher

        self.matcher.eval()
        self.matcher.to('cuda')

    def forward(self, Inputs):
        '''
        Args:
            Inputs: a dict with keys 'image0', 'image1' and 'preprocess_info'.
                'image0' and 'image1' are torch tensor with shape [1, 1, H1, W1]
                and [1, 1, H2, W2]. 'preprocess_info' contains the information of
                resizing, which will be used for postprocessing.
        '''
        self.matcher(Inputs)
        return {
            'kpts0': Inputs['mkpts0_f'],
            'kpts1': Inputs['mkpts1_f'],
            'conf': Inputs['mconf'],
            'preprocess_info': Inputs['preprocess_info']
        }

    def postprocess(self, Inputs):
        matching_result = Inputs

        results = {OutputKeys.MATCHES: matching_result}
        return results

    def inference(self, data):
        results = self.forward(data)

        return results
