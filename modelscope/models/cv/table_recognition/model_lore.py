# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
from os.path import join
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .lineless_table_process import (get_affine_transform,
                                     get_affine_transform_upper_left,
                                     load_lore_model, process_detect_output,
                                     process_logic_output)
from .modules.lore_detector import LoreDetectModel
from .modules.lore_processor import LoreProcessModel

LOGGER = get_logger()


@MODELS.register_module(Tasks.lineless_table_recognition,
                        Models.lineless_table_recognition)
class LoreModel(TorchModel):
    '''
    The model first locates table cells in the input image by key point segmentation.
    Then the logical locations are predicted along with the spatial locations
    employing two cascading regressors.
    See details in paper "LORE: Logical Location Regression Network for Table Structure Recognition"
    (https://arxiv.org/abs/2303.03730).
    '''

    def __init__(self, model_dir: str, **kwargs):
        '''initialize the LORE model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        '''
        super(LoreModel, self).__init__()

        model_path = join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = torch.load(model_path, map_location='cpu')
        # init detect infer model
        self.detect_infer_model = LoreDetectModel()
        load_lore_model(self.detect_infer_model, checkpoint, 'model')
        # init process infer model
        self.process_infer_model = LoreProcessModel()
        load_lore_model(self.process_infer_model, checkpoint, 'processor')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            img (`torch.Tensor`): image tensor,
                shape of each tensor is [3, H, W].

        Return:
            dets (`torch.Tensor`): the locations of detected table cells,
                shape of each tensor is [N_cell, 8].
            dets (`torch.Tensor`): the logical coordinates of detected table cells,
                shape of each tensor is [N_cell, 4].
            meta (`Dict`): the meta info of original image.
        """
        outputs = self.detect_infer_model(input['img'])
        output = outputs[-1]
        meta = input['meta']
        slct_logi_feat, slct_dets_feat, slct_output_dets = process_detect_output(
            output, meta)
        _, slct_logi = self.process_infer_model(
            slct_logi_feat, dets=slct_dets_feat.to(torch.int64))
        return {
            'dets': slct_output_dets,
            'logi': slct_logi,
            'meta': input['meta']
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        slct_dets = inputs['dets']
        slct_logi = process_logic_output(inputs['logi'])
        result = {
            OutputKeys.POLYGONS: slct_dets,
            OutputKeys.BOXES: np.array(slct_logi[0].cpu().numpy())
        }
        return result
