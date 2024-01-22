# The implementation is made publicly available under the
# Apache 2.0 license at https://github.com/cvg/LightGlue

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
from .config.default import lightglue_default_conf
from .lightglue import ALIKED, DISK, SIFT, LightGlue, SuperPoint
from .lightglue.utils import numpy_image_to_torch, rbd


@MODELS.register_module(
    Tasks.image_matching, module_name=Models.lightglue_image_matching)
class LightGlueImageMatching(TorchModel):
    '''
    LightGlue is an simple but effective enhancement of the state-of-the-art image matching method, SuperGlue.
    For more details, please refer to https://arxiv.org/abs/2306.13643
    '''

    def __init__(self, model_dir: str, max_num_keypoints=2048, **kwargs):

        super().__init__(model_dir, **kwargs)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

        features = lightglue_default_conf.get('features', 'superpoint')

        if features == 'disk':
            self.extractor = DISK(
                max_num_keypoints=max_num_keypoints).eval().to(self.device)
        elif features == 'aliked':
            self.extractor = ALIKED(
                max_num_keypoints=max_num_keypoints).eval().to(self.device)
        elif features == 'sift':
            self.extractor = SIFT(
                max_num_keypoints=max_num_keypoints).eval().to(self.device)
        else:
            self.extractor = SuperPoint(
                model_dir=model_dir,
                max_num_keypoints=max_num_keypoints).eval().to(self.device)

        self.matcher = LightGlue(
            model_dir=model_dir,
            default_conf=lightglue_default_conf).eval().to(self.device)

    def forward(self, inputs):
        '''
        Args:
            inputs: a dict with keys 'image0', 'image1'
        '''

        feats0 = self.extractor.extract(
            numpy_image_to_torch(inputs['image0']).to(self.device))
        feats1 = self.extractor.extract(
            numpy_image_to_torch(inputs['image1']).to(self.device))
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        return [feats0, feats1, matches01]

    def postprocess(self, inputs):
        '''
        Args:
            inputs: a list of feats0, feats1, matches01
        '''
        matching_result = inputs
        feats0, feats1, matches01 = [rbd(x) for x in matching_result
                                     ]  # remove batch dimension

        kpts0, kpts1, matches = feats0['keypoints'], feats1[
            'keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # match confidence
        confidence = matches01['scores']

        matches_result = {
            'kpts0': m_kpts0,
            'kpts1': m_kpts1,
            'confidence': confidence
        }

        results = {OutputKeys.MATCHES: matches_result}
        return results

    def inference(self, data):
        results = self.forward(data)

        return results
