# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os.path as osp
from copy import deepcopy

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_local_feature_matching.src.loftr import (
    LoFTR, default_cfg)
from modelscope.models.cv.image_local_feature_matching.src.utils.plotting import \
    make_matching_figure
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.image_local_feature_matching,
    module_name=Models.loftr_image_local_feature_matching)
class LocalFeatureMatching(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        # build model
        # Initialize LoFTR
        _default_cfg = deepcopy(default_cfg)
        self.model = LoFTR(config=_default_cfg)

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def forward(self, Inputs):
        self.model(Inputs)
        result = {
            'kpts0': Inputs['mkpts0_f'],
            'kpts1': Inputs['mkpts1_f'],
            'conf': Inputs['mconf'],
        }
        Inputs.update(result)
        return Inputs

    def postprocess(self, Inputs):
        # Draw
        color = cm.jet(Inputs['conf'].cpu().numpy())
        img0, img1, mkpts0, mkpts1 = Inputs['image0'].squeeze().cpu().numpy(
        ), Inputs['image1'].squeeze().cpu().numpy(), Inputs['kpts0'].cpu(
        ).numpy(), Inputs['kpts1'].cpu().numpy()
        text = [
            'LoFTR',
            'Matches: {}'.format(len(Inputs['kpts0'])),
        ]
        img0, img1 = (img0 * 255).astype(np.uint8), (img1 * 255).astype(
            np.uint8)
        fig = make_matching_figure(
            img0, img1, mkpts0, mkpts1, color, text=text)
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='png', dpi=75)
        io_buf.seek(0)
        buf_data = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        io_buf.close()
        vis_img = cv2.imdecode(buf_data, 1)

        results = {OutputKeys.MATCHES: Inputs, OutputKeys.OUTPUT_IMG: vis_img}
        return results

    def inference(self, data):
        results = self.forward(data)

        return results
