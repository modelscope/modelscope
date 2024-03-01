import argparse
import os.path as osp

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.dense_optical_flow_estimation.core.raft import RAFT
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.dense_optical_flow_estimation,
    module_name=Models.raft_dense_optical_flow_estimation)
class DenseOpticalFlowEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        # build model
        args = argparse.Namespace()
        args.model = model_dir
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        self.model = torch.nn.DataParallel(RAFT(args))

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.module
        self.model.to('cuda')
        self.model.eval()

    def forward(self, Inputs):
        image1 = Inputs['image1']
        image2 = Inputs['image2']

        flow_ups = self.model(image1, image2)
        flow_up = flow_ups[-1]

        return flow_up

    def postprocess(self, inputs):
        results = {OutputKeys.FLOWS: inputs}
        return results

    def inference(self, data):
        results = self.forward(data)
        return results
