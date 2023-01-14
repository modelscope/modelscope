# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.pointcloud_sceneflow_estimation,
    module_name=Pipelines.pointcloud_sceneflow_estimation)
class PointCloudSceneFlowEstimationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('pointcloud scenflow estimation model, pipeline init')

    def check_input_pcd(self, pcd):
        assert pcd.ndim == 2, 'pcd ndim must equal to 2'
        assert pcd.shape[1] == 3, 'pcd.shape[1] must equal to 3'

    def preprocess(self, input: Input) -> Dict[str, Any]:
        assert isinstance(input, tuple), 'only support tuple input'
        assert isinstance(input[0], str) and isinstance(
            input[1], str), 'only support tuple input with str type'

        pcd1_file, pcd2_file = input
        logger.info(f'input pcd file:{pcd1_file},  \n  {pcd2_file}')
        pcd1 = np.load(pcd1_file)
        pcd2 = np.load(pcd2_file)
        self.check_input_pcd(pcd1)
        self.check_input_pcd(pcd2)
        pcd1_torch = torch.from_numpy(pcd1).float().unsqueeze(0).cuda()
        pcd2_torch = torch.from_numpy(pcd2).float().unsqueeze(0).cuda()

        data = {
            'pcd1': pcd1_torch,
            'pcd2': pcd2_torch,
            'pcd1_ori': pcd1,
            'pcd2_ori': pcd2
        }

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        output = self.model.inference(input)
        results['output'] = output
        results['pcd1_ori'] = input['pcd1_ori']
        results['pcd2_ori'] = input['pcd2_ori']
        return results

    def save_ply_data(self, pcd1, pcd2):
        vertexs = np.concatenate([pcd1, pcd2], axis=0)
        color1 = np.array([[255, 0, 0]], dtype=np.uint8)
        color2 = np.array([[0, 255, 0]], dtype=np.uint8)
        color1 = np.tile(color1, (pcd1.shape[0], 1))
        color2 = np.tile(color2, (pcd2.shape[0], 1))
        vertex_colors = np.concatenate([color1, color2], axis=0)

        vertexs = np.array([tuple(v) for v in vertexs],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_colors = np.array([tuple(v) for v in vertex_colors],
                                 dtype=[('red', 'u1'), ('green', 'u1'),
                                        ('blue', 'u1')])

        vertex_all = np.empty(
            len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
        for prop in vertexs.dtype.names:
            vertex_all[prop] = vertexs[prop]
        for prop in vertex_colors.dtype.names:
            vertex_all[prop] = vertex_colors[prop]

        el = PlyElement.describe(vertex_all, 'vertex')
        ply_data = PlyData([el])
        # .write(save_name)
        return ply_data

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        flow = results[OutputKeys.OUTPUT]

        pcd1 = inputs['pcd1_ori']
        pcd2 = inputs['pcd2_ori']
        if isinstance(pcd1, torch.Tensor):
            pcd1 = pcd1.cpu().numpy()
        if isinstance(pcd2, torch.Tensor):
            pcd2 = pcd2.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        outputs = {
            OutputKeys.OUTPUT: flow,
            OutputKeys.PCD12: self.save_ply_data(pcd1, pcd2),
            OutputKeys.PCD12_ALIGN: self.save_ply_data(pcd1 + flow, pcd2),
        }
        return outputs
