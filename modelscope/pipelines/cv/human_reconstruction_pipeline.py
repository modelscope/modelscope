# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict

import numpy as np
import torch
import trimesh

from modelscope.metainfo import Pipelines
from modelscope.models.cv.human_reconstruction.utils import (
    keep_largest, reconstruction, save_obj_mesh, save_obj_mesh_with_color,
    to_tensor)
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.human_reconstruction, module_name=Pipelines.human_reconstruction)
class HumanReconstructionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """The inference pipeline for human reconstruction task.
        Human Reconstruction Pipeline. Given one image generate a human mesh.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> test_input = 'human_reconstruction.jpg' # input image path
            >>> pipeline_humanRecon = pipeline('human-reconstruction',
                model='damo/cv_hrnet_image-human-reconstruction')
            >>> result = pipeline_humanRecon(test_input)
            >>> output =  result[OutputKeys.OUTPUT]
        """
        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img_crop = self.model.crop_img(input)
        img, mask = self.model.get_mask(img_crop)
        normal_f, normal_b = self.model.generation_normal(img, mask)
        image = to_tensor(img_crop) * 2 - 1
        normal_b = to_tensor(normal_b) * 2 - 1
        normal_f = to_tensor(normal_f) * 2 - 1
        mask = to_tensor(mask)
        result = {
            'img': image,
            'mask': mask,
            'normal_F': normal_f,
            'normal_B': normal_b
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        image = input['img']
        mask = input['mask']
        normF = input['normal_F']
        normB = input['normal_B']
        normF[1, ...] = -normF[1, ...]
        normB[0, ...] = -normB[0, ...]
        img = image * mask
        normal_b = normB * mask
        normal_f = normF * mask
        img = torch.cat([img, normal_f, normal_b], dim=0).float()
        image_tensor = img.unsqueeze(0).to(self.model.device)
        calib_tensor = self.model.calib
        net = self.model.meshmodel
        net.extract_features(image_tensor)
        verts, faces = reconstruction(net, calib_tensor, self.model.coords,
                                      self.model.mat)
        pre_mesh = trimesh.Trimesh(
            verts, faces, process=False, maintain_order=True)
        final_mesh = keep_largest(pre_mesh)
        verts = final_mesh.vertices
        faces = final_mesh.faces
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(
            self.model.device).float()
        color = torch.zeros(verts.shape)
        interval = 20000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            pred_color = net.query_rgb(verts_tensor[:, :, left:right],
                                       calib_tensor)
            rgb = pred_color[0].detach().cpu() * 0.5 + 0.5
            color[left:right] = rgb.T
        vert_min = np.min(verts[:, 1])
        verts[:, 1] = verts[:, 1] - vert_min
        save_obj_mesh('human_reconstruction.obj', verts, faces)
        save_obj_mesh_with_color('human_color.obj', verts, faces,
                                 color.numpy())
        results = {'vertices': verts, 'faces': faces, 'colors': color.numpy()}
        return {OutputKeys.OUTPUT: results}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
