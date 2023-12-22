# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
from typing import Any, Dict

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_reconstruction.utils import mesh_to_string
from modelscope.models.cv.human3d_animation import (projection, read_obj,
                                                    render, rotate_x, rotate_y,
                                                    translate)
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import is_model
from modelscope.utils.constant import Invoke, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.human3d_render, module_name=Pipelines.human3d_render)
class Human3DRenderPipeline(Pipeline):
    """ Human3D library render pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> human3d = pipeline(Tasks.human3d_render,
                'damo/cv_3d-human-synthesis-library')
    >>> human3d({
            'data_dir': '/data/human3d-syn-library', # data dir path (str)
            'case_id': '3f2a7538253e42a8', # case id (str)
        })
    >>> #
    ```
    """

    def __init__(self, model: str, device='gpu', **kwargs):
        """
        use model to create a image sky change pipeline for image editing
        Args:
            model (str or Model): model_id on modelscope hub
            device (str): only support gpu
        """
        super().__init__(model=model, **kwargs)
        self.model_dir = model

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def load_3d_model(self, mesh_path):
        mesh = read_obj(mesh_path)
        tex_path = mesh_path.replace('.obj', '.png')
        if not os.path.exists(tex_path):
            tex = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            tex = cv2.imread(tex_path)
        mesh['texture_map'] = tex.copy()
        return mesh, tex

    def format_nvdiffrast_format(self, mesh, tex):
        vert = mesh['vertices']
        cent = (vert.max(axis=0) + vert.min(axis=0)) / 2
        vert -= cent
        tri = mesh['faces']
        tri = tri - 1 if tri.min() == 1 else tri
        vert_uv = mesh['uvs']
        tri_uv = mesh['faces_uv']
        tri_uv = tri_uv - 1 if tri_uv.min() == 1 else tri_uv
        vtx_pos = torch.from_numpy(vert.astype(np.float32)).cuda()
        pos_idx = torch.from_numpy(tri.astype(np.int32)).cuda()
        vtx_uv = torch.from_numpy(vert_uv.astype(np.float32)).cuda()
        uv_idx = torch.from_numpy(tri_uv.astype(np.int32)).cuda()
        tex = tex[::-1, :, ::-1]
        tex = torch.from_numpy(tex.astype(np.float32) / 255.0).cuda()
        return vtx_pos, pos_idx, vtx_uv, uv_idx, tex

    def render_scene(self, mesh_path, resolution=512):
        if not os.path.exists(mesh_path):
            logger.info('can not found %s, use default one' % mesh_path)
            mesh_path = os.path.join(self.model_dir, '3D-assets',
                                     '3f2a7538253e42a8', 'body.obj')

        mesh, texture = self.load_3d_model(mesh_path)
        vtx_pos, pos_idx, vtx_uv, uv_idx, tex = self.format_nvdiffrast_format(
            mesh, texture)

        glctx = dr.RasterizeCudaContext()
        ang = 0.0
        frame_length = 80
        step = 2 * np.pi / frame_length
        frames_color = []
        frames_normals = []
        for i in tqdm.tqdm(range(frame_length)):
            proj = projection(x=0.4, n=1.0, f=200.0)
            a_rot = np.matmul(rotate_x(0.0), rotate_y(ang))
            a_mv = np.matmul(translate(0, 0, -2.7), a_rot)
            r_mvp = np.matmul(proj, a_mv).astype(np.float32)
            pred_img, pred_mask, normal = render(
                glctx,
                r_mvp,
                vtx_pos,
                pos_idx,
                vtx_uv,
                uv_idx,
                tex,
                resolution=resolution,
                enable_mip=False,
                max_mip_level=9)
            color = np.clip(
                np.rint(pred_img[0].detach().cpu().numpy() * 255.0), 0,
                255).astype(np.uint8)[::-1, :, :]
            normals = np.clip(
                np.rint(normal[0].detach().cpu().numpy() * 255.0), 0,
                255).astype(np.uint8)[::-1, :, :]
            frames_color.append(color)
            frames_normals.append(normals)
            ang = ang + step

        logger.info('render case %s done'
                    % os.path.basename(os.path.dirname(mesh_path)))

        return mesh, frames_color, frames_normals

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = input['dataset_id']
        case_id = input['case_id']
        if 'resolution' in input:
            resolution = input['resolution']
        else:
            resolution = 512
        if case_id.endswith('.obj'):
            mesh_path = case_id
        else:
            dataset_name = dataset_id.split('/')[-1]
            user_name = dataset_id.split('/')[0]
            data_dir = MsDataset.load(
                dataset_name, namespace=user_name,
                subset_name=case_id).config_kwargs['split_config']['test']
            case_dir = os.path.join(data_dir, case_id)
            mesh_path = os.path.join(case_dir, 'body.obj')

        mesh, colors, normals = self.render_scene(mesh_path, resolution)

        results = {
            'mesh': mesh,
            'frames_color': colors,
            'frames_normal': normals,
        }
        return {OutputKeys.OUTPUT_OBJ: None, OutputKeys.OUTPUT: results}

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        render = kwargs.get('render', False)
        output_obj = inputs[OutputKeys.OUTPUT_OBJ]
        results = inputs[OutputKeys.OUTPUT]

        if render:
            output_obj = io.BytesIO()
            mesh_str = mesh_to_string(results['mesh'])
            mesh_bytes = mesh_str.encode(encoding='utf-8')
            output_obj.write(mesh_bytes)

        result = {
            OutputKeys.OUTPUT_OBJ: output_obj,
            OutputKeys.OUTPUT: None if render else results,
        }
        return result
