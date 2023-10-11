# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import shutil
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_reconstruction.utils import (
    align_for_lm, align_img, draw_line, enlarged_bbox, image_warp_grid1,
    load_lm3d, mesh_to_string, read_obj, resize_on_long_side, spread_flow,
    write_obj)
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device, device_placement
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_to_head, module_name=Pipelines.text_to_head)
class TextToHeadPipeline(Pipeline):

    def __init__(self, model: str, device: str, hair_tex=True):
        """The inference pipeline for text-to-head task.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            device ('str'): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope.models.cv.face_reconstruction.utils import write_obj
            >>> test_prompt = "a clown with red nose"
            >>> pipeline_textToHead = pipeline('text-to-head',
                model='damo/cv_HRN_text-to-head')
            >>> result = pipeline_textToHead(test_prompt)
            >>> mesh = result[OutputKeys.OUTPUT]['mesh']
            >>> texture_map = result[OutputKeys.OUTPUT_IMG]
            >>> mesh['texture_map'] = texture_map
            >>> write_obj('text_to_head.obj', mesh)
        """
        super().__init__(model=model, device=device)

        self.hair_tex = hair_tex

        head_recon_model_id = 'damo/cv_HRN_head-reconstruction'
        self.head_reconstructor = pipeline(
            Tasks.head_reconstruction,
            model=head_recon_model_id,
            model_revision='v0.1',
            hair_tex=hair_tex)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        result = {'text': input}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        image = self.model(input)
        image = np.array(image)

        results = self.head_reconstructor(image)
        results['image'] = image
        return results

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        render = kwargs.get('render', False)
        output_obj = inputs[OutputKeys.OUTPUT_OBJ]
        texture_map = inputs[OutputKeys.OUTPUT_IMG]
        results = inputs[OutputKeys.OUTPUT]

        if render:
            output_obj = io.BytesIO()
            mesh_str = mesh_to_string(results['mesh'])
            mesh_bytes = mesh_str.encode(encoding='utf-8')
            output_obj.write(mesh_bytes)

        result = {
            OutputKeys.OUTPUT_OBJ: output_obj,
            OutputKeys.OUTPUT_IMG: texture_map,
            OutputKeys.OUTPUT: None if render else results,
            'image': inputs['image']
        }
        return result
