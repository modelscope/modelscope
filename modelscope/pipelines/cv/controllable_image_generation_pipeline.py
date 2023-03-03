# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import math
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.cv.controllable_image_generation import ControlNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import is_model, is_official_hub_path
from modelscope.preprocessors.cv.controllable_image_generation import \
    ControllableImageGenerationPreprocessor
from modelscope.utils.constant import Frameworks, Invoke, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ControllableImageGenerationPipeline']


@PIPELINES.register_module(
    Tasks.controllable_image_generation,
    module_name=Pipelines.controllable_image_generation)
class ControllableImageGenerationPipeline(Pipeline):
    """  controllable image generation Pipeline.

    Examples:

    >>> import cv2
    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> input_location = 'data/test/images/image_inpainting/image_inpainting_mask_1.png'
    >>> prompt = 'hot air balloon'
    >>> output_image_path = './result.png'
    >>> input = {
    >>>     'image': input_location,
    >>>     'prompt': prompt
    >>> }
    >>> controllable_image_generation = pipeline(
    >>>     Tasks.controllable_image_generation,
    >>>     model='damo/cv_controlnet_scribble-to-image_base',
    >>>     control_type='scribble')
    >>> output = controllable_image_generation(input)[OutputKeys.OUTPUT_IMG]
    >>> cv2.imwrite(output_image_path, output)
    >>> print('pipeline: the output image path is {}'.format(output_image_path))
    """

    def initiate_single_model(self, model):
        if isinstance(model, str):
            logger.info(f'initiate model from {model}')
        if isinstance(model, str) and is_official_hub_path(model):
            logger.info(f'initiate model from location {model}.')
            # expecting model has been prefetched to local cache beforehand
            return Model.from_pretrained(
                model,
                device=self.device_name,
                model_prefetched=True,
                invoked_by=Invoke.PIPELINE,
                control_type=self.init_control_type) if is_model(
                    model) else model
        else:
            return model

    def __init__(self,
                 model: Union[ControlNet, str],
                 preprocessor=None,
                 device='cuda',
                 auto_collate=False,
                 **kwargs):
        self.init_control_type = kwargs.get('control_type', 'hed')
        if device == 'gpu':
            device = 'cuda'
        self.device_name = device
        cnet = self.initiate_single_model(model)
        model_path = os.path.join(cnet.get_model_dir(), './ckpt/annotator/')
        CIGPreprocessor = ControllableImageGenerationPreprocessor(
            control_type=self.init_control_type,
            model_path=model_path,
            device=device)
        super().__init__(
            model=cnet,
            preprocessor=CIGPreprocessor,
            device=device,
            auto_collate=auto_collate,
            **kwargs)
        self.device = device

        logger.info('load ControlNet done')

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method

        Returns:
            Dict[str, str]:  preprocess_params = {'image_resolution': self.model.get_resolution()}
            Dict[str, str]:  forward_params = pipeline_parameters
            Dict[str, str]:  postprocess_params = {}
        """
        pipeline_parameters['image_resolution'] = self.model.get_resolution()
        pipeline_parameters['modelsetting'] = self.model.get_config()
        pipeline_parameters['model_dir'] = self.model.get_model_dir()
        pipeline_parameters['control_type'] = self.init_control_type
        pipeline_parameters['device'] = self.device

        return pipeline_parameters, {}, {}

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:

        result = self.model(inputs)

        return result

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = np.array(inputs['result'][0])
        is_cat_img = inputs['is_cat_img']

        if is_cat_img:
            detected_map = inputs['detected_map']
            cat = np.concatenate((detected_map, result), axis=1)
            return {OutputKeys.OUTPUT_IMG: cat[:, :, ::-1]}
        else:
            return {OutputKeys.OUTPUT_IMG: result[:, :, ::-1]}
