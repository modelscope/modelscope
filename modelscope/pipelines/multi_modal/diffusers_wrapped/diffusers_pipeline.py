# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Generator, List, Union

from modelscope.pipelines.base import Input, Pipeline
from modelscope.utils.constant import Hubs
from modelscope.utils.device import create_device
from modelscope.utils.hub import snapshot_download


class DiffusersPipeline(Pipeline):

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """
        use `model` to create a diffusers pipeline
        Args:
            model: model id on modelscope hub or local dir.
            device: str = 'gpu'
        """

        self.device_name = device
        self.cfg = None
        self.preprocessor = None
        self.framework = None
        self.device = create_device(self.device_name)
        self.hubs = kwargs.get('hubs', Hubs.modelscope)

        # make sure we download the model from modelscope hub
        model_folder = model
        if not os.path.isdir(model_folder):
            if self.hubs != Hubs.modelscope:
                raise NotImplementedError(
                    'Only support model retrieval from ModelScope hub for now.'
                )
            model_folder = snapshot_download(model)

        self.model = model_folder
        self.models = [self.model]
        self.has_multiple_models = len(self.models) > 1

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {'num_inference_steps': 30, 'guidance_scale': 7.5}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        forward_params = {}
        forward_params['num_inference_steps'] = pipeline_parameters.pop('num_inference_steps', 30)
        forward_params['guidance_scale'] = pipeline_parameters.pop('guidance_scale', 7.5)
        return {}, forward_params, pipeline_parameters

    def __call__(self,
                 input: Union[Input, List[Input]],
                #  num_inference_steps: int = 30,
                #  guidance_scale: float = 7.5,
                 *args,
                 **kwargs) -> Union[Dict[str, Any], Generator]:
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs)
        self._check_input(input)
        out = self.preprocess(input, **preprocess_params)
        out = self.forward(out, **forward_params)
        out = self.postprocess(out, **postprocess_params)
        self._check_output(out)
        return out
