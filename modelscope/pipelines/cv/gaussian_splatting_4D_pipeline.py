# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import is_model, is_official_hub_path
from modelscope.utils.constant import Invoke, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.gaussian_splatting_4D, module_name=Pipelines.gaussian_splatting_4D)
class GaussianSplatting4DPipeline(Pipeline):
    def __init__(self, model: str, **kwargs):

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        model_dir = input['model_dir']
        source_dir = input['source_dir']
        self.model.render(model_dir, source_dir)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        return inputs
