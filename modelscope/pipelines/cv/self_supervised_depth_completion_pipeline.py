# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.self_supervised_depth_completion, module_name=Pipelines.self_supervised_depth_completion)
class SelfSupervisedDepthCompletionPipeline(Pipeline):
    """SelfSupervisedDepthCompletionPipeline Class"""
    def __init__(self, model: str, **kwargs):

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """preprocess, not used at present"""
        return inputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """forward"""
        model_dir = inputs['model_dir']
        source_dir = inputs['source_dir']
        self.model.run(model_dir, source_dir)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """postprocess, not used at present"""
        return inputs
