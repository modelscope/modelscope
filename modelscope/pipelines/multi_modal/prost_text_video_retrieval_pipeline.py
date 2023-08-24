# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_video_retrieval,
    module_name=Pipelines.prost_text_video_retrieval)
class ProSTTextVideoRetrievalPipeline(Pipeline):
    '''
    https://www.modelscope.cn/models/damo/multi_modal_clip_vtretrieval_prost/summary

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    text_video_retrieval= pipeline(
                Tasks.text_video_retrieval,
                model='damo/multi_modal_clip_vtretrieval_prost')
    video_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/multi_modal_test_video_9770.mp4'
    caption = 'a person is connecting something to system'
    _input = {'video': video_path, 'text': caption}
    result = text_video_retrieval(_input)
    '''

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a text_video_retrieval pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)
        self.model.eval()

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def _process_single(self, input: Input, *args, **kwargs) -> Dict[str, Any]:
        with device_placement(self.framework, self.device_name):
            out = self.forward(input)

        self._check_output(out)
        return out

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
