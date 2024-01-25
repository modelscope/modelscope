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
    Tasks.self_supervised_depth_completion,
    module_name=Pipelines.self_supervised_depth_completion)
class SelfSupervisedDepthCompletionPipeline(Pipeline):
    """Self Supervise dDepth Completion Pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> model_id = 'Damo_XR_Lab/Self_Supervised_Depth_Completion'
    >>> data_dir = MsDataset.load(
            'KITTI_Depth_Dataset',
            namespace='Damo_XR_Lab',
            split='test',
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['test']
    >>> source_dir = os.path.join(data_dir, 'selected_data')
    >>> self_supervised_depth_completion = pipeline(Tasks.self_supervised_depth_completion,
                'Damo_XR_Lab/Self_Supervised_Depth_Completion')
    >>> result = self_supervised_depth_completion({
            'model_dir': model_id
            'source_dir': source_dir
        })
        cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT])
    >>> #
    ```
    """

    def __init__(self, model: str, **kwargs):

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """preprocess, not used at present"""
        return inputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """forward"""
        source_dir = inputs['source_dir']
        result = self.model.forward(source_dir)
        return {OutputKeys.OUTPUT: result}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """postprocess, not used at present"""
        return inputs
