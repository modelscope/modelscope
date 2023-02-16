# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.models.cv.nerf_recon_acc import NeRFReconPreprocessor
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.nerf_recon_acc, module_name=Pipelines.nerf_recon_acc)
class NeRFReconAccPipeline(Pipeline):
    """ NeRF reconstruction acceleration pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> nerf_recon_acc = pipeline(Tasks.nerf_recon_acc,
                'damo/cv_nerf-3d-reconstruction-accelerate_damo')
    >>> nerf_recon_acc({
            'video_input_path': 'input.mp4', # input video path (str)
            'data_dir': '/data/lego', # data dir path (str)
        })
       {
        "output": 'render.mp4' # saved path of render video (str)
        }
    >>> #
    ```
    """

    def __init__(self, model, data_type='colmap', use_mask=True, **kwargs):
        """
        use `model` to create a image sky change pipeline for image editing
        Args:
            model (`str` or `Model`): model_id on modelscope hub
            preprocessor(`Preprocessor`, *optional*,  defaults to None): `NeRFReconPreprocessor`.
        """
        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        self.data_type = data_type
        if self.data_type != 'blender' and self.data_type != 'colmap':
            raise Exception('data type {} is not support currently'.format(
                self.data_type))
        self.use_mask = use_mask

        self.preprocessor = NeRFReconPreprocessor(
            data_type=self.data_type, use_mask=self.use_mask)
        logger.info('load model done')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        data_dir = input['data_dir']
        result = self.model.nerf_reconstruction(data_dir)
        return {OutputKeys.OUTPUT_VIDEO: result}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
