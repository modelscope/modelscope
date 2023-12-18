# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
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
    Tasks.gaussian_splatting_recon,
    module_name=Pipelines.gaussian_splatting_recon)
class GaussianSplattingReconPipeline(Pipeline):
    """ Gaussian Splatting reconstruction pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> gaussian_splatting_recon = pipeline(Tasks.gaussian_splatting_recon,
                'damo/cv_gaussian-splatting-recon_damo')
    >>> gaussian_splatting_recon({
            'model_dir': '', # pretrained model dir (str)
        })
    >>> #
    ```
    """

    def __init__(self,
                 model,
                 device='gpu',
                 data_type='colmap',
                 data_dir='',
                 ckpt_path='',
                 **kwargs):
        """
        use model to create a image sky change pipeline for image editing
        Args:
            model (str or Model): model_id on modelscope hub
            data_type (str): currently only support 'blender' and 'colmap'
            ckpt_path (str): the checkpoint ckpt_path
            device (str): only support gpu
        """
        model = Model.from_pretrained(
            model,
            device=device,
            model_prefetched=True,
            data_type=data_type,
            data_dir=data_dir,
            ckpt_path=ckpt_path) if is_model(model) else model

        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        logger.info('init model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs['test_mode'] = "evaluation"
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        test_mode = input['test_mode']
        render_dir = input['render_dir']

        if test_mode == 'evaluation':
            self.model.render(render_dir)
        else:
            raise Exception('test mode {} is not support'.format(test_mode))
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
