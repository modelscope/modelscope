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
    Tasks.surface_recon_common, module_name=Pipelines.surface_recon_common)
class SurfaceReconCommonPipeline(Pipeline):
    """ Surface reconstruction common pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> surface_recon_common = pipeline(Tasks.surface_recon_common,
                'damo/cv_surface-reconstruction-common')
    >>> surface_recon_common({
            'data_dir': '/data/lego', # data dir path (str)
            'save_dir': './output', # save dir path (str)
        })
    >>> #
    ```
    """

    def __init__(self, model, device='gpu', **kwargs):
        """
        use model to create a image sky change pipeline for image editing
        Args:
            model (str or Model): model_id on modelscope hub
            device (str): only support gpu
        """
        model = Model.from_pretrained(
            model,
            device=device,
            model_prefetched=True,
            invoked_by=Invoke.PIPELINE) if is_model(model) else model

        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        data_dir = input['data_dir']
        save_dir = input['save_dir']
        if 'color' in input:
            color = input['color']
        else:
            color = False
        if 'n_directions' in input:
            n_directions = input['n_directions']
        else:
            n_directions = 8
        self.model.surface_reconstruction(data_dir, save_dir, color,
                                          n_directions)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
