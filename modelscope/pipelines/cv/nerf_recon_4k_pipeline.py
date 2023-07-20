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
    Tasks.nerf_recon_4k, module_name=Pipelines.nerf_recon_4k)
class NeRFRecon4KPipeline(Pipeline):
    """ NeRF reconstruction acceleration pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> nerf_recon_acc = pipeline(Tasks.nerf_recon_acc,
                'damo/cv_nerf-3d-reconstruction-accelerate_damo')
    >>> nerf_recon_acc({
            'data_dir': '/data/lego', # data dir path (str)
            'render_dir': 'save_dir', # save dir path (str)
        })
    >>> #
    ```
    """

    def __init__(self,
                 model,
                 data_type='blender',
                 test_ray_chunk=8192,
                 test_tile=510,
                 stepsize=1.0,
                 factor=4,
                 load_sr=1,
                 device='gpu',
                 **kwargs):
        """
        use model to create a image sky change pipeline for image editing
        Args:
            model (str or Model): model_id on modelscope hub
            data_type (str): currently only support 'blender' and 'colmap'
            use_mask (bool): segment the object or not
            ckpt_path (str): the checkpoint ckpt_path
            save_mesh (bool): render mesh or not
            n_test_traj_steps (int): number of random sampled images for test view, only for colmap data.
            test_ray_chunk (int): ray chunk size for test, avoid GPU OOM
            device (str): only support gpu
        """
        model = Model.from_pretrained(
            model,
            device=device,
            model_prefetched=True,
            invoked_by=Invoke.PIPELINE,
            data_type=data_type,
            test_ray_chunk=test_ray_chunk,
            test_tile=test_tile,
            stepsize=stepsize,
            factor=factor,
            load_sr=load_sr) if is_model(model) else model

        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        self.data_type = data_type
        if self.data_type != 'blender' and self.data_type != 'llff':
            raise Exception('data type {} is not support currently'.format(
                self.data_type))
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        data_cfg = input['data_cfg']
        render_dir = input['render_dir']
        self.model.nerf_reconstruction(data_cfg, render_dir)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
