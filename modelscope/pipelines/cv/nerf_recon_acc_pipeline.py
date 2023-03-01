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
    Tasks.nerf_recon_acc, module_name=Pipelines.nerf_recon_acc)
class NeRFReconAccPipeline(Pipeline):
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
                 use_mask=True,
                 ckpt_path='',
                 save_mesh=False,
                 n_test_traj_steps=120,
                 test_ray_chunk=1024,
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
            use_mask=use_mask,
            ckpt_path=ckpt_path,
            save_mesh=save_mesh,
            n_test_traj_steps=n_test_traj_steps,
            test_ray_chunk=test_ray_chunk) if is_model(model) else model

        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        self.data_type = data_type
        if self.data_type != 'blender' and self.data_type != 'colmap':
            raise Exception('data type {} is not support currently'.format(
                self.data_type))
        self.use_mask = use_mask
        logger.info('load model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        data_dir = input['data_dir']
        render_dir = input['render_dir']
        self.model.nerf_reconstruction(data_dir, render_dir)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
