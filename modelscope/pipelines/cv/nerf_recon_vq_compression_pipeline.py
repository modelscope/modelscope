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
    Tasks.nerf_recon_vq_compression,
    module_name=Pipelines.nerf_recon_vq_compression)
class NeRFReconVQCompressionPipeline(Pipeline):
    """ NeRF reconstruction VQ compression pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> nerf_recon_vq_compress = pipeline(Tasks.nerf_recon_vq_compression,
                'damo/cv_nerf-3d-reconstruction-vq-compression_damo')
    >>> nerf_recon_vq_compress({
            'data_dir': '/data/lego', # data dir path (str)
            'render_dir': 'save_dir', # save dir path (str)
            'ckpt_path': 'ckpt_path', # ckpt path (str)
        })
    >>> #
    ```
    """

    def __init__(self,
                 model,
                 dataset_name='blender',
                 data_dir='',
                 downsample=1,
                 ndc_ray=False,
                 ckpt_path='',
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
            dataset_name=dataset_name,
            data_dir=data_dir,
            downsample=downsample,
            ndc_ray=ndc_ray,
            ckpt_path=ckpt_path) if is_model(model) else model

        super().__init__(model=model, **kwargs)
        if not isinstance(self.model, Model):
            logger.error('model object is not initialized.')
            raise Exception('model object is not initialized.')
        logger.info('init model done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        test_mode = inputs['test_mode']
        if 'test' in test_mode or 'eval' in test_mode:
            inputs['test_mode'] = 'evaluation_test'
        elif 'path' in test_mode:
            inputs['test_mode'] = 'render_path'
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        render_dir = input['render_dir']
        test_mode = input['test_mode']
        N_vis = input.get('N_vis', 5)
        if test_mode == 'evaluation_test':
            self.model.evaluation(render_dir, N_vis)
        elif test_mode == 'render_path':
            self.model.render_path(render_dir, N_vis)
        else:
            raise Exception('test mode {} is not support'.format(test_mode))
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
