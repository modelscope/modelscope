# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import argparse
import os

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from modelscope.msdatasets import MsDataset
from modelscope.hub.snapshot_download import snapshot_download

from .gaussian_renderer.arguments import ModelParams, PipelineParams
from .gaussian_renderer.arguments import get_combined_args
from .gaussian_renderer.render import render_datasets
from .gaussian_renderer.utils import safe_state

logger = get_logger()

__all__ = ['GaussianSplattingRecon']


@MODELS.register_module(
    Tasks.gaussian_splatting_recon,
    module_name=Models.gaussian_splatting_recon)
class GaussianSplattingRecon(TorchModel):

    def __init__(self, model_dir=None, **kwargs):
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')
        self.model_id = 'Damo_XR_Lab/cv_gaussian-splatting-recon_damo'
        self.device = torch.device('cuda')
        self.data_type = kwargs['data_type']
        self.data_dir = kwargs['data_dir']
        self.ckpt_path = kwargs['ckpt_path']

    def render_example_model(self):
        print('render example model from {}'.format(self.model_id))
        self.data_type = 'blender'
        self.pretrained_model = 'chair'

        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))

        ckpt_path = os.path.join(snapshot_path, 'pretrained_models', self.pretrained_model)
        print('ckpt_path: {}'.format(ckpt_path))
        data_dir = MsDataset.load('nerf_recon_dataset', namespace='damo',
                                  split='train').config_kwargs['split_config']['train']
        nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
        blender_scene = self.pretrained_model
        data_dir = os.path.join(nerf_synthetic_dataset, blender_scene)
        return ckpt_path, data_dir

    def render(self, render_dir: str):
        parser = argparse.ArgumentParser(description="inference parameters")
        model_params = ModelParams(parser, sentinel=True)
        pipe_params = PipelineParams(parser)

        if self.data_dir == '' or self.ckpt_path == '':
            self.ckpt_path, self.data_dir = self.render_example_model()
            self.render_dir = self.ckpt_path
        elif not os.path.exists(self.data_dir):
            print('source data {} not existed.'.format(self.data_dir))
            return
        elif not os.path.exists(self.ckpt_path):
            print('pretrained model {} not existed.'.format(self.ckpt_path))
            return
        else:
            self.render_dir = render_dir
            print('start to render from {}'.format(self.ckpt_path))

        parser.model_path = self.ckpt_path
        args = get_combined_args(parser)
        args.source_path = self.data_dir

        safe_state(False)

        print('pretrained_dir', self.ckpt_path)
        print('render_dir', self.render_dir)
        render_datasets(model_params.extract(args),
                        self.render_dir,
                        iteration=-1,
                        pipeline=pipe_params.extract(args),
                        skip_test=False,
                        skip_train=False)
        print('render image saved in {}'.format(self.render_dir))

    def get_render_pose(self, N_cameras=120):
        print('get_render_pose')
