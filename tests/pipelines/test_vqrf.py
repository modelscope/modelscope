# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level

class NeRFReconVQCompressionBlender(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'DAMOXR/cv_nerf_3d-reconstruction_vector-quantize-compression'
        pretrained_model = 'ficus_demo.pt'
        data_dir = MsDataset.load('nerf_recon_dataset', namespace='damo',
                    split='train').config_kwargs['split_config']['train']
        nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
        self.blender_scene = 'ficus'
        data_dir = os.path.join(nerf_synthetic_dataset, self.blender_scene)

        self.pipeline = pipeline(Tasks.nerf_recon_vq_compression, 
                                 model=self.model_id,
                                 dataset_name='blender',
                                 data_dir=data_dir,
                                 downsample=1,
                                 ndc_ray=False,
                                 ckpt_path=pretrained_model
                                 )
    
    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_evalutaion(self):
        render_dir = f'./exp/{self.blender_scene}'
        self.pipeline(dict(test_mode='evaluation_test', render_dir=render_dir, N_vis=5))
    
    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_render_path(self):
        render_dir = f'./exp/{self.blender_scene}'
        self.pipeline(dict(test_mode='render_path', render_dir=render_dir, N_vis=30))

class NeRFReconVQCompressionLLFF(unittest.TestCase):
    def setUp(self) -> None:
        self.model_id = 'DAMOXR/cv_nerf_3d-reconstruction_vector-quantize-compression'
        pretrained_model = 'fern_demo.pt'
        data_dir = MsDataset.load(
            'DAMOXR/nerf_llff_data',
            subset_name='default',
            split='test',
        ).config_kwargs['split_config']['test']
        nerf_llff = os.path.join(data_dir, 'nerf_llff_data')
        self.llff_scene = 'fern'
        data_dir = os.path.join(nerf_llff, self.llff_scene)

        self.pipeline = pipeline(Tasks.nerf_recon_vq_compression, 
                                 model=self.model_id,
                                 dataset_name='llff',
                                 data_dir=data_dir,
                                 downsample=4,
                                 ndc_ray=True,
                                 ckpt_path=pretrained_model
                                 )
    
    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_evalutaion(self):
        render_dir = f'./exp/{self.llff_scene}'
        self.pipeline(dict(test_mode='evaluation_test', render_dir=render_dir, N_vis=5))
    
    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_render_path(self):
        render_dir = f'./exp/{self.llff_scene}'
        self.pipeline(dict(test_mode='render_path', render_dir=render_dir, N_vis=10))



if __name__ == '__main__':
    unittest.main()
