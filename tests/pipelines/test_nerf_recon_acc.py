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


class NeRFReconAccTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'
        data_dir = MsDataset.load(
            'nerf_recon_dataset',
            namespace='damo',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['train']
        nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
        blender_scene = 'lego'
        self.data_dir = os.path.join(nerf_synthetic_dataset, blender_scene)
        self.render_dir = 'exp'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_by_direct_model_download(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        nerf_recon_acc = pipeline(
            Tasks.nerf_recon_acc,
            model=snapshot_path,
            data_type='blender',
        )

        nerf_recon_acc(
            dict(data_dir=self.data_dir, render_dir=self.render_dir))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub(self):
        nerf_recon_acc = pipeline(
            Tasks.nerf_recon_acc,
            model=self.model_id,
            data_type='blender',
        )

        nerf_recon_acc(
            dict(data_dir=self.data_dir, render_dir=self.render_dir))
        print('facefusion.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub_default_model(self):
        nerf_recon_acc = pipeline(Tasks.nerf_recon_acc)
        nerf_recon_acc(
            dict(data_dir=self.data_dir, render_dir=self.render_dir))
        print('facefusion.test_run_modelhub_default_model done')


if __name__ == '__main__':
    unittest.main()
