# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.msdatasets import MsDataset
from modelscope.trainers.cv import NeRFReconAccTrainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level


class TestNeRFReconAccTrainer(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'

        data_dir = MsDataset.load(
            'nerf_recon_dataset',
            namespace='damo',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['train']

        trainer = NeRFReconAccTrainer(
            model=model_id,
            data_type='blender',
            work_dir='exp_nerf',
            render_images=False)

        nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
        blender_scene = 'lego'
        nerf_synthetic_dataset = os.path.join(nerf_synthetic_dataset,
                                              blender_scene)
        trainer.train(data_dir=nerf_synthetic_dataset)


if __name__ == '__main__':
    unittest.main()
