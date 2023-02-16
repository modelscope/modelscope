# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class NeRFReconAccTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_face_fusion
        self.model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'
        self.video_path = 'data/test/videos/video_nerf_recon_test.mp4'
        self.data_dir = 'data/test/videos/nerf_dir'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_by_direct_model_download(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        nerf_recon_acc = pipeline(
            Tasks.nerf_recon_acc,
            model=snapshot_path,
        )

        result = nerf_recon_acc(
            dict(data_dir=self.data_dir, video_input_path=self.video_path))
        print(result[OutputKeys.OUTPUT_VIDEO])
        print('facefusion.test_run_direct_model_download done')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub(self):
        nerf_recon_acc = pipeline(Tasks.nerf_recon_acc, model=self.model_id)

        result = nerf_recon_acc(
            dict(data_dir=self.data_dir, video_input_path=self.video_path))
        print(result[OutputKeys.OUTPUT_VIDEO])
        print('facefusion.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub_default_model(self):
        nerf_recon_acc = pipeline(Tasks.nerf_recon_acc)

        result = nerf_recon_acc(
            dict(data_dir=self.data_dir, video_input_path=self.video_path))
        print(result[OutputKeys.OUTPUT_VIDEO])
        print('facefusion.test_run_modelhub_default_model done')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
