# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class ImageControl3dPortraitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_vit_image-control-3d-portrait-synthesis'
        self.test_image = 'data/test/images/image_control_3d_portrait.jpg'
        self.save_dir = 'exp'
        os.makedirs(self.save_dir, exist_ok=True)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id, revision='v1.1')
        print('model dir is: {}'.format(model_dir))
        image_control_3d_portrait = pipeline(
            Tasks.image_control_3d_portrait,
            model=model_dir,
        )
        image_control_3d_portrait(
            dict(image=self.test_image, save_dir=self.save_dir))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_control_3d_portrait = pipeline(
            Tasks.image_control_3d_portrait,
            model=self.model_id,
        )

        image_control_3d_portrait(
            dict(image=self.test_image, save_dir=self.save_dir))
        print('image_control_3d_portrait.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_control_3d_portrait = pipeline(Tasks.image_control_3d_portrait)
        image_control_3d_portrait(
            dict(image=self.test_image, save_dir=self.save_dir))
        print('image_control_3d_portrait.test_run_modelhub_default_model done')


if __name__ == '__main__':
    unittest.main()
