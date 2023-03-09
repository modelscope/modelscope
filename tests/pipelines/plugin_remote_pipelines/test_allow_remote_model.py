# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.plugins import PluginsManager
from modelscope.utils.test_utils import test_level


class AllowRemoteModelTest(unittest.TestCase):

    def setUp(self):
        self.package = 'moviepy'

    def tearDown(self):
        # make sure uninstalled after installing
        uninstall_args = [self.package, '-y']
        PluginsManager.pip_command('uninstall', uninstall_args)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_bilibili_image(self):

        model_path = snapshot_download(
            'bilibili/cv_bilibili_image-super-resolution', revision='v1.0.5')
        file_path = f'{model_path}/demos/title-compare1.png'
        weight_path = f'{model_path}/weights_v3/up2x-latest-denoise3x.pth'
        inference = pipeline(
            'image-super-resolution',
            model='bilibili/cv_bilibili_image-super-resolution',
            weight_path=weight_path,
            device='cpu',
            half=False)  # GPU环境可以设置为True

        output = inference(file_path, tile_mode=0, cache_mode=1, alpha=1)
        print(output)
