# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import tempfile
import unittest

import cv2

from modelscope.fileio import File
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class StyleTransferTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_aams_style-transfer_damo'

    @unittest.skip('deprecated, download model from model hub instead')
    def test_run_by_direct_model_download(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        style_transfer = pipeline(Tasks.style_transfer, model=snapshot_path)

        result = style_transfer(
            'data/test/images/style_transfer_content.jpg',
            style='data/test/images/style_transfer_style.jpg')
        cv2.imwrite('result_styletransfer1.png', result[OutputKeys.OUTPUT_IMG])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub(self):
        style_transfer = pipeline(Tasks.style_transfer, model=self.model_id)

        result = style_transfer(
            'data/test/images/style_transfer_content.jpg',
            style='data/test/images/style_transfer_style.jpg')
        cv2.imwrite('result_styletransfer2.png', result[OutputKeys.OUTPUT_IMG])
        print('style_transfer.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        style_transfer = pipeline(Tasks.style_transfer)

        result = style_transfer(
            'data/test/images/style_transfer_content.jpg',
            style='data/test/images/style_transfer_style.jpg')
        cv2.imwrite('result_styletransfer3.png', result[OutputKeys.OUTPUT_IMG])
        print('style_transfer.test_run_modelhub_default_model done')


if __name__ == '__main__':
    unittest.main()
