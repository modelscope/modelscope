# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageStyleTransferTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_style_transfer
        self.model_id = 'damo/cv_aams_style-transfer_damo'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        image_style_transfer = pipeline(
            Tasks.image_style_transfer, model=snapshot_path)

        result = image_style_transfer(
            dict(
                content='data/test/images/style_transfer_content.jpg',
                style='data/test/images/style_transfer_style.jpg'))
        cv2.imwrite('result_styletransfer1.png', result[OutputKeys.OUTPUT_IMG])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_style_transfer = pipeline(
            Tasks.image_style_transfer, model=self.model_id)

        result = image_style_transfer(
            dict(
                content='data/test/images/style_transfer_content.jpg',
                style='data/test/images/style_transfer_style.jpg'))
        cv2.imwrite('result_styletransfer2.png', result[OutputKeys.OUTPUT_IMG])
        print('style_transfer.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_style_transfer = pipeline(Tasks.image_style_transfer)

        result = image_style_transfer(
            dict(
                content='data/test/images/style_transfer_content.jpg',
                style='data/test/images/style_transfer_style.jpg'))
        cv2.imwrite('result_styletransfer3.png', result[OutputKeys.OUTPUT_IMG])
        print('style_transfer.test_run_modelhub_default_model done')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
