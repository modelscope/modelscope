# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class UniversalMattingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_unet_universal-matting'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        input_location = ['data/test/images/universal_matting.jpg']

        dataset = MsDataset.load(input_location, target='image')
        img_matting = pipeline(Tasks.universal_matting, model=self.model_id)
        result = img_matting(dataset)
        cv2.imwrite('result.png', next(result)[OutputKeys.OUTPUT_IMG])
        print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        img_matting = pipeline(Tasks.universal_matting, model=self.model_id)

        result = img_matting('data/test/images/universal_matting.jpg')
        cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
        print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
