# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageDepthEstimationBtsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_depth_estimation
        self.model_id = 'damo/cv_densenet161_image-depth-estimation_bts'
        self.image = 'data/test/images/image_depth_estimation_kitti_007517.png'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_bts = pipeline(task=self.task, model=model)
        result = pipeline_bts(input=self.image)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        cv2.imwrite('result_modelhub.jpg', depth_vis)
        print('Test run with model from modelhub ok.')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_bts = pipeline(task=self.task, model=self.model_id)
        result = pipeline_bts(input=self.image)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        cv2.imwrite('result_modelname.jpg', depth_vis)
        print('Test run with model name ok.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline_bts = pipeline(self.task, model=cache_path)
        result = pipeline_bts(input=self.image)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        cv2.imwrite('result_snapshot.jpg', depth_vis)
        print('Test run with snapshot ok.')


if __name__ == '__main__':
    unittest.main()
