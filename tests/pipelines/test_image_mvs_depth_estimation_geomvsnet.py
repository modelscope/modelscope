# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageMVSDepthEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'image-multi-view-depth-estimation'
        self.model_id = 'Damo_XR_Lab/cv_geomvsnet_multi-view-depth-estimation_general'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_mvs_depth_estimation(self):
        estimator = pipeline(
            Tasks.image_multi_view_depth_estimation,
            model='Damo_XR_Lab/cv_geomvsnet_multi-view-depth-estimation_general'
        )
        model_dir = snapshot_download(self.model_id)
        input_location = os.path.join(model_dir, 'test_data')

        result = estimator(input_location)
        pcd = result[OutputKeys.OUTPUT]
        pcd.write('./pcd_fusion.ply')
        print('test_image_mvs_depth_estimation DONE')


if __name__ == '__main__':
    unittest.main()
