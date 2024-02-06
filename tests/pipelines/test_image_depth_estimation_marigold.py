# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import os

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageDepthEstimationMarigoldPipeline
from modelscope.outputs import OutputKeys
from modelscope.hub.snapshot_download import snapshot_download

from modelscope.utils.test_utils import test_level


class ImageDepthEstimationMarigoldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_depth_estimation
        self.model_id = 'Damo_XR_Lab/cv_marigold_monocular-depth-estimation'
        self.image = 'data/in-the-wild_example/example_0.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        marigold = pipeline(task=self.task, model=self.model_id)
        input_path = os.path.join(marigold.model, self.image)
        result = marigold(input=input_path)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        depth_vis.save('result_modelname.jpg')
        print('Test run with model name ok.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        marigold_pipe = ImageDepthEstimationMarigoldPipeline(cache_path)
        marigold_pipe.group_key = self.task
        input_path = os.path.join(cache_path, self.image)
        result = marigold_pipe(input=input_path)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        depth_vis.save('result_snapshot.jpg')
        print('Test run with snapshot ok.')


if __name__ == '__main__':
    unittest.main()
