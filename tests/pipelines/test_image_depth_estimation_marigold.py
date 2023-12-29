# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import os

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageDepthEstimationMarigoldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_depth_estimation
        self.model_id = 'Damo_XR_Lab/cv_marigold_monocular-depth-estimation'
        self.image = 'data/in-the-wild_example/example_0.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_marigold = pipeline(task=self.task, model=model)
        input_path = os.path.join(pipeline_marigold.model.model_dir, self.image)
        result = pipeline_marigold(input=input_path)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        depth_vis.save('result_modelhub.jpg')
        print('Test run with model from modelhub ok.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_marigold = pipeline(task=self.task, model=self.model_id)
        input_path = os.path.join(pipeline_marigold.model.model_dir, self.image)
        result = pipeline_marigold(input=input_path)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        depth_vis.save('result_modelname.jpg')
        print('Test run with model name ok.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline_marigold = pipeline(self.task, model=cache_path)
        input_path = os.path.join(pipeline_marigold.model.model_dir, self.image)
        result = pipeline_marigold(input=input_path)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        depth_vis.save('result_snapshot.jpg')
        print('Test run with snapshot ok.')


if __name__ == '__main__':
    unittest.main()
