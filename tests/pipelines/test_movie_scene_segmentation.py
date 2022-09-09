# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class MovieSceneSegmentationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.movie_scene_segmentation
        self.model_id = 'damo/cv_resnet50-bert_video-scene-segmentation_movienet'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_movie_scene_segmentation(self):
        input_location = 'data/test/videos/movie_scene_segmentation_test_video.mp4'
        movie_scene_segmentation_pipeline = pipeline(
            Tasks.movie_scene_segmentation, model=self.model_id)
        result = movie_scene_segmentation_pipeline(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_movie_scene_segmentation_with_default_task(self):
        input_location = 'data/test/videos/movie_scene_segmentation_test_video.mp4'
        movie_scene_segmentation_pipeline = pipeline(
            Tasks.movie_scene_segmentation)
        result = movie_scene_segmentation_pipeline(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
