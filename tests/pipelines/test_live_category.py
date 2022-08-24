# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class LiveCategoryTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        category_pipeline = pipeline(
            Tasks.live_category, model='damo/cv_resnet50_live-category')
        result = category_pipeline(
            'data/test/videos/live_category_test_video.mp4')

        print(f'live category output: {result}.')


if __name__ == '__main__':
    unittest.main()
