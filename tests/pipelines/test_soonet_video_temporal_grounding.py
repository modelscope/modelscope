# Copyright 2022-2023 The Alibaba Fundamental Vision  Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.multi_modal.soonet import SOONet
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SOONetVideoTemporalGroundingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_temporal_grounding
        self.model_id = 'damo/multi-modal_soonet_video-temporal-grounding'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        soonet_pipeline = pipeline(self.task, self.model_id)
        result = soonet_pipeline(
            ('a man takes food out of the refrigerator.',
             'soonet_video_temporal_grounding_test_video.mp4'))
        print(f'soonet output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_load_model_from_pretrained(self):
        model = Model.from_pretrained(self.model_id)
        self.assertTrue(model.__class__ == SOONet)


if __name__ == '__main__':
    unittest.main()
