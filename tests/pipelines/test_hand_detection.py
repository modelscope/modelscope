# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.domain_specific_object_detection
        self.model_id = 'damo/cv_yolox-pai_hand-detection'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_hand_detection_pipeline(self):
        test_image = 'data/test/images/hand_detection.jpg'

        hand_detection = pipeline(self.task, model=self.model_id)

        result = hand_detection(test_image)
        hand_detection.show_result(test_image, result,
                                   'hand_detection_ret.jpg')

        print(f'hand detection result={result}')


if __name__ == '__main__':
    unittest.main()
