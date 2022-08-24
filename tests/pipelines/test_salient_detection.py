# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SalientDetectionTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_salient_detection(self):
        input_location = 'data/test/images/image_salient_detection.jpg'
        model_id = 'damo/cv_u2net_salient-detection'
        salient_detect = pipeline(Tasks.image_segmentation, model=model_id)
        result = salient_detect(input_location)
        import cv2
        # result[OutputKeys.MASKS] is salient map result,other keys are not used
        cv2.imwrite(input_location + '_salient.jpg', result[OutputKeys.MASKS])


if __name__ == '__main__':
    unittest.main()
