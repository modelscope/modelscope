# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextDrivenSegmentationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_text_driven_segmentation(self):
        input_location = 'data/test/images/text_driven_segmentation.jpg'
        test_input = {
            'image': input_location,
            'text': 'bear',
        }
        model_id = 'damo/cv_vitl16_segmentation_text-driven-seg'
        shop_seg = pipeline(Tasks.text_driven_segmentation, model=model_id)
        result = shop_seg(test_input)
        import cv2
        # result[OutputKeys.MASKS] is segment map result,other keys are not used
        cv2.imwrite(input_location + '_lseg.jpg', result[OutputKeys.MASKS])


if __name__ == '__main__':
    unittest.main()
