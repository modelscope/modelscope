# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ShopSegmentationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_shop_segmentation(self):
        input_location = 'data/test/images/shop_segmentation.jpg'
        model_id = 'damo/cv_vitb16_segmentation_shop-seg'
        shop_seg = pipeline(Tasks.shop_segmentation, model=model_id)
        result = shop_seg(input_location)
        import cv2
        # result[OutputKeys.MASKS] is segment map result,other keys are not used
        cv2.imwrite(input_location + '_shopseg.jpg', result[OutputKeys.MASKS])


if __name__ == '__main__':
    unittest.main()
