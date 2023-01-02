# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import panoptic_seg_masks_to_image
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class EasyCVPanopticSegmentationPipelineTest(unittest.TestCase,
                                             DemoCompatibilityCheck):
    img_path = 'data/test/images/image_semantic_segmentation.jpg'

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_r50_panoptic-segmentation_cocopan'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_r50(self):
        segmentor = pipeline(task=self.task, model=self.model_id)
        outputs = segmentor(self.img_path)
        draw_img = panoptic_seg_masks_to_image(outputs[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('print ' + self.model_id + ' success')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
