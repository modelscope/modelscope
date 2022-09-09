import unittest

import cv2
import PIL

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import panoptic_seg_masks_to_image
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImagePanopticSegmentationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_swinL_panoptic-segmentation_cocopan'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_panoptic_segmentation(self):
        input_location = 'data/test/images/image_panoptic_segmentation.jpg'
        pan_segmentor = pipeline(Tasks.image_segmentation, model=self.model_id)
        result = pan_segmentor(input_location)

        draw_img = panoptic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('print test_image_panoptic_segmentation return success')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_panoptic_segmentation_from_PIL(self):
        input_location = 'data/test/images/image_panoptic_segmentation.jpg'
        pan_segmentor = pipeline(Tasks.image_segmentation, model=self.model_id)
        PIL_array = PIL.Image.open(input_location)
        result = pan_segmentor(PIL_array)

        draw_img = panoptic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('print test_image_panoptic_segmentation from PIL return success')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
