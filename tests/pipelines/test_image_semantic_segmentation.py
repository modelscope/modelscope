import unittest

import cv2
import PIL

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import semantic_seg_masks_to_image
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level


class ImageSemanticSegmentationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_semantic_segmentation_panmerge(self):
        input_location = 'data/test/images/image_semantic_segmentation.jpg'
        model_id = 'damo/cv_swinL_semantic-segmentation_cocopanmerge'
        segmenter = pipeline(Tasks.image_segmentation, model=model_id)
        result = segmenter(input_location)

        draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('test_image_semantic_segmentation_panmerge DONE')

        PIL_array = PIL.Image.open(input_location)
        result = segmenter(PIL_array)

        draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('test_image_semantic_segmentation_panmerge_from_PIL DONE')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_semantic_segmentation_vitadapter(self):
        input_location = 'data/test/images/image_semantic_segmentation.jpg'
        model_id = 'damo/cv_vitadapter_semantic-segmentation_cocostuff164k'
        segmenter = pipeline(Tasks.image_segmentation, model=model_id)
        result = segmenter(input_location)

        draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('test_image_semantic_segmentation_vitadapter DONE')

        PIL_array = PIL.Image.open(input_location)
        result = segmenter(PIL_array)

        draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('test_image_semantic_segmentation_vitadapter_from_PIL DONE')


if __name__ == '__main__':
    unittest.main()
