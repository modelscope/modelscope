# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from distutils.version import LooseVersion

import cv2
import easycv
import numpy as np
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import semantic_seg_masks_to_image
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class EasyCVSegmentationPipelineTest(unittest.TestCase,
                                     DemoCompatibilityCheck):
    img_path = 'data/test/images/image_segmentation.jpg'

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k'

    def _internal_test_(self, model_id):
        semantic_seg = pipeline(task=Tasks.image_segmentation, model=model_id)
        outputs = semantic_seg(self.img_path)

        draw_img = semantic_seg_masks_to_image(outputs[OutputKeys.MASKS])
        cv2.imwrite('result.jpg', draw_img)
        print('test ' + model_id + ' DONE')

    def _internal_test_batch_(self, model_id, num_samples=2, batch_size=2):
        # TODO: support in the future
        img = np.asarray(Image.open(self.img_path))
        num_samples = num_samples
        batch_size = batch_size
        semantic_seg = pipeline(
            task=Tasks.image_segmentation,
            model=model_id,
            batch_size=batch_size)
        outputs = semantic_seg([self.img_path] * num_samples)

        self.assertEqual(semantic_seg.predict_op.batch_size, batch_size)
        self.assertEqual(len(outputs), num_samples)

        for output in outputs:
            self.assertListEqual(
                list(img.shape)[:2], list(output['seg_pred'].shape))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b0(self):
        model_id = 'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b1(self):
        model_id = 'damo/cv_segformer-b1_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b2(self):
        model_id = 'damo/cv_segformer-b2_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b3(self):
        model_id = 'damo/cv_segformer-b3_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b4(self):
        model_id = 'damo/cv_segformer-b4_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_segformer_b5(self):
        model_id = 'damo/cv_segformer-b5_image_semantic-segmentation_coco-stuff164k'
        self._internal_test_(model_id)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
