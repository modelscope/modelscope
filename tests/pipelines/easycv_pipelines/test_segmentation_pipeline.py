# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EasyCVSegmentationPipelineTest(unittest.TestCase):

    img_path = 'data/test/images/image_segmentation.jpg'

    def _internal_test__(self, model_id):
        img = np.asarray(Image.open(self.img_path))

        semantic_seg = pipeline(task=Tasks.image_segmentation, model=model_id)
        outputs = semantic_seg(self.img_path)

        self.assertEqual(len(outputs), 1)

        results = outputs[0]
        self.assertListEqual(
            list(img.shape)[:2], list(results['seg_pred'][0].shape))
        self.assertListEqual(results['seg_pred'][0][1, 4:10].tolist(),
                             [161 for i in range(6)])
        self.assertListEqual(results['seg_pred'][0][-1, -10:].tolist(),
                             [133 for i in range(10)])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b0(self):
        model_id = 'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b1(self):
        model_id = 'damo/cv_segformer-b1_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b2(self):
        model_id = 'damo/cv_segformer-b2_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b3(self):
        model_id = 'damo/cv_segformer-b3_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b4(self):
        model_id = 'damo/cv_segformer-b4_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b5(self):
        model_id = 'damo/cv_segformer-b5_image_semantic-segmentation_coco-stuff164k'
        self._internal_test__(model_id)


if __name__ == '__main__':
    unittest.main()
