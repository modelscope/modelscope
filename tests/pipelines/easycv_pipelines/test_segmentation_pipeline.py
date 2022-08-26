# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EasyCVSegmentationPipelineTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_segformer_b0(self):
        img_path = 'data/test/images/image_segmentation.jpg'
        model_id = 'EasyCV/EasyCV-Segformer-b0'
        img = np.asarray(Image.open(img_path))

        object_detect = pipeline(task=Tasks.image_segmentation, model=model_id)
        outputs = object_detect(img_path)
        self.assertEqual(len(outputs), 1)

        results = outputs[0]
        self.assertListEqual(
            list(img.shape)[:2], list(results['seg_pred'][0].shape))
        self.assertListEqual(results['seg_pred'][0][1, :10].tolist(),
                             [161 for i in range(10)])
        self.assertListEqual(results['seg_pred'][0][-1, -10:].tolist(),
                             [133 for i in range(10)])


if __name__ == '__main__':
    unittest.main()
