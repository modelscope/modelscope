# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import match_pair_visualization
from modelscope.utils.test_utils import test_level


class ImageMatchingFastTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'image-matching'
        self.model_id = 'Damo_XR_Lab/cv_transformer_image-matching_fast'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_matching(self):
        input_location = [[
            'data/test/images/image_matching1.jpg',
            'data/test/images/image_matching2.jpg'
        ]]
        estimator = pipeline(Tasks.image_matching, model=self.model_id)
        result = estimator(input_location)
        kpts0, kpts1, confidence = result[0][OutputKeys.MATCHES]

        match_pair_visualization(
            input_location[0][0],
            input_location[0][1],
            kpts0,
            kpts1,
            confidence,
            output_filename='lightglue-matches.png',
            method='lightglue')

        print('test_image_matching DONE')


if __name__ == '__main__':
    unittest.main()
