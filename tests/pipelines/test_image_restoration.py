# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageRestorationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_demoireing
        self.model_id = 'damo/cv_uhdm_image-demoireing'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_demoireing(self):
        input_location = 'data/test/images/image_moire.jpg'
        model_id = 'damo/cv_uhdm_image-demoireing'
        image_demoire = pipeline(Tasks.image_demoireing, model=model_id)
        result = image_demoire(input_location)
        from PIL import Image
        Image.fromarray(result[OutputKeys.OUTPUT_IMG]).save(input_location
                                                            + '_demoire.jpg')


if __name__ == '__main__':
    unittest.main()
