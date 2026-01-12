# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from PIL import Image

from modelscope.preprocessors import load_image
from modelscope.utils.test_utils import test_level


class ImagePreprocessorTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_load(self):
        img = load_image('data/test/images/image_matting.png')
        self.assertTrue(isinstance(img, Image.Image))
        self.assertEqual(img.size, (948, 533))


if __name__ == '__main__':
    unittest.main()
