# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import PIL

from modelscope.preprocessors import load_image


class ImagePreprocessorTest(unittest.TestCase):

    def test_load(self):
        img = load_image('data/test/images/image_matting.png')
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertEqual(img.size, (948, 533))


if __name__ == '__main__':
    unittest.main()
