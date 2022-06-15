# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import PIL

from modelscope.preprocessors import load_image
from modelscope.utils.logger import get_logger


class ImagePreprocessorTest(unittest.TestCase):

    def test_load(self):
        img = load_image(
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
        )
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertEqual(img.size, (948, 533))


if __name__ == '__main__':
    unittest.main()
