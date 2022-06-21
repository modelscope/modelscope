# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageCaptionTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run(self):
        img_captioning = pipeline(
            Tasks.image_captioning,
            model='damo/ofa_image-caption_coco_large_en')
        result = img_captioning('data/test/images/image_captioning.png')
        print(result['caption'])


if __name__ == '__main__':
    unittest.main()
