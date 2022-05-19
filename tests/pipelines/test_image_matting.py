# Copyright (c) Alibaba, Inc. and its affiliates.

import tempfile
import unittest
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL

from maas_lib.fileio import File
from maas_lib.pipelines import pipeline
from maas_lib.utils.constant import Tasks


class ImageMattingTest(unittest.TestCase):

    def test_run(self):
        model_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs' \
                     '.com/data/test/maas/image_matting/matting_person.pb'
        with tempfile.NamedTemporaryFile('wb', suffix='.pb') as ofile:
            ofile.write(File.read(model_path))
            img_matting = pipeline(Tasks.image_matting, model_path=ofile.name)

            result = img_matting(
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
            )
            cv2.imwrite('result.png', result['output_png'])


if __name__ == '__main__':
    unittest.main()
