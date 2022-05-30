# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = osp.join(tmp_dir, 'matting_person.pb')
            with open(model_file, 'wb') as ofile:
                ofile.write(File.read(model_path))
            img_matting = pipeline(Tasks.image_matting, model=tmp_dir)

            result = img_matting(
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
            )
            cv2.imwrite('result.png', result['output_png'])

    def test_run_modelhub(self):
        img_matting = pipeline(
            Tasks.image_matting, model='damo/image-matting-person')

        result = img_matting(
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
        )
        cv2.imwrite('result.png', result['output_png'])


if __name__ == '__main__':
    unittest.main()
