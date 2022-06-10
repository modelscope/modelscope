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


class OCRDetectionTest(unittest.TestCase):

    def test_run(self):
        # wget http://duguang-database.oss-cn-hangzhou-zmf.aliyuncs.com/ \
        # model_zoo/ocr_detection_line_level_v0531.zip and unzip
        model_path = '/mnt/common/xixing.tj/project/ocr-seglink/exp/general_\
                resnet18_public_baseline_sequence_768finetune_linewithchar/checkpoint-80000'

        # with tempfile.NamedTemporaryFile('wb', suffix='.pb') as ofile:
        #    ofile.write(File.read(model_path))
        ocr_detection = pipeline(Tasks.ocr_detection, model_path=model_path)

        result = ocr_detection(
            'https://duguang-image-viewer.oss-cn-hangzhou-zmf.aliyuncs.com/\
                    xixing.tj/165391027548/TB1bKwlHpXXXXc1XXXXXXXXXXXX_%21%210-item_pic.jpg.jpg'
        )
        print(result)


if __name__ == '__main__':
    unittest.main()
