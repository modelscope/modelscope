# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG' or extend == '.HEIC':
                L.append(os.path.join(root, file))
    return L


class ImageCartoonTest(unittest.TestCase):

    def test_run(self):
        model_dir = './assets'
        if not os.path.exists(model_dir):
            os.system(
                'wget https://invi-label.oss-cn-shanghai.aliyuncs.com/label/model/cartoon/assets.zip'
            )
            os.system('unzip assets.zip')

        img_cartoon = pipeline(Tasks.image_generation, model=model_dir)
        result = img_cartoon(os.path.join(model_dir, 'test.png'))
        if result is not None:
            cv2.imwrite('result.png', result['output_png'])


if __name__ == '__main__':
    unittest.main()
