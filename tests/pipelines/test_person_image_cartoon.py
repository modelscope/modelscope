# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageCartoonTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_unet_person-image-cartoon_compound-models'
        self.test_image = \
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com' \
            '/data/test/maas/image_carton/test.png'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        if result is not None:
            cv2.imwrite('result.png', result['output_png'])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skip('deprecated, download model from model hub instead')
    def test_run_by_direct_model_download(self):
        model_dir = './assets'
        if not os.path.exists(model_dir):
            os.system(
                'wget https://invi-label.oss-cn-shanghai.aliyuncs.com/label/model/cartoon/assets.zip'
            )
            os.system('unzip assets.zip')

        img_cartoon = pipeline(Tasks.image_generation, model=model_dir)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub(self):
        img_cartoon = pipeline(Tasks.image_generation, model=self.model_id)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        img_cartoon = pipeline(Tasks.image_generation)
        self.pipeline_inference(img_cartoon, self.test_image)


if __name__ == '__main__':
    unittest.main()
