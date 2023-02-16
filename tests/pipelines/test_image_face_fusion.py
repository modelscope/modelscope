# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageFaceFusionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_face_fusion
        self.model_id = 'damo/cv_unet-image-face-fusion_damo'
        self.template_img = 'data/test/images/facefusion_template.jpg'
        self.user_img = 'data/test/images/facefusion_user.jpg'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        image_face_fusion = pipeline(
            Tasks.image_face_fusion, model=snapshot_path)

        result = image_face_fusion(
            dict(template=self.template_img, user=self.user_img))
        cv2.imwrite('result_facefusion.png', result[OutputKeys.OUTPUT_IMG])
        print('facefusion.test_run_direct_model_download done')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_face_fusion = pipeline(
            Tasks.image_face_fusion, model=self.model_id)

        result = image_face_fusion(
            dict(template=self.template_img, user=self.user_img))
        cv2.imwrite('result_facefusion.png', result[OutputKeys.OUTPUT_IMG])
        print('facefusion.test_run_modelhub done')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_face_fusion = pipeline(Tasks.image_face_fusion)

        result = image_face_fusion(
            dict(template=self.template_img, user=self.user_img))
        cv2.imwrite('result_facefusion.png', result[OutputKeys.OUTPUT_IMG])
        print('facefusion.test_run_modelhub_default_model done')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
