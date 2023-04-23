# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import PIL

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class OCRRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_convnextTiny_ocr-recognition-general_damo'
        self.test_image = 'data/test/images/ocr_recognition.jpg'
        self.task = Tasks.ocr_recognition

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        print('ocr recognition results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=self.model_id,
            model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_handwritten(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo',
            model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_scene(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-scene_damo',
            model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_document(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-document_damo',
            model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_licenseplate(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo',
            model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_crnn(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_crnn_ocr-recognition-general_damo',
            model_revision='v2.2.2')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub_PILinput(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=self.model_id,
            model_revision='v2.3.0')
        imagePIL = PIL.Image.open(self.test_image)
        self.pipeline_inference(ocr_recognition, imagePIL)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition, model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=self.model_id,
            model_revision='v2.3.0',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_handwritten_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo',
            model_revision='v2.3.0',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_scene_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-scene_damo',
            model_revision='v2.3.0',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_document_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-document_damo',
            model_revision='v2.3.0',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_licenseplate_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo',
            model_revision='v2.3.0',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_crnn_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_crnn_ocr-recognition-general_damo',
            model_revision='v2.2.2',
            device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub_PILinput_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=self.model_id,
            model_revision='v2.3.0',
            device='cpu')
        imagePIL = PIL.Image.open(self.test_image)
        self.pipeline_inference(ocr_recognition, imagePIL)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model_cpu(self):
        ocr_recognition = pipeline(
            Tasks.ocr_recognition, model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)


if __name__ == '__main__':
    unittest.main()
