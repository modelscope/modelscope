# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TableRecognitionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_dla34_table-structure-recognition_cycle-centernet'
        self.test_image = 'data/test/images/table_recognition.jpg'
        self.task = Tasks.table_recognition

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        print('table recognition results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        table_recognition = pipeline(Tasks.table_recognition, model=self.model_id)
        self.pipeline_inference(table_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        table_recognition = pipeline(Tasks.table_recognition)
        self.pipeline_inference(table_recognition, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
