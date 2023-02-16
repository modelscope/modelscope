# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageQualityAssessmentDegradationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level

sys.path.insert(0, '.')


class ImageQualityAssessmentDegradationTest(unittest.TestCase,
                                            DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_quality_assessment_degradation
        self.model_id = 'damo/cv_resnet50_image-quality-assessment_degradation'
        self.test_img = 'data/test/images/dogs.jpg'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageQualityAssessmentDegradationPipeline(cache_path)
        pipeline.group_key = self.task
        out_path = pipeline(input=self.test_img)[OutputKeys.SCORES]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(
            task=Tasks.image_quality_assessment_degradation, model=model)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORES]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.image_quality_assessment_degradation,
            model=self.model_id)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORES]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(
            task=Tasks.image_quality_assessment_degradation)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORES]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
