# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageQualityAssessmentMosPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageQualityAssessmentMosTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_quality_assessment_mos
        self.model_id = 'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC'
        self.test_img = 'data/test/images/dogs.jpg'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageQualityAssessmentMosPipeline(cache_path)
        pipeline.group_key = self.task
        out_path = pipeline(input=self.test_img)[OutputKeys.SCORE]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(
            task=Tasks.image_quality_assessment_mos, model=model)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORE]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.image_quality_assessment_mos, model=self.model_id)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORE]
        print('pipeline: the out_path is {}'.format(out_path))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.image_quality_assessment_mos)
        out_path = pipeline_ins(input=self.test_img)[OutputKeys.SCORE]
        print('pipeline: the out_path is {}'.format(out_path))


if __name__ == '__main__':
    unittest.main()
