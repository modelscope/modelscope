# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import BadImageDetecingPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class BadImageDetectingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.bad_image_detecting
        self.model_id = 'damo/cv_mobilenet-v2_bad-image-detecting'
        self.test_img = 'data/test/images/dogs.jpg'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = BadImageDetecingPipeline(cache_path)
        pipeline.group_key = self.task
        out = pipeline(input=self.test_img)
        labels = out[OutputKeys.LABELS]
        scores = out[OutputKeys.SCORES]
        print('pipeline: the out_label is {}'.format(labels))
        print('pipeline: the out_score is {}'.format(scores))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(task=Tasks.bad_image_detecting, model=model)
        out = pipeline_ins(input=self.test_img)
        labels = out[OutputKeys.LABELS]
        scores = out[OutputKeys.SCORES]
        print('pipeline: the out_label is {}'.format(labels))
        print('pipeline: the out_score is {}'.format(scores))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.bad_image_detecting, model=self.model_id)
        out = pipeline_ins(input=self.test_img)
        labels = out[OutputKeys.LABELS]
        scores = out[OutputKeys.SCORES]
        print('pipeline: the out_label is {}'.format(labels))
        print('pipeline: the out_score is {}'.format(scores))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.bad_image_detecting)
        out = pipeline_ins(input=self.test_img)
        labels = out[OutputKeys.LABELS]
        scores = out[OutputKeys.SCORES]
        print('pipeline: the out_label is {}'.format(labels))
        print('pipeline: the out_score is {}'.format(scores))


if __name__ == '__main__':
    unittest.main()
