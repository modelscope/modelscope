# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.models.cv.vop_retrieval import VoP
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VopRetrievalTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.vop_retrieval
        # self.model_id = '../cv_vit-b32_retrieval_vop'
        self.model_id = 'damo/cv_vit-b32_retrieval_vop'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        vop_pipeline = pipeline(self.task, self.model_id)
        # t2v
        result = vop_pipeline('a squid is talking')
        # v2t
        # result = vop_pipeline('video10.mp4')
        print(f'vop output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_load_model_from_pretrained(self):
        # model = Model.from_pretrained('../cv_vit-b32_retrieval_vop')
        model = Model.from_pretrained('damo/cv_vit-b32_retrieval_vop')
        self.assertTrue(model.__class__ == VoP)


if __name__ == '__main__':
    unittest.main()
