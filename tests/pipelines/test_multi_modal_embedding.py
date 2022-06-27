# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiModalEmbeddingTest(unittest.TestCase):
    model_id = 'damo/multi-modal_clip-vit-large-patch14-chinese_multi-modal-embedding'
    test_text = {'text': '一张风景图'}

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        pipe_line_multi_modal_embedding = pipeline(
            Tasks.multi_modal_embedding, model=self.model_id)
        test_str_embedding = pipe_line_multi_modal_embedding(
            self.test_text)['text_embedding']
        print(np.sum(np.abs(test_str_embedding)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipe_line_multi_modal_embedding = pipeline(
            task=Tasks.multi_modal_embedding, model=model)
        test_str_embedding = pipe_line_multi_modal_embedding(
            self.test_text)['text_embedding']
        print(np.sum(np.abs(test_str_embedding)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipe_line_multi_modal_embedding = pipeline(
            task=Tasks.multi_modal_embedding, model=self.model_id)
        test_str_embedding = pipe_line_multi_modal_embedding(
            self.test_text)['text_embedding']
        print(np.sum(np.abs(test_str_embedding)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipe_line_multi_modal_embedding = pipeline(
            task=Tasks.multi_modal_embedding)
        test_str_embedding = pipe_line_multi_modal_embedding(
            self.test_text)['text_embedding']
        print(np.sum(np.abs(test_str_embedding)))


if __name__ == '__main__':
    unittest.main()
