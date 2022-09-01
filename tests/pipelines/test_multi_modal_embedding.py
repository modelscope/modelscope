# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import torch

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiModalEmbeddingTest(unittest.TestCase):
    model_id = 'damo/multi-modal_clip-vit-base-patch16_zh'
    test_input = {'text': '皮卡丘'}
    model_version = 'dev'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        pipeline_multi_modal_embedding = pipeline(
            Tasks.multi_modal_embedding,
            model=self.model_id,
            model_revision=self.model_version)
        text_embedding = pipeline_multi_modal_embedding(
            self.test_input)[OutputKeys.TEXT_EMBEDDING]
        print('l1-norm: {}'.format(
            torch.norm(text_embedding, p=1, dim=-1).item()))
        print('l2-norm: {}'.format(torch.norm(text_embedding,
                                              dim=-1).item()))  # should be 1.0

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(
            self.model_id, revision=self.model_version)
        pipeline_multi_modal_embedding = pipeline(
            task=Tasks.multi_modal_embedding, model=model)
        text_embedding = pipeline_multi_modal_embedding(
            self.test_input)[OutputKeys.TEXT_EMBEDDING]
        print('l1-norm: {}'.format(
            torch.norm(text_embedding, p=1, dim=-1).item()))
        print('l2-norm: {}'.format(torch.norm(text_embedding,
                                              dim=-1).item()))  # should be 1.0

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_multi_modal_embedding = pipeline(
            task=Tasks.multi_modal_embedding,
            model_revision=self.model_version)
        text_embedding = pipeline_multi_modal_embedding(
            self.test_input)[OutputKeys.TEXT_EMBEDDING]
        print('l1-norm: {}'.format(
            torch.norm(text_embedding, p=1, dim=-1).item()))
        print('l2-norm: {}'.format(torch.norm(text_embedding,
                                              dim=-1).item()))  # should be 1.0


if __name__ == '__main__':
    unittest.main()
