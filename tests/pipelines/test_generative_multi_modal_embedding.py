# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class GEMMMultiModalEmbeddingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.generative_multi_modal_embedding
        self.model_id = 'damo/multi-modal_rleg-vit-large-patch14'

    test_input = {
        'image': 'data/test/images/generative_multimodal.jpg',
        'text':
        'interior design of modern living room with fireplace in a new house',
        'captioning': False
    }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        generative_multi_modal_embedding_pipeline = pipeline(
            Tasks.generative_multi_modal_embedding, model=self.model_id)
        output = generative_multi_modal_embedding_pipeline(self.test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        generative_multi_modal_embedding_pipeline = pipeline(
            task=Tasks.generative_multi_modal_embedding)
        output = generative_multi_modal_embedding_pipeline(self.test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        generative_multi_modal_embedding_pipeline = pipeline(
            task=Tasks.generative_multi_modal_embedding, model=model)
        output = generative_multi_modal_embedding_pipeline(self.test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_output_captioning(self):
        generative_multi_modal_embedding_pipeline = pipeline(
            task=Tasks.generative_multi_modal_embedding, model=self.model_id)
        test_input = {'image': self.test_input['image'], 'captioning': True}
        output = generative_multi_modal_embedding_pipeline(test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_output_only_image(self):
        generative_multi_modal_embedding_pipeline = pipeline(
            task=Tasks.generative_multi_modal_embedding, model=self.model_id)
        test_input = {'image': self.test_input['image'], 'captioning': False}
        output = generative_multi_modal_embedding_pipeline(test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_output_only_text(self):
        generative_multi_modal_embedding_pipeline = pipeline(
            task=Tasks.generative_multi_modal_embedding, model=self.model_id)
        test_input = {'text': self.test_input['text']}
        output = generative_multi_modal_embedding_pipeline(test_input)
        print(output)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
