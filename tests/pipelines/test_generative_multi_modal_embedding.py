# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class GEMMMultiModalEmbeddingTest(unittest.TestCase):
    model_id = 'damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding'
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


if __name__ == '__main__':
    unittest.main()
