# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiModalSimilarityTest(unittest.TestCase):
    model_id = 'damo/multi-modal_team-vit-large-patch14_multi-modal-similarity'
    test_input = {
        'img': 'data/test/images/generative_multimodal.jpg',
        'text': '起居室照片'
    }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        multi_modal_similarity_pipeline = pipeline(
            Tasks.multi_modal_similarity, model=self.model_id)
        output = multi_modal_similarity_pipeline(self.test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        multi_modal_similarity_pipeline = pipeline(
            task=Tasks.multi_modal_similarity)
        output = multi_modal_similarity_pipeline(self.test_input)
        print(output)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        multi_modal_similarity_pipeline = pipeline(
            task=Tasks.multi_modal_similarity, model=model)
        output = multi_modal_similarity_pipeline(self.test_input)
        print(output)


if __name__ == '__main__':
    unittest.main()
