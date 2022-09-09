import unittest

import numpy as np

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ProductRetrievalEmbeddingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.product_retrieval_embedding
        self.model_id = 'damo/cv_resnet50_product-bag-embedding-models'

    img_input = 'data/test/images/product_embed_bag.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        product_embed = pipeline(Tasks.product_retrieval_embedding,
                                 self.model_id)
        result = product_embed(self.img_input)[OutputKeys.IMG_EMBEDDING]
        print('abs sum value is: {}'.format(np.sum(np.abs(result))))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        product_embed = pipeline(
            task=Tasks.product_retrieval_embedding, model=model)
        result = product_embed(self.img_input)[OutputKeys.IMG_EMBEDDING]
        print('abs sum value is: {}'.format(np.sum(np.abs(result))))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        product_embed = pipeline(task=Tasks.product_retrieval_embedding)
        result = product_embed(self.img_input)[OutputKeys.IMG_EMBEDDING]
        print('abs sum value is: {}'.format(np.sum(np.abs(result))))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
