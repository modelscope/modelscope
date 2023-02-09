# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines.multi_modal.gridvlp_pipeline import (
    GridVlpClassificationPipeline, GridVlpEmbeddingPipeline)
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class GridVlpClassificationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate'

    text = '女装快干弹力轻型短裤448575'
    image = 'https://yejiabo-public.oss-cn-zhangjiakou.aliyuncs.com/alinlp/clothes.png'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_pipeline(self):

        gridvlp_classification_pipeline = GridVlpClassificationPipeline(
            'rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate')
        input_params = {'text': self.text, 'image': self.image}
        inputs = gridvlp_classification_pipeline.preprocess(input_params)
        outputs = gridvlp_classification_pipeline.forward(inputs)
        print(f'text: {self.text}\nimage: {self.image}\n'
              f'outputs: {outputs}')

        gridvlp_classification_pipeline = GridVlpClassificationPipeline(
            'rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate-large'
        )
        input_params = {'text': self.text, 'image': self.image}
        inputs = gridvlp_classification_pipeline.preprocess(input_params)
        outputs = gridvlp_classification_pipeline.forward(inputs)
        print(f'text: {self.text}\nimage: {self.image}\n'
              f'outputs: {outputs}')

        gridvlp_classification_pipeline = GridVlpClassificationPipeline(
            'rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-brand'
        )
        input_params = {'text': self.text, 'image': self.image}
        inputs = gridvlp_classification_pipeline.preprocess(input_params)
        outputs = gridvlp_classification_pipeline.forward(inputs)
        print(f'text: {self.text}\nimage: {self.image}\n'
              f'outputs: {outputs}')

        gridvlp_classification_pipeline = GridVlpClassificationPipeline(
            'rgtjf1/multi-modal_gridvlp_classification_chinese-base-similarity'
        )
        input_params = {'text': self.text, 'image': self.image}
        inputs = gridvlp_classification_pipeline.preprocess(input_params)
        outputs = gridvlp_classification_pipeline.forward(inputs)
        print(f'text: {self.text}\nimage: {self.image}\n'
              f'outputs: {outputs}')

        gridvlp_embedding_pipeline = GridVlpEmbeddingPipeline(
            'rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-embedding'
        )
        input_params = {'text': self.text, 'image': self.image}
        inputs = gridvlp_embedding_pipeline.preprocess(input_params)
        outputs = gridvlp_embedding_pipeline.forward(inputs)
        print(f'text: {self.text}\nimage: {self.image}\n'
              f'outputs shape: {outputs.shape}')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
