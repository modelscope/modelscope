# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from unittest.mock import patch

from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.nlp.token_classification_pipeline import \
    TokenClassificationPipeline


class _ModelStub(Model):

    def __init__(self):
        super().__init__('', device='cpu')

    def forward(self, *args, **kwargs):
        return {}

    def eval(self):
        pass


class TokenClassificationPipelineTest(unittest.TestCase):

    def test_forwards_trust_remote_code_to_base_pipeline(self):
        received = []

        def fake_base_init(pipeline, **kwargs):
            received.append(kwargs.get('trust_remote_code', False))
            pipeline.model = _ModelStub()
            pipeline.preprocessor = type('PreprocessorStub', (),
                                         {'id2label': {}})()

        with patch.object(Pipeline, '__init__', fake_base_init):
            for trust_remote_code in (True, False, None):
                kwargs = ({} if trust_remote_code is None else {
                    'trust_remote_code': trust_remote_code
                })
                with self.subTest(kwargs=kwargs):
                    TokenClassificationPipeline(
                        model='model-id', preprocessor=object(), **kwargs)

        self.assertEqual(received, [True, False, False])


if __name__ == '__main__':
    unittest.main()
