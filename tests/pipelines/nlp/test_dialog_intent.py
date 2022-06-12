# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from tests.case.nlp.dialog_intent_case import test_case

from modelscope.models.nlp import DialogIntentModel
from modelscope.pipelines import DialogIntentPipeline, pipeline
from modelscope.preprocessors import DialogIntentPreprocessor
from modelscope.utils.constant import Tasks


class DialogGenerationTest(unittest.TestCase):

    def test_run(self):

        modeldir = '/Users/yangliu/Desktop/space-dialog-intent'

        preprocessor = DialogIntentPreprocessor(model_dir=modeldir)
        model = DialogIntentModel(
            model_dir=modeldir,
            text_field=preprocessor.text_field,
            config=preprocessor.config)
        pipeline1 = DialogIntentPipeline(
            model=model, preprocessor=preprocessor)
        # pipeline1 = pipeline(task=Tasks.dialog_intent, model=model, preprocessor=preprocessor)

        for item in test_case:
            print(pipeline1(item))


if __name__ == '__main__':
    unittest.main()
