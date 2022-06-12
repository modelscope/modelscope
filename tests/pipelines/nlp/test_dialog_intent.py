# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from tests.case.nlp.dialog_intent_case import test_case

from maas_lib.models.nlp import DialogIntentModel
from maas_lib.pipelines import DialogIntentPipeline, pipeline
from maas_lib.preprocessors import DialogIntentPreprocessor
from maas_lib.utils.constant import Tasks


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
            pipeline1(item)


if __name__ == '__main__':
    unittest.main()
