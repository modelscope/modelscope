# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from tests.case.nlp.dialog_intent_case import test_case

from maas_lib.preprocessors import DialogIntentPreprocessor
from maas_lib.utils.constant import Fields, InputFields
from maas_lib.utils.logger import get_logger

logger = get_logger()


class DialogGenerationPreprocessorTest(unittest.TestCase):

    def test_tokenize(self):
        modeldir = '/Users/yangliu/Desktop/space-dialog-intent'
        processor = DialogIntentPreprocessor(model_dir=modeldir)

        for item in test_case:
            print(item)
            print(processor(item))


if __name__ == '__main__':
    unittest.main()
