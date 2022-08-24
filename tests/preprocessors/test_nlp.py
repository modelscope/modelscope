# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.preprocessors import build_preprocessor, nlp
from modelscope.utils.constant import Fields, InputFields
from modelscope.utils.logger import get_logger

logger = get_logger()


class NLPPreprocessorTest(unittest.TestCase):

    def test_tokenize(self):
        cfg = dict(type='Tokenize', tokenizer_name='bert-base-cased')
        preprocessor = build_preprocessor(cfg, Fields.nlp)
        input = {
            InputFields.text:
            'Do not meddle in the affairs of wizards, '
            'for they are subtle and quick to anger.'
        }
        output = preprocessor(input)
        self.assertTrue(InputFields.text in output)
        self.assertEqual(output['input_ids'], [
            101, 2091, 1136, 1143, 13002, 1107, 1103, 5707, 1104, 16678, 1116,
            117, 1111, 1152, 1132, 11515, 1105, 3613, 1106, 4470, 119, 102
        ])
        self.assertEqual(
            output['token_type_ids'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(
            output['attention_mask'],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
