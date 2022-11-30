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

    def test_token_classification_tokenize_bert(self):
        cfg = dict(
            type='token-cls-tokenizer',
            padding=False,
            label_all_tokens=False,
            model_dir='bert-base-cased',
            label2id={
                'O': 0,
                'B': 1,
                'I': 2
            })
        preprocessor = build_preprocessor(cfg, Fields.nlp)
        input = 'Do not meddle in the affairs of wizards, ' \
                'for they are subtle and quick to anger.'
        output = preprocessor(input)
        self.assertTrue(InputFields.text in output)
        self.assertEqual(output['input_ids'].tolist()[0], [
            101, 2091, 1136, 1143, 13002, 1107, 1103, 5707, 1104, 16678, 1116,
            117, 1111, 1152, 1132, 11515, 1105, 3613, 1106, 4470, 119, 102
        ])
        self.assertEqual(
            output['attention_mask'].tolist()[0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(output['label_mask'].tolist()[0], [
            False, True, True, True, False, True, True, True, True, True,
            False, True, True, True, True, True, True, True, True, True, True,
            False
        ])
        self.assertEqual(
            output['offset_mapping'].tolist()[0],
            [[0, 2], [3, 6], [7, 13], [14, 16], [17, 20], [21, 28], [29, 31],
             [32, 39], [39, 40], [41, 44], [45, 49], [50, 53], [54, 60],
             [61, 64], [65, 70], [71, 73], [74, 79], [79, 80]])

    def test_token_classification_tokenize_roberta(self):
        cfg = dict(
            type='token-cls-tokenizer',
            padding=False,
            label_all_tokens=False,
            model_dir='xlm-roberta-base',
            label2id={
                'O': 0,
                'B': 1,
                'I': 2
            })
        preprocessor = build_preprocessor(cfg, Fields.nlp)
        input = 'Do not meddle in the affairs of wizards, ' \
                'for they are subtle and quick to anger.'
        output = preprocessor(input)
        self.assertTrue(InputFields.text in output)
        self.assertEqual(output['input_ids'].tolist()[0], [
            0, 984, 959, 128, 19298, 23, 70, 103086, 7, 111, 6, 44239, 99397,
            4, 100, 1836, 621, 1614, 17991, 136, 63773, 47, 348, 56, 5, 2
        ])
        self.assertEqual(output['attention_mask'].tolist()[0], [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1
        ])
        self.assertEqual(output['label_mask'].tolist()[0], [
            False, True, True, True, False, True, True, True, False, True,
            True, False, False, False, True, True, True, True, False, True,
            True, True, True, False, False, False
        ])
        self.assertEqual(
            output['offset_mapping'].tolist()[0],
            [[0, 2], [3, 6], [7, 13], [14, 16], [17, 20], [21, 28], [29, 31],
             [32, 40], [41, 44], [45, 49], [50, 53], [54, 60], [61, 64],
             [65, 70], [71, 73], [74, 80]])


if __name__ == '__main__':
    unittest.main()
