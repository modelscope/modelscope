# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope import (AutoConfig, AutoModel, AutoModelForCausalLM,
                        AutoTokenizer, GenerationConfig)


class HFUtilTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_auto_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            'baichuan-inc/Baichuan2-7B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(tokenizer.vocab_size, 125696)
        self.assertEqual(tokenizer.model_max_length, 4096)
        self.assertFalse(tokenizer.is_fast)

    def test_quantization_import(self):
        from modelscope import GPTQConfig, BitsAndBytesConfig
        self.assertTrue(BitsAndBytesConfig is not None)

    def test_auto_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            'baichuan-inc/baichuan-7B', trust_remote_code=True)
        self.assertTrue(model is not None)

    def test_auto_config(self):
        config = AutoConfig.from_pretrained(
            'baichuan-inc/Baichuan-13B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(config.model_type, 'baichuan')
        gen_config = GenerationConfig.from_pretrained(
            'baichuan-inc/Baichuan-13B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(gen_config.assistant_token_id, 196)

    def test_transformer_patch(self):
        tokenizer = AutoTokenizer.from_pretrained(
            'skyline2006/llama-7b', revision='v1.0.1')
        self.assertIsNotNone(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            'skyline2006/llama-7b', revision='v1.0.1')
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
