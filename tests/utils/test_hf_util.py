# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.utils.hf_util.patcher import patch_context


class HFUtilTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_auto_tokenizer(self):
        from modelscope import AutoTokenizer
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
        from modelscope import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            'baichuan-inc/baichuan-7B', trust_remote_code=True)
        self.assertTrue(model is not None)

    def test_auto_config(self):
        from modelscope import AutoConfig, GenerationConfig
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
        with patch_context():
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-base')
            self.assertIsNotNone(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-base')
            self.assertIsNotNone(model)

    def test_patch_model(self):
        from modelscope.utils.hf_util.patcher import patch_context
        with patch_context():
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(model is not None)
        try:
            model = AutoModel.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_config(self):
        with patch_context():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(config is not None)
        try:
            config = AutoConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_diffusers(self):
        with patch_context():
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                'AI-ModelScope/stable-diffusion-v1-5')
            self.assertTrue(pipe is not None)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                'AI-ModelScope/stable-diffusion-v1-5')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_peft(self):
        with patch_context():
            from peft import PeftModel
            self.assertTrue(hasattr(PeftModel, '_from_pretrained_origin'))
        self.assertFalse(hasattr(PeftModel, '_from_pretrained_origin'))


if __name__ == '__main__':
    unittest.main()
