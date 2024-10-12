# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.preprocessors.templates import TemplateType
from modelscope.preprocessors.templates.loader import TemplateLoader
from modelscope.utils.test_utils import test_level


class TestToOllama(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_load_template(self):
        template = TemplateLoader.load_by_model_id(
            'LLM-Research/Meta-Llama-3-8B-Instruct')
        self.assertTrue(template.template_type == TemplateType.llama3)

        template = TemplateLoader.load_by_model_id(
            'swift/Meta-Llama-3-70B-Instruct-AWQ')
        self.assertTrue(template.template_type == TemplateType.llama3)

        template = TemplateLoader.load_by_model_id(
            'deepseek-ai/DeepSeek-V2-Lite-Chat')
        self.assertTrue(template.template_type == TemplateType.deepseek2)

        template = TemplateLoader.load_by_model_id('deepseek-ai/DeepSeek-V2.5')
        self.assertTrue(template.template_type == TemplateType.deepseek2_5)

        template = TemplateLoader.load_by_model_id(
            'deepseek-ai/deepseek-coder-1.3b-instruct')
        self.assertTrue(template.template_type == TemplateType.deepseek_coder)

        template = TemplateLoader.load_by_model_id(
            'OpenBuddy/openbuddy-deepseek-67b-v15.2')
        self.assertTrue(template is None)

        template = TemplateLoader.load_by_model_id(
            'deepseek-ai/deepseek-llm-67b-chat')
        self.assertTrue(template.template_type == TemplateType.deepseek)

        template = TemplateLoader.load_by_model_id(
            'deepseek-ai/DeepSeek-Coder-V2-Instruct')
        self.assertTrue(template.template_type == TemplateType.deepseek2)

        template = TemplateLoader.load_by_model_id('01ai/Yi-1.5-9B-Chat')
        self.assertTrue(template.template_type == TemplateType.chatml)

        template = TemplateLoader.load_by_model_id('01ai/Yi-Coder-9B-Chat')
        self.assertTrue(template.template_type == TemplateType.yi_coder)

        template = TemplateLoader.load_by_model_id(
            'LLM-Research/gemma-2-27b-it')
        self.assertTrue(template.template_type == TemplateType.gemma)

        template = TemplateLoader.load_by_model_id('AI-ModelScope/gemma-2b')
        self.assertTrue(template is None)

        template = TemplateLoader.load_by_model_id(
            'AI-ModelScope/gemma-2b-instruct')
        self.assertTrue(template is None)

        template = TemplateLoader.load_by_model_id(
            'AI-ModelScope/gemma2-2b-instruct')
        self.assertTrue(template.template_type == TemplateType.gemma)

        template = TemplateLoader.load_by_model_id(
            'AI-ModelScope/paligemma-3b-mix-224')
        self.assertTrue(template is None)

        template = TemplateLoader.load_by_model_id(
            'LLM-Research/Phi-3-vision-128k-instruct')
        self.assertTrue(template is None)

        template = TemplateLoader.load_by_model_id(
            'LLM-Research/Phi-3-128k-instruct')
        self.assertTrue(template.template_type == TemplateType.phi3)

        template = TemplateLoader.load_by_model_id(
            'LLM-Research/Phi-3-128k-instruct-GGUF')
        self.assertTrue(template.template_type == TemplateType.phi3)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_load_ollama(self):
        ollama = TemplateLoader.to_ollama(
            'LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            'QuantFactory/Gemma-2-Ataraxy-9B-Chat-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama('Xorbits/Llama-2-7b-Chat-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            'AI-ModelScope/gemma2-2b-instruct-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            'LLM-Research/Phi-3-128k-instruct-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(template_name='phi3')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            'QuantFactory/Mistral-Nemo-Japanese-Instruct-2408-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama('AI-ModelScope/Yi-1.5-9B-32K-GGUF')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            'AI-ModelScope/llava-llama-3-8b-v1_1-gguf')
        self.assertTrue(ollama is not None)
        ollama = TemplateLoader.to_ollama(
            '01ai/Yi-1.5-9B-Chat', ignore_oss_model_file=True)
        self.assertTrue(ollama is not None)


if __name__ == '__main__':
    unittest.main()
