# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.preprocessors.templates import TemplateType
from modelscope.preprocessors.templates.loader import TemplateLoader
from modelscope.utils.test_utils import test_level


def _test_check_tmpl_type(model, tmpl_type):
    ollama, info = TemplateLoader.to_ollama(model, debug=True)
    assert info.__dict__.get('modelfile_prefix').split(
        '/')[-1] == tmpl_type, info


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
            'deepseek-ai/DeepSeek-V2-Lite-Chat')
        self.assertTrue(template.template_type == TemplateType.deepseek2)

        template = TemplateLoader.load_by_model_id('01ai/Yi-1.5-9B-Chat')
        self.assertTrue(template.template_type == TemplateType.chatml)

        template = TemplateLoader.load_by_model_id('01ai/Yi-Coder-9B-Chat')
        self.assertTrue(template.template_type == TemplateType.yi_coder)

        template = TemplateLoader.load_by_model_id(
            'LLM-Research/gemma-2-27b-it')
        self.assertTrue(template.template_type == TemplateType.gemma)

        template = TemplateLoader.load_by_model_id('AI-ModelScope/gemma-2b')
        self.assertTrue(template.template_type == TemplateType.gemma)

        template = TemplateLoader.load_by_model_id(
            'AI-ModelScope/gemma-2b-instruct')
        self.assertTrue(template.template_type == TemplateType.gemma)

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
        ollama = TemplateLoader.to_ollama(
            'QuantFactory/Mistral-7B-Instruct-v0.1-GGUF',
            ignore_oss_model_file=True)
        self.assertTrue(ollama is not None)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_check_template_type(self):
        _test_check_tmpl_type(
            'AI-ModelScope/Llama-3.2-11B-Vision-Instruct-GGUF',
            'llama3.2-vision')
        _test_check_tmpl_type('LLM-Research/Meta-Llama-3.2-8B-Instruct-GGUF',
                              'llama3.2')
        _test_check_tmpl_type('LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF',
                              'llama3.1')
        _test_check_tmpl_type('LLM-Research/Meta-Llama-3-8B-Instruct-GGUF',
                              'llama3')
        _test_check_tmpl_type(
            'LLM-Research/Llama-3-8B-Instruct-Gradient-4194k-GGUF',
            'llama3-gradient')
        _test_check_tmpl_type('QuantFactory/Llama-3-Groq-8B-Tool-Use-GGUF',
                              'llama3-groq-tool-use')
        _test_check_tmpl_type('QuantFactory/Llama3-ChatQA-1.5-8B-GGUF',
                              'llama3-chatqa')
        _test_check_tmpl_type('SinpxAI/Llama2-Chinese-7B-Chat-GGUF',
                              'llama2-chinese')
        _test_check_tmpl_type('QuantFactory/dolphin-2.9-llama3-70b-GGUF',
                              'dolphin-llama3')
        _test_check_tmpl_type('AI-ModelScope/llava-llama-3-8b-v1_1-gguf',
                              'llava-llama3')
        _test_check_tmpl_type('Xorbits/Llama-2-7b-Chat-GGUF', 'llama2')
        _test_check_tmpl_type('QuantFactory/MathCoder2-CodeLlama-7B-GGUF',
                              'codellama')
        _test_check_tmpl_type('QuantFactory/TinyLlama-1.1B-Chat-v1.0-GGUF',
                              'tinyllama')
        _test_check_tmpl_type('AI-ModelScope/LLaMA-Pro-8B-Instruct',
                              'llama-pro')
        _test_check_tmpl_type('LLM-Research/Llama-Guard-3-8B', 'llama-guard3')
        _test_check_tmpl_type('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5')
        _test_check_tmpl_type('Xorbits/Qwen-14B-Chat-GGUF', 'qwen')
        _test_check_tmpl_type('QuantFactory/Qwen2-7B-GGUF', 'qwen2')
        _test_check_tmpl_type('QuantFactory/Qwen2-Math-7B-GGUF', 'qwen2-math')
        _test_check_tmpl_type('Qwen/CodeQwen1.5-7B-Chat-GGUF', 'codeqwen')
        _test_check_tmpl_type('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
                              'qwen2.5-coder')
        _test_check_tmpl_type('QuantFactory/Gemma-2-Ataraxy-9B-Chat-GGUF',
                              'gemma2')
        _test_check_tmpl_type(
            'QuantFactory/Athene-codegemma-2-7b-it-alpaca-v1.1-GGUF',
            'codegemma')
        _test_check_tmpl_type('QuantFactory/gemma-7b-GGUF', 'gemma')
        _test_check_tmpl_type('QuantFactory/shieldgemma-2b-GGUF',
                              'shieldgemma')
        _test_check_tmpl_type(
            'ZhaoningLi/laser-dolphin-mixtral-2x7b-dpo.fp16.gguf',
            'dolphin-mixtral')
        _test_check_tmpl_type('QuantFactory/dolphin-2.1-mistral-7b-GGUF',
                              'dolphin-mistral')
        _test_check_tmpl_type('xtuner/llava-phi-3-mini', 'llava-phi3')
        _test_check_tmpl_type('QuantFactory/Phi-3.5-mini-instruct-GGUF',
                              'phi3.5')
        _test_check_tmpl_type('AI-ModelScope/Phi-3-medium-128k-instruct-GGUF',
                              'phi3')
        _test_check_tmpl_type('QuantFactory/phi-2-GGUF', 'phi')
        _test_check_tmpl_type('alignmentforever/alpaca-Yarn-Mistral-7b-128k',
                              'yarn-mistral')
        _test_check_tmpl_type('LLM-Research/Mistral-Large-Instruct-2407',
                              'mistral-large')
        _test_check_tmpl_type('AI-ModelScope/MistralLite', 'mistrallite')
        _test_check_tmpl_type('AI-ModelScope/Mistral-Small-Instruct-2409',
                              'mistral-small')
        _test_check_tmpl_type('LLM-Research/Mistral-Nemo-Instruct-2407-GGUF',
                              'mistral-nemo')
        _test_check_tmpl_type('QuantFactory/Mistral-7B-OpenOrca-GGUF',
                              'mistral-openorca')
        _test_check_tmpl_type('QuantFactory/Mistral-7B-Instruct-v0.1-GGUF',
                              'mistral')
        _test_check_tmpl_type(
            'second-state/Nous-Hermes-2-Mixtral-8x7B-SFT-GGUF',
            'nous-hermes2-mixtral')
        _test_check_tmpl_type('AI-ModelScope/Mixtral-8x22B-v0.1-GGUF',
                              'mixtral')
        _test_check_tmpl_type('QuantFactory/Nemotron-Mini-4B-Instruct-GGUF',
                              'nemotron-mini')
        _test_check_tmpl_type('AI-ModelScope/Nemotron-4-340B-Instruct',
                              'nemotron')
        _test_check_tmpl_type('TIGER-Lab/Mantis-bakllava-7b', 'bakllava')
        _test_check_tmpl_type('fireicewolf/llava-v1.6-34B-gguf', 'llava')
        _test_check_tmpl_type(
            'AI-ModelScope/DeepSeek-Coder-V2-Lite-Instruct-GGUF',
            'deepseek-coder-v2')
        _test_check_tmpl_type('QuantFactory/deepseek-coder-6.7B-kexer-GGUF',
                              'deepseek-coder')
        _test_check_tmpl_type('deepseek-ai/DeepSeek-V2.5', 'deepseek-v2.5')
        _test_check_tmpl_type('deepseek-ai/DeepSeek-V2-Lite-Chat',
                              'deepseek-v2')
        _test_check_tmpl_type('deepseek-ai/deepseek-llm-67b-chat',
                              'deepseek-llm')
        _test_check_tmpl_type('LLM-Research/glm-4-9b-chat-GGUF', 'glm4')
        _test_check_tmpl_type('AI-ModelScope/Yi-Coder-9B-Chat-GGUF',
                              'yi-coder')
        _test_check_tmpl_type('01ai/Yi-1.5-9B-Chat', 'yi')
        _test_check_tmpl_type('AI-ModelScope/c4ai-command-r-plus',
                              'command-r-plus')
        _test_check_tmpl_type('AI-ModelScope/c4ai-command-r-v01', 'command-r')
        _test_check_tmpl_type('LLM-Research/codegeex4-all-9b-GGUF',
                              'codegeex4')
        _test_check_tmpl_type('a7823093/Wizard-Vicuna-13B-Uncensored-HF',
                              'wizard-vicuna-uncensored')
        _test_check_tmpl_type('AI-ModelScope/WizardLM-2-8x22B-GGUF',
                              'wizardlm2')
        _test_check_tmpl_type('AI-ModelScope/WizardCoder-Python-34B-V1.0',
                              'wizardcoder')
        _test_check_tmpl_type('AI-ModelScope/WizardMath-7B-V1.0',
                              'wizard-math')
        _test_check_tmpl_type('AI-ModelScope/WizardLM-7B-V1.0', 'wizardlm')
        _test_check_tmpl_type('QuantFactory/vicuna-13b-v1.5-GGUF', 'vicuna')
        _test_check_tmpl_type('QuantFactory/Nous-Hermes-2-SOLAR-10.7B-GGUF',
                              'nous-hermes2')
        _test_check_tmpl_type('QuantFactory/stable-code-instruct-3b-GGUF',
                              'stable-code')
        _test_check_tmpl_type('AI-ModelScope/stablelm-tuned-alpha-7b',
                              'stablelm2')
        _test_check_tmpl_type('QuantFactory/internlm2-chat-7b-GGUF',
                              'internlm2')
        _test_check_tmpl_type('openbmb/MiniCPM-V-2-gguf', 'minicpm-v')
        _test_check_tmpl_type('QuantFactory/Codestral-22B-v0.1-GGUF',
                              'codestral')
        _test_check_tmpl_type('AI-ModelScope/nomic-embed-text-v1',
                              'nomic-embed-text')
        _test_check_tmpl_type('AI-ModelScope/mxbai-embed-large-v1',
                              'mxbai-embed-large')
        _test_check_tmpl_type('AI-ModelScope/starcoder2-7b', 'starcoder2')
        _test_check_tmpl_type('QwenCollection/orca_mini_v7_72b', 'orca-mini')
        _test_check_tmpl_type('modelscope/zephyr-7b-beta', 'zephyr')
        _test_check_tmpl_type('LLM-Research/snowflake-arctic-embed-m',
                              'snowflake-arctic-embed')
        _test_check_tmpl_type('TabbyML/StarCoder-1B', 'starcoder')
        _test_check_tmpl_type('QuantFactory/granite-8b-code-instruct-4k-GGUF',
                              'granite-code')
        _test_check_tmpl_type('AI-ModelScope/all-MiniLM-L6-v2', 'all-minilm')
        _test_check_tmpl_type('QuantFactory/openchat-3.6-8b-20240522-GGUF',
                              'openchat')
        _test_check_tmpl_type('AI-ModelScope/aya-101', 'aya')
        _test_check_tmpl_type('LLM-Research/OpenHermes-2.5-Mistral-7B',
                              'openhermes')
        _test_check_tmpl_type('AI-ModelScope/Reflection-Llama-3.1-70B',
                              'reflection')
        _test_check_tmpl_type('AI-ModelScope/neural-chat-7b-v3-1',
                              'neural-chat')
        _test_check_tmpl_type('AI-ModelScope/moondream1', 'moondream')
        _test_check_tmpl_type('AI-ModelScope/Xwin-LM-70B-V0.1', 'xwinlm')
        _test_check_tmpl_type(
            'QuantFactory/smollm-360M-instruct-add-basics-GGUF', 'smollm')
        _test_check_tmpl_type('AI-ModelScope/sqlcoder-7b-2', 'sqlcoder')
        _test_check_tmpl_type('LLM-Research/Starling-LM-7B-beta',
                              'starling-lm')
        _test_check_tmpl_type('AI-ModelScope/falcon-7b', 'falcon')
        _test_check_tmpl_type('QuantFactory/SOLAR-10.7B-v1.0-GGUF', 'solar')
        _test_check_tmpl_type('AI-ModelScope/Orca-2-13b', 'orca2')
        _test_check_tmpl_type('AI-ModelScope/Hermes-3-Llama-3.1-8B', 'hermes3')
        _test_check_tmpl_type('QuantFactory/meditron-7b-GGUF', 'meditron')
        _test_check_tmpl_type('QuantFactory/NexusRaven-V2-13B-GGUF',
                              'nexusraven')
        _test_check_tmpl_type('davideuler/Magicoder-s-DS-6.7B-GGUF',
                              'magicoder')
        _test_check_tmpl_type('ZhejiangLab-LifeScience/falcon-assistant-2',
                              'falcon2')
        _test_check_tmpl_type('Xorbits/bge-m3', 'bge-m3')
        _test_check_tmpl_type('AI-ModelScope/notux-8x7b-v1', 'notux')
        _test_check_tmpl_type('AI-ModelScope/OpenOrca-Platypus2-13B',
                              'open-orca-platypus2')
        _test_check_tmpl_type('QuantFactory/notus-7b-v1-GGUF', 'notus')
        _test_check_tmpl_type('AI-ModelScope/mathstral-7B-v0.1', 'mathstral')
        _test_check_tmpl_type('AI-ModelScope/solar-pro-preview-instruct',
                              'solar-pro')
        _test_check_tmpl_type('AI-ModelScope/dbrx-instruct', 'dbrx')
        _test_check_tmpl_type('QuantFactory/NuExtract-GGUF', 'nuextract')
        _test_check_tmpl_type('QuantFactory/reader-lm-1.5b-GGUF', 'reader-lm')
        _test_check_tmpl_type(
            'SDXL-LoRA/KappaNeuro-alfred-augustus-glendening-style', 'alfred')
        _test_check_tmpl_type('AI-ModelScope/bge-large-zh-v1.5', 'bge-large')
        _test_check_tmpl_type(
            'Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2',
            'paraphrase-multilingual')
        _test_check_tmpl_type('bartowski/Marco-o1-GGUF', 'marco-o1')


if __name__ == '__main__':
    unittest.main()
