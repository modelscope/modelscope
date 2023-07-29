# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.polylm_13b_model_id = 'damo/nlp_polylm_13b_text_generation'
        self.polylm_multialpaca_13b_model_id = 'damo/nlp_polylm_multialpaca_13b_text_generation'
        self.polylm_multialpaca_ecomm_13b_model_id = 'damo/nlp_polylm_multialpaca_13b_text_generation_ecomm'
        self.input_text = 'Beijing is the capital of China.\nTranslate this sentence from English to Chinese.'
        self.ecomm_input_text = \
            'Below is an instruction that describes the attribute value detection task. ' \
            + 'Write a response that appropriately completes the request.\n' \
            + '### Instruction:\n暗格格纹纹路搭配磨砂表面\n' \
            + 'Extract all attribute value with attribute name about 鞋跟高度, 下摆类型, 工艺, 裙长, 腰型, 图案, 开衩类型, 风格, 领型, 版型, ' \
            + '鞋帮高度, 裤长, 裤型, 适用季节, 厚度, 弹性, 形状, 开口深度, 靴筒高度, 颜色, 闭合方式, 材质, 袖长, 鞋头款式, 袖型, 口袋类型 in the sentence. \n' \
            + '### Response:'

    @unittest.skip('oom error for 13b model')
    def test_polylm_13b_with_model_name(self):
        kwargs = {
            'do_sample': False,
            'num_beams': 4,
            'max_new_tokens': 128,
            'early_stopping': True,
            'eos_token_id': 2
        }
        pipeline_ins = pipeline(
            Tasks.text_generation, model=self.polylm_13b_model_id)
        result = pipeline_ins(self.input_text, **kwargs)
        print(result['text'])

    @unittest.skip('oom error for 13b model')
    def test_polylm_multialpaca_13b_with_model_name(self):
        kwargs = {
            'do_sample': True,
            'top_p': 0.8,
            'temperature': 0.7,
            'repetition_penalty': 1.02,
            'max_new_tokens': 128,
            'num_return_sequences': 1,
            'early_stopping': True,
            'eos_token_id': 2
        }
        pipeline_ins = pipeline(
            Tasks.text_generation, model=self.polylm_multialpaca_13b_model_id)

        input_text = f'{self.input_text}\n\n'
        result = pipeline_ins(input_text, **kwargs)
        print(result['text'])

    @unittest.skip('oom error for 13b model')
    def test_polylm_multialpaca_ecomm_13b_with_model_name(self):
        kwargs = {
            'do_sample': True,
            'top_p': 0.9,
            'temperature': 1.0,
            'repetition_penalty': 1.02,
            'max_new_tokens': 128,
            'num_return_sequences': 1,
            'early_stopping': True,
            'eos_token_id': 2
        }
        pipeline_ins = pipeline(
            Tasks.text_generation,
            model=self.polylm_multialpaca_ecomm_13b_model_id)
        input_text = f'{self.ecomm_input_text}\n\n'
        result = pipeline_ins(input_text, **kwargs)
        print(result['text'])

    @unittest.skip('oom error for 13b model')
    def test_run_with_default_model(self):
        kwargs = {
            'do_sample': False,
            'num_beams': 4,
            'max_new_tokens': 128,
            'early_stopping': True,
            'eos_token_id': 2
        }
        pipeline_ins = pipeline(
            Tasks.text_generation, model=self.polylm_13b_model_id)
        result = pipeline_ins(self.input_text, **kwargs)
        print(result['text'])


if __name__ == '__main__':
    unittest.main()
