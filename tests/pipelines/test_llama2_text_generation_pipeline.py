# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Llama2TextGenerationPipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        self.llama2_model_id_7B_ms = 'modelscope/Llama-2-7b-ms'
        self.llama2_model_id_13B_ms = 'modelscope/Llama-2-13b-ms'
        self.llama2_model_id_7B_chat_ms = 'modelscope/Llama-2-7b-chat-ms'
        self.llama2_model_id_13B_chat_ms = 'modelscope/Llama-2-13b-chat-ms'

        self.llama2_input_chat_ch = '天空为什么是蓝色的？'
        self.llama2_input_gen_ch = '面朝大海，'
        self.llama2_input_chat_en = 'What is the difference between a tiger and a lion?'
        self.llama2_input_gen_en = 'I used to be a soldier,'

    def run_pipeline_with_model_id(self,
                                   model_id,
                                   input,
                                   init_kwargs={},
                                   run_kwargs={}):
        pipeline_ins = pipeline(
            task=Tasks.text_generation, model=model_id, **init_kwargs)
        pipeline_ins._model_prepare = True
        result = pipeline_ins(input, **run_kwargs)
        print(result['text'])

    # 7B_ms

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_chat_ch(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_chat_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_chat_en(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_chat_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_chat_en_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_chat_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_gen_ch(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_gen_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_gen_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_gen_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_gen_en(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_gen_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_ms_with_model_name_with_gen_en_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_ms,
            self.llama2_input_gen_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    # 13B_ms
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_chat_ch(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_chat_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_gen_ch(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_gen_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_gen_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_gen_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_chat_en(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_chat_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_chat_en_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_chat_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_gen_en(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_gen_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_13B_ms_with_model_name_with_gen_en_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_13B_ms,
            self.llama2_input_gen_en,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })

    # 7B_ms_chat
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_chat_ms_with_model_name_with_chat_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_chat_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16
            },
            run_kwargs={
                'max_length': 200,
                'do_sample': True,
                'top_p': 0.85
            })


if __name__ == '__main__':
    unittest.main()
