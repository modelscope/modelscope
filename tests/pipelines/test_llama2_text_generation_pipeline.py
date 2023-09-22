# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Llama2TextGenerationPipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        self.llama2_model_id_7B_chat_ms = 'modelscope/Llama-2-7b-chat-ms'
        self.llama2_input_chat_ch = 'What are the company there?'
        self.history_demo = [(
            'Where is the capital of Zhejiang?',
            'Thank you for asking! The capital of Zhejiang Province is Hangzhou.'
        )]

    def run_pipeline_with_model_id(self,
                                   model_id,
                                   input,
                                   init_kwargs={},
                                   run_kwargs={}):
        pipeline_ins = pipeline(task=Tasks.chat, model=model_id, **init_kwargs)
        pipeline_ins._model_prepare = True
        result = pipeline_ins(input, **run_kwargs)
        print(result['response'])

    # 7B_ms_chat
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_llama2_7B_chat_ms_with_model_name_with_chat_ch_with_args(self):
        self.run_pipeline_with_model_id(
            self.llama2_model_id_7B_chat_ms,
            self.llama2_input_chat_ch,
            init_kwargs={
                'device_map': 'auto',
                'torch_dtype': torch.float16,
                'model_revision': 'v1.0.5',
                'ignore_file_pattern': [r'.+\.bin$']
            },
            run_kwargs={
                'max_length': 512,
                'do_sample': True,
                'top_p': 0.9,
                'history': self.history_demo
            })


if __name__ == '__main__':
    unittest.main()
