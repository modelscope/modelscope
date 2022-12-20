# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch
from scipy.io.wavfile import write

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

import tensorflow as tf  # isort:skip

logger = get_logger()


class TextToSpeechSambertHifigan16kPipelineTest(unittest.TestCase,
                                                DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_to_speech
        self.zhcn_text = '今天北京天气怎么样'
        self.en_text = 'How is the weather in Beijing?'
        self.test_model_name = [
            'pretrain_16k', 'pretrain_24k', 'zhitian_emo', 'zhizhe_emo',
            'zhiyan_emo', 'zhibei_emo', 'zhcn_16k', 'luca', 'luna', 'andy',
            'annie', 'engb_16k', 'enus_16k'
        ]
        self.test_models = [{
            'model':
            'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k',
            'text': self.zhcn_text
        }, {
            'model':
            'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_24k',
            'text': self.zhcn_text,
            'sample_rate': 24000
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k',
            'text': self.zhcn_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_zhizhe_emo_zh-cn_16k',
            'text': self.zhcn_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k',
            'text': self.zhcn_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_zhibei_emo_zh-cn_16k',
            'text': self.zhcn_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_zh-cn_16k',
            'text': self.zhcn_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_luca_en-gb_16k',
            'text': self.en_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_luna_en-gb_16k',
            'text': self.en_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_andy_en-us_16k',
            'text': self.en_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_annie_en-us_16k',
            'text': self.en_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_en-gb_16k',
            'text': self.en_text
        }, {
            'model': 'damo/speech_sambert-hifigan_tts_en-us_16k',
            'text': self.en_text
        }]

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pipeline(self):
        for i in range(len(self.test_models)):
            logger.info('test %s' % self.test_model_name[i])
            sambert_hifigan_tts = pipeline(
                task=self.task, model=self.test_models[i]['model'])
            self.assertTrue(sambert_hifigan_tts is not None)
            output = sambert_hifigan_tts(input=self.test_models[i]['text'])
            self.assertIsNotNone(output[OutputKeys.OUTPUT_PCM])
            pcm = output[OutputKeys.OUTPUT_PCM]
            sr = 16000
            if 'sample_rate' in self.test_models[i]:
                sr = self.test_models[i]['sample_rate']
            write('output_%s.wav' % self.test_model_name[i], sr, pcm)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
