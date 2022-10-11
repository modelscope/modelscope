# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch
from scipy.io.wavfile import write

from modelscope.models import Model
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
        zhcn_text = '今天北京天气怎么样'
        en_text = 'How is the weather in Beijing?'
        zhcn_voice = ['zhitian_emo', 'zhizhe_emo', 'zhiyan_emo', 'zhibei_emo']
        enus_voice = ['andy', 'annie']
        engb_voice = ['luca', 'luna']
        self.tts_test_cases = []
        for voice in zhcn_voice:
            model_id = 'damo/speech_sambert-hifigan_tts_%s_%s_16k' % (voice,
                                                                      'zh-cn')
            self.tts_test_cases.append({
                'voice': voice,
                'model_id': model_id,
                'text': zhcn_text
            })
        for voice in enus_voice:
            model_id = 'damo/speech_sambert-hifigan_tts_%s_%s_16k' % (voice,
                                                                      'en-us')
            self.tts_test_cases.append({
                'voice': voice,
                'model_id': model_id,
                'text': en_text
            })
        for voice in engb_voice:
            model_id = 'damo/speech_sambert-hifigan_tts_%s_%s_16k' % (voice,
                                                                      'en-gb')
            self.tts_test_cases.append({
                'voice': voice,
                'model_id': model_id,
                'text': en_text
            })
        zhcn_model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
        enus_model_id = 'damo/speech_sambert-hifigan_tts_en-us_16k'
        engb_model_id = 'damo/speech_sambert-hifigan_tts_en-gb_16k'
        self.tts_test_cases.append({
            'voice': 'zhcn',
            'model_id': zhcn_model_id,
            'text': zhcn_text
        })
        self.tts_test_cases.append({
            'voice': 'enus',
            'model_id': enus_model_id,
            'text': en_text
        })
        self.tts_test_cases.append({
            'voice': 'engb',
            'model_id': engb_model_id,
            'text': en_text
        })

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pipeline(self):
        for case in self.tts_test_cases:
            logger.info('test %s' % case['voice'])
            model = Model.from_pretrained(
                model_name_or_path=case['model_id'], revision='pytorch_am')
            sambert_hifigan_tts = pipeline(task=self.task, model=model)
            self.assertTrue(sambert_hifigan_tts is not None)
            output = sambert_hifigan_tts(input=case['text'])
            self.assertIsNotNone(output[OutputKeys.OUTPUT_PCM])
            pcm = output[OutputKeys.OUTPUT_PCM]
            write('output_%s.wav' % case['voice'], 16000, pcm)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
