import unittest

# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch
from scipy.io.wavfile import write

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

import tensorflow as tf  # isort:skip

logger = get_logger()


class TextToSpeechSambertHifigan16kPipelineTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_pipeline(self):
        single_test_case_label = 'test_case_label_0'
        text = '今天北京天气怎么样？'
        model_id = 'damo/speech_sambert-hifigan_tts_zhcn_16k'
        voice = 'zhitian_emo'

        sambert_hifigan_tts = pipeline(
            task=Tasks.text_to_speech, model=model_id)
        self.assertTrue(sambert_hifigan_tts is not None)
        inputs = {single_test_case_label: text, 'voice': voice}
        output = sambert_hifigan_tts(inputs)
        self.assertIsNotNone(output[OutputKeys.OUTPUT_PCM])
        pcm = output[OutputKeys.OUTPUT_PCM][single_test_case_label]
        write('output.wav', 16000, pcm)


if __name__ == '__main__':
    unittest.main()
