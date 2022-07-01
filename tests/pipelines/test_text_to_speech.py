import unittest

import tensorflow as tf
# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch
from scipy.io.wavfile import write

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import build_preprocessor
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class TextToSpeechSambertHifigan16kPipelineTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_pipeline(self):
        lang_type = 'pinyin'
        text = '明天天气怎么样'
        preprocessor_model_id = 'damo/speech_binary_tts_frontend_resource'
        am_model_id = 'damo/speech_sambert16k_tts_zhitian_emo'
        voc_model_id = 'damo/speech_hifigan16k_tts_zhitian_emo'

        cfg_preprocessor = dict(
            type=Preprocessors.text_to_tacotron_symbols,
            model_name=preprocessor_model_id,
            lang_type=lang_type)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        am = Model.from_pretrained(am_model_id)
        self.assertTrue(am is not None)

        voc = Model.from_pretrained(voc_model_id)
        self.assertTrue(voc is not None)

        sambert_tts = pipeline(
            task=Tasks.text_to_speech,
            pipeline_name=Pipelines.sambert_hifigan_16k_tts,
            config_file='',
            model=[am, voc],
            preprocessor=preprocessor)
        self.assertTrue(sambert_tts is not None)

        output = sambert_tts(text)
        self.assertTrue(len(output['output']) > 0)
        write('output.wav', 16000, output['output'])


if __name__ == '__main__':
    unittest.main()
