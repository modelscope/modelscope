import time
import unittest

import json
import tensorflow as tf
# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch
from scipy.io.wavfile import write

from modelscope.fileio import File
from modelscope.models import Model, build_model
from modelscope.models.audio.tts.am import SambertNetHifi16k
from modelscope.models.audio.tts.vocoder import AttrDict, Hifigan16k
from modelscope.pipelines import pipeline
from modelscope.preprocessors import build_preprocessor
from modelscope.utils.constant import Fields, InputFields, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class TextToSpeechSambertHifigan16kPipelineTest(unittest.TestCase):

    def test_pipeline(self):
        lang_type = 'pinyin'
        text = '明天天气怎么样'
        preprocessor_model_id = 'damo/speech_binary_tts_frontend_resource'
        am_model_id = 'damo/speech_sambert16k_tts_zhitian_emo'
        voc_model_id = 'damo/speech_hifigan16k_tts_zhitian_emo'

        cfg_preprocessor = dict(
            type='text_to_tacotron_symbols',
            model_name=preprocessor_model_id,
            lang_type=lang_type)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        am = Model.from_pretrained(am_model_id)
        self.assertTrue(am is not None)

        voc = Model.from_pretrained(voc_model_id)
        self.assertTrue(voc is not None)

        sambert_tts = pipeline(
            pipeline_name='tts-sambert-hifigan-16k',
            config_file='',
            model=[am, voc],
            preprocessor=preprocessor)
        self.assertTrue(sambert_tts is not None)

        output = sambert_tts(text)
        self.assertTrue(len(output['output']) > 0)
        write('output.wav', 16000, output['output'])


if __name__ == '__main__':
    unittest.main()
