# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
import unittest

import numpy

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

MIX_SPEECH_FILE = 'data/test/audios/mix_speech.wav'


class SpeechSeparationTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_normal(self):
        import soundfile as sf
        model_id = 'damo/speech_mossformer_separation_temporal_8k'
        separation = pipeline(Tasks.speech_separation, model=model_id)
        result = separation(os.path.join(os.getcwd(), MIX_SPEECH_FILE))
        self.assertTrue(OutputKeys.OUTPUT_PCM_LIST in result)
        self.assertEqual(len(result[OutputKeys.OUTPUT_PCM_LIST]), 2)
        for i, signal in enumerate(result[OutputKeys.OUTPUT_PCM_LIST]):
            save_file = f'output_spk{i}.wav'
            sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16),
                     8000)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_mossformer2(self):
        import soundfile as sf
        model_id = 'damo/speech_mossformer2_separation_temporal_8k'
        separation = pipeline(Tasks.speech_separation, model=model_id)
        result = separation(os.path.join(os.getcwd(), MIX_SPEECH_FILE))
        self.assertTrue(OutputKeys.OUTPUT_PCM_LIST in result)
        self.assertEqual(len(result[OutputKeys.OUTPUT_PCM_LIST]), 2)
        for i, signal in enumerate(result[OutputKeys.OUTPUT_PCM_LIST]):
            save_file = f'output_spk{i}.wav'
            sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16),
                     8000)


if __name__ == '__main__':
    unittest.main()
