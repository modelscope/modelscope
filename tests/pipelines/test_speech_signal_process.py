import os.path
import unittest

from modelscope.metainfo import Pipelines
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level

NEAREND_MIC_FILE = 'data/test/audios/nearend_mic.wav'
FAREND_SPEECH_FILE = 'data/test/audios/farend_speech.wav'
NEAREND_MIC_URL = 'https://modelscope.cn/api/v1/models/damo/' \
                  'speech_dfsmn_aec_psm_16k/repo?Revision=master' \
                  '&FilePath=examples/nearend_mic.wav'
FAREND_SPEECH_URL = 'https://modelscope.cn/api/v1/models/damo/' \
                    'speech_dfsmn_aec_psm_16k/repo?Revision=master' \
                    '&FilePath=examples/farend_speech.wav'

NOISE_SPEECH_FILE = 'data/test/audios/speech_with_noise.wav'
NOISE_SPEECH_URL = 'https://modelscope.cn/api/v1/models/damo/' \
                   'speech_frcrn_ans_cirm_16k/repo?Revision=master' \
                   '&FilePath=examples/speech_with_noise.wav'


class SpeechSignalProcessTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        pass

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_aec(self):
        model_id = 'damo/speech_dfsmn_aec_psm_16k'
        input = {
            'nearend_mic': os.path.join(os.getcwd(), NEAREND_MIC_FILE),
            'farend_speech': os.path.join(os.getcwd(), FAREND_SPEECH_FILE)
        }
        aec = pipeline(Tasks.acoustic_echo_cancellation, model=model_id)
        output_path = os.path.abspath('output.wav')
        aec(input, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_aec_url(self):
        model_id = 'damo/speech_dfsmn_aec_psm_16k'
        input = {
            'nearend_mic': NEAREND_MIC_URL,
            'farend_speech': FAREND_SPEECH_URL
        }
        aec = pipeline(Tasks.acoustic_echo_cancellation, model=model_id)
        output_path = os.path.abspath('output.wav')
        aec(input, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_aec_bytes(self):
        model_id = 'damo/speech_dfsmn_aec_psm_16k'
        input = {}
        with open(os.path.join(os.getcwd(), NEAREND_MIC_FILE), 'rb') as f:
            input['nearend_mic'] = f.read()
        with open(os.path.join(os.getcwd(), FAREND_SPEECH_FILE), 'rb') as f:
            input['farend_speech'] = f.read()
        aec = pipeline(
            Tasks.acoustic_echo_cancellation,
            model=model_id,
            pipeline_name=Pipelines.speech_dfsmn_aec_psm_16k)
        output_path = os.path.abspath('output.wav')
        aec(input, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_aec_tuple_bytes(self):
        model_id = 'damo/speech_dfsmn_aec_psm_16k'
        with open(os.path.join(os.getcwd(), NEAREND_MIC_FILE), 'rb') as f:
            nearend_bytes = f.read()
        with open(os.path.join(os.getcwd(), FAREND_SPEECH_FILE), 'rb') as f:
            farend_bytes = f.read()
        inputs = (nearend_bytes, farend_bytes)
        aec = pipeline(
            Tasks.acoustic_echo_cancellation,
            model=model_id,
            pipeline_name=Pipelines.speech_dfsmn_aec_psm_16k)
        output_path = os.path.abspath('output.wav')
        aec(inputs, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ans(self):
        model_id = 'damo/speech_frcrn_ans_cirm_16k'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        ans(os.path.join(os.getcwd(), NOISE_SPEECH_FILE),
            output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_ans_url(self):
        model_id = 'damo/speech_frcrn_ans_cirm_16k'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        ans(NOISE_SPEECH_URL, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ans_bytes(self):
        model_id = 'damo/speech_frcrn_ans_cirm_16k'
        ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model=model_id,
            pipeline_name=Pipelines.speech_frcrn_ans_cirm_16k)
        output_path = os.path.abspath('output.wav')
        with open(os.path.join(os.getcwd(), NOISE_SPEECH_FILE), 'rb') as f:
            data = f.read()
            ans(data, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
