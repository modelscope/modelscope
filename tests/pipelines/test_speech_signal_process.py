# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
import unittest

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

NEAREND_MIC_FILE = 'data/test/audios/nearend_mic.wav'
FAREND_SPEECH_FILE = 'data/test/audios/farend_speech.wav'
NEAREND_MIC_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/' \
                  'test/audios/nearend_mic.wav'
FAREND_SPEECH_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/' \
                    'test/audios/farend_speech.wav'

NOISE_SPEECH_FILE = 'data/test/audios/speech_with_noise.wav'
NOISE_SPEECH_FILE_48K = 'data/test/audios/speech_with_noise_48k.wav'
NOISE_SPEECH_FILE_48K_PCM = 'data/test/audios/speech_with_noise_48k.PCM'
NOISE_SPEECH_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/' \
                   'test/audios/speech_with_noise.wav'


@unittest.skip('For librosa numpy compatible')
class SpeechSignalProcessTest(unittest.TestCase):

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
    def test_frcrn_ans(self):
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

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dfsmn_ans(self):
        model_id = 'damo/speech_dfsmn_ans_psm_48k_causal'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        ans(os.path.join(os.getcwd(), NOISE_SPEECH_FILE_48K),
            output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_dfsmn_ans_bytes(self):
        model_id = 'damo/speech_dfsmn_ans_psm_48k_causal'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        with open(os.path.join(os.getcwd(), NOISE_SPEECH_FILE_48K), 'rb') as f:
            data = f.read()
            ans(data, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_dfsmn_ans_stream(self):
        model_id = 'damo/speech_dfsmn_ans_psm_48k_causal'
        ans = pipeline(
            Tasks.acoustic_noise_suppression, model=model_id, stream_mode=True)
        with open(os.path.join(os.getcwd(), NOISE_SPEECH_FILE_48K_PCM),
                  'rb') as f:
            block_size = 3840
            audio = f.read(block_size)
            with open('output.pcm', 'wb') as w:
                while len(audio) >= block_size:
                    result = ans(audio)
                    pcm = result[OutputKeys.OUTPUT_PCM]
                    w.write(pcm)
                    audio = f.read(block_size)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_zipenhancer_ans(self):
        model_id = 'damo/speech_zipenhancer_ans_multiloss_16k_base'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        ans(os.path.join(os.getcwd(), NOISE_SPEECH_FILE),
            output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_zipenhancer_ans_url(self):
        model_id = 'damo/speech_zipenhancer_ans_multiloss_16k_base'
        ans = pipeline(Tasks.acoustic_noise_suppression, model=model_id)
        output_path = os.path.abspath('output.wav')
        ans(NOISE_SPEECH_URL, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_zipenhancer_ans_bytes(self):
        model_id = 'damo/speech_zipenhancer_ans_multiloss_16k_base'
        ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model=model_id,
            pipeline_name=Pipelines.speech_zipenhancer_ans_multiloss_16k_base)
        output_path = os.path.abspath('output.wav')
        with open(os.path.join(os.getcwd(), NOISE_SPEECH_FILE), 'rb') as f:
            data = f.read()
            ans(data, output_path=output_path)
        print(f'Processed audio saved to {output_path}')



if __name__ == '__main__':
    unittest.main()
