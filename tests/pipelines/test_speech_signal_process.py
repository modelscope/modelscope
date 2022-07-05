import os.path
import shutil
import unittest

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

NEAREND_MIC_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/AEC/sample_audio/nearend_mic.wav'
FAREND_SPEECH_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/AEC/sample_audio/farend_speech.wav'
NEAREND_MIC_FILE = 'nearend_mic.wav'
FAREND_SPEECH_FILE = 'farend_speech.wav'

AEC_LIB_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/dependencies/ics_MaaS_AEC_lib_libmitaec_pyio.so'
AEC_LIB_FILE = 'libmitaec_pyio.so'

NOISE_SPEECH_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ANS/sample_audio/speech_with_noise.wav'
NOISE_SPEECH_FILE = 'speech_with_noise.wav'


def download(remote_path, local_path):
    local_dir = os.path.dirname(local_path)
    if len(local_dir) > 0:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
    with open(local_path, 'wb') as ofile:
        ofile.write(File.read(remote_path))


class SpeechSignalProcessTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_aec(self):
        # A temporary hack to provide c++ lib. Download it first.
        download(AEC_LIB_URL, AEC_LIB_FILE)
        # Download audio files
        download(NEAREND_MIC_URL, NEAREND_MIC_FILE)
        download(FAREND_SPEECH_URL, FAREND_SPEECH_FILE)
        model_id = 'damo/speech_dfsmn_aec_psm_16k'
        input = {
            'nearend_mic': NEAREND_MIC_FILE,
            'farend_speech': FAREND_SPEECH_FILE
        }
        aec = pipeline(
            Tasks.speech_signal_process,
            model=model_id,
            pipeline_name=Pipelines.speech_dfsmn_aec_psm_16k)
        output_path = os.path.abspath('output.wav')
        aec(input, output_path=output_path)
        print(f'Processed audio saved to {output_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ans(self):
        # Download audio files
        download(NOISE_SPEECH_URL, NOISE_SPEECH_FILE)
        model_id = 'damo/speech_frcrn_ans_cirm_16k'
        ans = pipeline(
            Tasks.speech_signal_process,
            model=model_id,
            pipeline_name=Pipelines.speech_frcrn_ans_cirm_16k)
        output_path = os.path.abspath('output.wav')
        ans(NOISE_SPEECH_FILE, output_path=output_path)
        print(f'Processed audio saved to {output_path}')


if __name__ == '__main__':
    unittest.main()
