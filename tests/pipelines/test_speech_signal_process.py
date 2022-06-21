import os.path
import shutil
import unittest

from modelscope.fileio import File
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

NEAREND_MIC_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/AEC/sample_audio/nearend_mic.wav'
FAREND_SPEECH_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/AEC/sample_audio/farend_speech.wav'
NEAREND_MIC_FILE = 'nearend_mic.wav'
FAREND_SPEECH_FILE = 'farend_speech.wav'

AEC_LIB_URL = 'http://isv-data.oss-cn-hangzhou.aliyuncs.com/ics%2FMaaS%2FAEC%2Flib%2Flibmitaec_pyio.so' \
              '?Expires=1664085465&OSSAccessKeyId=LTAIxjQyZNde90zh&Signature=Y7gelmGEsQAJRK4yyHSYMrdWizk%3D'
AEC_LIB_FILE = 'libmitaec_pyio.so'


def download(remote_path, local_path):
    local_dir = os.path.dirname(local_path)
    if len(local_dir) > 0:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
    with open(local_path, 'wb') as ofile:
        ofile.write(File.read(remote_path))


class SpeechSignalProcessTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/speech_dfsmn_aec_psm_16k'
        # A temporary hack to provide c++ lib. Download it first.
        download(AEC_LIB_URL, AEC_LIB_FILE)

    def test_run(self):
        download(NEAREND_MIC_URL, NEAREND_MIC_FILE)
        download(FAREND_SPEECH_URL, FAREND_SPEECH_FILE)
        input = {
            'nearend_mic': NEAREND_MIC_FILE,
            'farend_speech': FAREND_SPEECH_FILE
        }
        aec = pipeline(
            Tasks.speech_signal_process,
            model=self.model_id,
            pipeline_name=r'speech_dfsmn_aec_psm_16k')
        aec(input, output_path='output.wav')


if __name__ == '__main__':
    unittest.main()
