# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest
from typing import Any, Dict, Union

import numpy as np
import soundfile

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ColorCodes, Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import download_and_untar, test_level

logger = get_logger()

WAV_FILE = 'data/test/audios/asr_example.wav'
URL_FILE = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav'


class WeNetAutomaticSpeechRecognitionTest(unittest.TestCase,
                                          DemoCompatibilityCheck):
    action_info = {
        'test_run_with_pcm': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_url': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_wav': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'wav_example': {
            'text': '每一天都要快乐喔'
        }
    }

    def setUp(self) -> None:
        self.am_model_id = 'wenet/u2pp_conformer-asr-cn-16k-online'
        # this temporary workspace dir will store waveform files
        self.workspace = os.path.join(os.getcwd(), '.tmp')
        self.task = Tasks.auto_speech_recognition
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

    def tearDown(self) -> None:
        # remove workspace dir (.tmp)
        shutil.rmtree(self.workspace, ignore_errors=True)

    def run_pipeline(self,
                     model_id: str,
                     audio_in: Union[str, bytes],
                     sr: int = None) -> Dict[str, Any]:
        inference_16k_pipline = pipeline(
            task=Tasks.auto_speech_recognition, model=model_id)
        rec_result = inference_16k_pipline(audio_in, audio_fs=sr)
        return rec_result

    def log_error(self, functions: str, result: Dict[str, Any]) -> None:
        logger.error(ColorCodes.MAGENTA + functions + ': FAILED.'
                     + ColorCodes.END)
        logger.error(
            ColorCodes.MAGENTA + functions + ' correct result example:'
            + ColorCodes.YELLOW
            + str(self.action_info[self.action_info[functions]['example']])
            + ColorCodes.END)
        raise ValueError('asr result is mismatched')

    def check_result(self, functions: str, result: Dict[str, Any]) -> None:
        if result.__contains__(self.action_info[functions]['checking_item']):
            logger.info(ColorCodes.MAGENTA + functions + ': SUCCESS.'
                        + ColorCodes.END)
            logger.info(
                ColorCodes.YELLOW
                + str(result[self.action_info[functions]['checking_item']])
                + ColorCodes.END)
        else:
            self.log_error(functions, result)

    def wav2bytes(self, wav_file):
        audio, fs = soundfile.read(wav_file)

        # float32 -> int16
        audio = np.asarray(audio)
        dtype = np.dtype('int16')
        i = np.iinfo(dtype)
        abs_max = 2**(i.bits - 1)
        offset = i.min + abs_max
        audio = (audio * abs_max + offset).clip(i.min, i.max).astype(dtype)

        # int16(PCM_16) -> byte
        audio = audio.tobytes()
        return audio, fs

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_pcm(self):
        """run with wav data
        """
        logger.info('Run ASR test with wav data (wenet)...')
        audio, sr = self.wav2bytes(os.path.join(os.getcwd(), WAV_FILE))
        rec_result = self.run_pipeline(
            model_id=self.am_model_id, audio_in=audio, sr=sr)
        self.check_result('test_run_with_pcm', rec_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav(self):
        """run with single waveform file
        """
        logger.info('Run ASR test with waveform file (wenet)...')
        wav_file_path = os.path.join(os.getcwd(), WAV_FILE)
        rec_result = self.run_pipeline(
            model_id=self.am_model_id, audio_in=wav_file_path)
        self.check_result('test_run_with_wav', rec_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_url(self):
        """run with single url file
        """
        logger.info('Run ASR test with url file (wenet)...')
        rec_result = self.run_pipeline(
            model_id=self.am_model_id, audio_in=URL_FILE)
        self.check_result('test_run_with_url', rec_result)


if __name__ == '__main__':
    unittest.main()
