# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
import unittest
from typing import Any, Dict, List

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()

SPEAKER1_A_EN_16K_WAV = 'data/test/audios/speaker1_a_en_16k.wav'
SPEAKER1_B_EN_16K_WAV = 'data/test/audios/speaker1_b_en_16k.wav'
SPEAKER2_A_EN_16K_WAV = 'data/test/audios/speaker2_a_en_16k.wav'


class SpeakerVerificationTest(unittest.TestCase):
    ecapatdnn_voxceleb_16k_model_id = 'damo/speech_ecapa-tdnn_sv_en_voxceleb_16k'
    campplus_voxceleb_16k_model_id = 'damo/speech_campplus_sv_en_voxceleb_16k'

    def setUp(self) -> None:
        self.task = Tasks.speaker_verification

    def run_pipeline(self, model_id: str, audios: List[str]) -> Dict[str, Any]:
        p = pipeline(task=self.task, model=model_id)
        result = p(audios)
        return result

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_ecapatdnn_voxceleb_16k(self):
        logger.info(
            'Run speaker verification for ecapatdnn_voxceleb_16k model')

        result = self.run_pipeline(
            model_id=self.ecapatdnn_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER2_A_EN_16K_WAV])
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_campplus_voxceleb_16k(self):
        logger.info('Run speaker verification for campplus_voxceleb_16k model')

        result = self.run_pipeline(
            model_id=self.campplus_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER2_A_EN_16K_WAV])
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)


if __name__ == '__main__':
    unittest.main()
