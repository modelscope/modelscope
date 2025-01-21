# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from typing import Any, Dict, List, Union

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()

SPEAKER1_A_EN_16K_WAV = 'data/test/audios/speaker1_a_en_16k.wav'
SPEAKER1_B_EN_16K_WAV = 'data/test/audios/speaker1_b_en_16k.wav'
SPEAKER2_A_EN_16K_WAV = 'data/test/audios/speaker2_a_en_16k.wav'
SCL_EXAMPLE_WAV = 'data/test/audios/scl_example1.wav'
SD_EXAMPLE_WAV = 'data/test/audios/2speakers_example.wav'


class SpeakerVerificationTest(unittest.TestCase):
    tdnn_voxceleb_16k_model_id = 'iic/speech_tdnn_sv_en_voxceleb_16k'
    ecapatdnn_voxceleb_16k_model_id = 'damo/speech_ecapa-tdnn_sv_en_voxceleb_16k'
    campplus_voxceleb_16k_model_id = 'damo/speech_campplus_sv_en_voxceleb_16k'
    rdino_voxceleb_16k_model_id = 'damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k'
    sdpn_voxceleb_16k_model_id = 'iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k'
    speaker_change_locating_cn_model_id = 'damo/speech_campplus-transformer_scl_zh-cn_16k-common'
    speaker_change_lcoating_xvector_cn_model_id = 'damo/speech_xvector_transformer_scl_zh-cn_16k-common'
    eres2net_voxceleb_16k_model_id = 'damo/speech_eres2net_sv_en_voxceleb_16k'
    speaker_diarization_model_id = 'damo/speech_campplus_speaker-diarization_common'
    speaker_diarization_eres2net_model_id = 'damo/speech_eres2net-large_speaker-diarization_common'
    lre_campplus_en_cn_16k_model_id = 'damo/speech_campplus_lre_en-cn_16k'
    lre_eres2net_base_en_cn_16k_model_id = 'damo/speech_eres2net_base_lre_en-cn_16k'
    lre_eres2net_large_en_cn_16k_model_id = 'damo/speech_eres2net_large_lre_en-cn_16k'
    eres2net_aug_zh_cn_16k_common_model_id = 'damo/speech_eres2net_sv_zh-cn_16k-common'
    eres2netv2_zh_cn_16k_common_model_id = 'iic/speech_eres2netv2_sv_zh-cn_16k-common'
    eres2netv2ep4_zh_cn_16k_common_model_id = 'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common'
    rdino_3dspeaker_16k_model_id = 'damo/speech_rdino_ecapa_tdnn_sv_zh-cn_3dspeaker_16k'
    eres2net_base_3dspeaker_16k_model_id = 'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k'
    eres2net_large_3dspeaker_16k_model_id = 'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k'
    resnet_3dspeaker_16k_model_id = 'iic/speech_resnet34_sv_zh-cn_3dspeaker_16k'
    res2net_3dspeaker_16k_model_id = 'iic/speech_res2net_sv_zh-cn_3dspeaker_16k'
    lre_eres2net_large_five_lang_8k_model_id = 'damo/speech_eres2net_large_five_lre_8k'

    def run_pipeline(self,
                     model_id: str,
                     audios: Union[List[str], str],
                     task: str = None,
                     model_revision=None) -> Dict[str, Any]:
        if task is not None:
            self.task = task
        else:
            self.task = Tasks.speaker_verification
        p = pipeline(
            task=self.task, model=model_id, model_revision=model_revision)
        result = p(audios)
        return result

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_tdnn_voxceleb_16k(self):
        logger.info(
            'Run speaker verification for ecapatdnn_voxceleb_16k model')
        result = self.run_pipeline(
            model_id=self.tdnn_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER2_A_EN_16K_WAV],
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

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

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_rdino_voxceleb_16k(self):
        logger.info('Run speaker verification for rdino_voxceleb_16k model')
        result = self.run_pipeline(
            model_id=self.rdino_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.1')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_sdpn_voxceleb_16k(self):
        logger.info('Run speaker verification for sdpn_voxceleb_16k model')
        result = self.run_pipeline(
            model_id=self.sdpn_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2net_base_3dspeaker_16k(self):
        logger.info(
            'Run speaker verification for eres2net_base_3dspeaker_16k model')
        result = self.run_pipeline(
            model_id=self.eres2net_base_3dspeaker_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.1')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2net_large_3dspeaker_16k(self):
        logger.info(
            'Run speaker verification for eres2net_large_3dspeaker_16k model')
        result = self.run_pipeline(
            model_id=self.eres2net_large_3dspeaker_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_resnet_3dspeaker_16k(self):
        logger.info('Run speaker verification for resnet_3dspeaker_16k model')
        result = self.run_pipeline(
            model_id=self.resnet_3dspeaker_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_res2net_3dspeaker_16k(self):
        logger.info('Run speaker verification for res2net_3dspeaker_16k model')
        result = self.run_pipeline(
            model_id=self.res2net_3dspeaker_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_rdino_3dspeaker_16k(self):
        logger.info('Run speaker verification for rdino_3dspeaker_16k model')
        result = self.run_pipeline(
            model_id=self.rdino_3dspeaker_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.1')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_change_locating_cn_16k(self):
        logger.info(
            'Run speaker change locating for campplus-transformer model')
        result = self.run_pipeline(
            model_id=self.speaker_change_locating_cn_model_id,
            task=Tasks.speaker_diarization,
            audios=SCL_EXAMPLE_WAV)
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_change_locating_xvector_cn_16k(self):
        logger.info(
            'Run speaker change locating for xvector-transformer model')
        result = self.run_pipeline(
            model_id=self.speaker_change_lcoating_xvector_cn_model_id,
            task=Tasks.speaker_diarization,
            audios=SCL_EXAMPLE_WAV)
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2net_voxceleb_16k(self):
        logger.info('Run speaker verification for eres2net_voxceleb_16k model')
        result = self.run_pipeline(
            model_id=self.eres2net_voxceleb_16k_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.3')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2net_aug_zh_cn_common_16k(self):
        logger.info(
            'Run speaker verification for eres2net_zh_cn_common_16k model')
        result = self.run_pipeline(
            model_id=self.eres2net_aug_zh_cn_16k_common_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.5')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2netv2_zh_cn_common_16k(self):
        logger.info(
            'Run speaker verification for eres2netv2_zh_cn_common_16k model')
        result = self.run_pipeline(
            model_id=self.eres2netv2_zh_cn_16k_common_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.2')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_verification_eres2netv2ep4w24s4_zh_cn_common_16k(
            self):
        logger.info(
            'Run speaker verification for eres2netv2ep4_zh_cn_common_16k model'
        )
        result = self.run_pipeline(
            model_id=self.eres2netv2ep4_zh_cn_16k_common_model_id,
            audios=[SPEAKER1_A_EN_16K_WAV, SPEAKER1_B_EN_16K_WAV],
            model_revision='v1.0.1')
        print(result)
        self.assertTrue(OutputKeys.SCORE in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_speaker_diarization_common(self):
        logger.info('Run speaker diarization task')
        result = self.run_pipeline(
            model_id=self.speaker_diarization_model_id,
            task=Tasks.speaker_diarization,
            audios=SD_EXAMPLE_WAV)
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_eres2net_speaker_diarization_common(self):
        logger.info('Run eres2net speaker diarization task')
        result = self.run_pipeline(
            model_id=self.speaker_diarization_eres2net_model_id,
            task=Tasks.speaker_diarization,
            audios=SD_EXAMPLE_WAV)
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_language_recognition_campplus_en_cn_16k(self):
        logger.info('Run language recognition for campplus_en_cn_16k')
        result = self.run_pipeline(
            model_id=self.lre_campplus_en_cn_16k_model_id,
            task=Tasks.speech_language_recognition,
            audios=SPEAKER1_A_EN_16K_WAV)
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_language_recognition_eres2net_base_en_cn_16k(self):
        logger.info('Run language recognition for eres2net_base_en_cn_16k')
        result = self.run_pipeline(
            model_id=self.lre_eres2net_base_en_cn_16k_model_id,
            task=Tasks.speech_language_recognition,
            audios=SPEAKER1_A_EN_16K_WAV,
            model_revision='v1.0.2')
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_language_recognition_eres2net_large_en_cn_16k(self):
        logger.info('Run language recognition for eres2net_large_en_cn_16k')
        result = self.run_pipeline(
            model_id=self.lre_eres2net_large_en_cn_16k_model_id,
            task=Tasks.speech_language_recognition,
            audios=SPEAKER1_A_EN_16K_WAV,
            model_revision='v1.0.0')
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_language_recognition_eres2net_large_five_lang_8k(self):
        logger.info('Run language recognition for eres2net_large_five_lang_8k')
        result = self.run_pipeline(
            model_id=self.lre_eres2net_large_five_lang_8k_model_id,
            task=Tasks.speech_language_recognition,
            audios=SPEAKER1_A_EN_16K_WAV,
            model_revision='v1.0.1')
        print(result)
        self.assertTrue(OutputKeys.TEXT in result)


if __name__ == '__main__':
    unittest.main()
