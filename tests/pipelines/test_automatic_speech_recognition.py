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

LITTLE_TESTSETS_FILE = 'data_aishell.tar.gz'
LITTLE_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/datasets/data_aishell.tar.gz'

TFRECORD_TESTSETS_FILE = 'tfrecord.tar.gz'
TFRECORD_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/datasets/tfrecord.tar.gz'


class AutomaticSpeechRecognitionTest(unittest.TestCase,
                                     DemoCompatibilityCheck):
    action_info = {
        'test_run_with_wav_pytorch': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_pcm_pytorch': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_wav_tf': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_pcm_tf': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_url_pytorch': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_url_tf': {
            'checking_item': OutputKeys.TEXT,
            'example': 'wav_example'
        },
        'test_run_with_wav_dataset_pytorch': {
            'checking_item': OutputKeys.TEXT,
            'example': 'dataset_example'
        },
        'test_run_with_wav_dataset_tf': {
            'checking_item': OutputKeys.TEXT,
            'example': 'dataset_example'
        },
        'test_run_with_funasr': {
            'checking_item': OutputKeys.TEXT,
            'example': 'dataset_example'
        },
        'dataset_example': {
            'Wrd': 49532,  # the number of words
            'Snt': 5000,  # the number of sentences
            'Corr': 47276,  # the number of correct words
            'Ins': 49,  # the number of insert words
            'Del': 152,  # the number of delete words
            'Sub': 2207,  # the number of substitution words
            'wrong_words': 2408,  # the number of wrong words
            'wrong_sentences': 1598,  # the number of wrong sentences
            'Err': 4.86,  # WER/CER
            'S.Err': 31.96  # SER
        },
        'wav_example': {
            'text': '每一天都要快乐喔'
        }
    }

    all_models_info = [
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id': 'damo/speech_paraformer_asr_nat-aishell1-pytorch',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id': 'damo/speech_paraformer_asr_nat-aishell2-pytorch',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1',
            'wav_path': 'data/test/audios/asr_example_8K.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_8K.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_8K.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-cn-en-moe-16k-vocab8358-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_cn_en.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-cn-en-moe-16k-vocab8358-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_cn_en.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_cn_dialect.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_cn_dialect.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab3444-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab3444-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_8K.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_en.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_en.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_ru.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_ru.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_es.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_es.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_ko.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_ko.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_pt.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_pt.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_ja.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_ja.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-offline',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
        {
            'model_id':
            'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            'wav_path': 'data/test/audios/asr_example_id.wav'
        },
    ]

    def setUp(self) -> None:
        self.am_pytorch_model_id = 'damo/speech_paraformer_asr_nat-aishell1-pytorch'
        self.am_tf_model_id = 'damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1'
        self.am_funasr_model_id = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
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

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_pcm(self):
        """run with wav data
        """

        logger.info('Run ASR test with wav data (tensorflow)...')

        audio, sr = self.wav2bytes(os.path.join(os.getcwd(), WAV_FILE))

        rec_result = self.run_pipeline(
            model_id=self.am_tf_model_id, audio_in=audio, sr=sr)
        self.check_result('test_run_with_pcm_tf', rec_result)

        logger.info('Run ASR test with wav data (pytorch)...')

        rec_result = self.run_pipeline(
            model_id=self.am_pytorch_model_id, audio_in=audio, sr=sr)
        self.check_result('test_run_with_pcm_pytorch', rec_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_wav(self):
        """run with single waveform file
        """

        logger.info('Run ASR test with waveform file (tensorflow)...')

        wav_file_path = os.path.join(os.getcwd(), WAV_FILE)

        rec_result = self.run_pipeline(
            model_id=self.am_tf_model_id, audio_in=wav_file_path)
        self.check_result('test_run_with_wav_tf', rec_result)

        logger.info('Run ASR test with waveform file (pytorch)...')

        rec_result = self.run_pipeline(
            model_id=self.am_pytorch_model_id, audio_in=wav_file_path)
        self.check_result('test_run_with_wav_pytorch', rec_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_url(self):
        """run with single url file
        """

        logger.info('Run ASR test with url file (tensorflow)...')

        rec_result = self.run_pipeline(
            model_id=self.am_tf_model_id, audio_in=URL_FILE)
        self.check_result('test_run_with_url_tf', rec_result)

        logger.info('Run ASR test with url file (pytorch)...')

        rec_result = self.run_pipeline(
            model_id=self.am_pytorch_model_id, audio_in=URL_FILE)
        self.check_result('test_run_with_url_pytorch', rec_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_funasr(self):
        """run with single url file using FunASR
        """

        logger.info('Run ASR test with url file (FunASR)...')

        rec_result = self.run_pipeline(
            model_id=self.am_funasr_model_id, audio_in=URL_FILE)
        self.check_result('test_run_with_funasr', rec_result)

        logger.info('Run ASR test with url file (pytorch)...')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_wav_dataset_pytorch(self):
        """run with datasets, and audio format is waveform
           datasets directory:
             <dataset_path>
               wav
                 test   # testsets
                   xx.wav
                   ...
                 dev    # devsets
                   yy.wav
                   ...
                 train  # trainsets
                   zz.wav
                   ...
               transcript
                 data.text  # hypothesis text
        """

        logger.info('Downloading waveform testsets file ...')

        dataset_path = download_and_untar(
            os.path.join(self.workspace, LITTLE_TESTSETS_FILE),
            LITTLE_TESTSETS_URL, self.workspace)
        dataset_path = os.path.join(dataset_path, 'wav', 'test')

        logger.info('Run ASR test with waveform dataset (tensorflow)...')

        rec_result = self.run_pipeline(
            model_id=self.am_tf_model_id, audio_in=dataset_path)
        self.check_result('test_run_with_wav_dataset_tf', rec_result)

        logger.info('Run ASR test with waveform dataset (pytorch)...')

        rec_result = self.run_pipeline(
            model_id=self.am_pytorch_model_id, audio_in=dataset_path)
        self.check_result('test_run_with_wav_dataset_pytorch', rec_result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_all_models(self):
        """run with all models
        """

        logger.info('Run ASR test with all models')

        for item in self.all_models_info:
            model_id = item['model_id']
            wav_path = item['wav_path']
            rec_result = self.run_pipeline(
                model_id=model_id, audio_in=wav_path)
            if rec_result.__contains__(OutputKeys.TEXT):
                logger.info(ColorCodes.MAGENTA + str(item['model_id']) + ' '
                            + ColorCodes.YELLOW
                            + str(rec_result[OutputKeys.TEXT])
                            + ColorCodes.END)
            else:
                logger.info(ColorCodes.MAGENTA + str(rec_result)
                            + ColorCodes.END)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
