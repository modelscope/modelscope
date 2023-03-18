# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest
from typing import Any, Dict, List, Union

import numpy as np
import soundfile

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ColorCodes, Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import download_and_untar, test_level

logger = get_logger()

POS_WAV_FILE = 'data/test/audios/kws_xiaoyunxiaoyun.wav'
BOFANGYINYUE_WAV_FILE = 'data/test/audios/kws_bofangyinyue.wav'
URL_FILE = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/20200707_xiaoyun.wav'

POS_TESTSETS_FILE = 'pos_testsets.tar.gz'
POS_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testsets.tar.gz'

NEG_TESTSETS_FILE = 'neg_testsets.tar.gz'
NEG_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/neg_testsets.tar.gz'


class KeyWordSpottingTest(unittest.TestCase, DemoCompatibilityCheck):
    action_info = {
        'test_run_with_wav': {
            'checking_item': [OutputKeys.KWS_LIST, 0, 'keyword'],
            'checking_value': '小云小云',
            'example': {
                'wav_count':
                1,
                'kws_type':
                'wav',
                'kws_list': [{
                    'keyword': '小云小云',
                    'offset': 5.76,
                    'length': 9.132938,
                    'confidence': 0.990368
                }]
            }
        },
        'test_run_with_pcm': {
            'checking_item': [OutputKeys.KWS_LIST, 0, 'keyword'],
            'checking_value': '小云小云',
            'example': {
                'wav_count':
                1,
                'kws_type':
                'pcm',
                'kws_list': [{
                    'keyword': '小云小云',
                    'offset': 5.76,
                    'length': 9.132938,
                    'confidence': 0.990368
                }]
            }
        },
        'test_run_with_wav_by_customized_keywords': {
            'checking_item': [OutputKeys.KWS_LIST, 0, 'keyword'],
            'checking_value': '播放音乐',
            'example': {
                'wav_count':
                1,
                'kws_type':
                'wav',
                'kws_list': [{
                    'keyword': '播放音乐',
                    'offset': 0.87,
                    'length': 2.158313,
                    'confidence': 0.646237
                }]
            }
        },
        'test_run_with_url': {
            'checking_item': [OutputKeys.KWS_LIST, 0, 'keyword'],
            'checking_value': '小云小云',
            'example': {
                'wav_count':
                1,
                'kws_type':
                'pcm',
                'kws_list': [{
                    'keyword': '小云小云',
                    'offset': 0.69,
                    'length': 1.67,
                    'confidence': 0.996023
                }]
            }
        },
        'test_run_with_pos_testsets': {
            'checking_item': ['recall'],
            'example': {
                'wav_count': 450,
                'kws_type': 'pos_testsets',
                'wav_time': 3013.75925,
                'keywords': ['小云小云'],
                'recall': 0.953333,
                'detected_count': 429,
                'rejected_count': 21,
                'rejected': ['yyy.wav', 'zzz.wav']
            }
        },
        'test_run_with_neg_testsets': {
            'checking_item': ['fa_rate'],
            'example': {
                'wav_count':
                751,
                'kws_type':
                'neg_testsets',
                'wav_time':
                3572.180813,
                'keywords': ['小云小云'],
                'fa_rate':
                0.001332,
                'fa_per_hour':
                1.007788,
                'detected_count':
                1,
                'rejected_count':
                750,
                'detected': [{
                    '6.wav': {
                        'confidence': '0.321170',
                        'keyword': '小云小云'
                    }
                }]
            }
        },
        'test_run_with_roc': {
            'checking_item': ['keywords', 0],
            'checking_value': '小云小云',
            'example': {
                'kws_type':
                'roc',
                'keywords': ['小云小云'],
                '小云小云': [{
                    'threshold': 0.0,
                    'recall': 0.953333,
                    'fa_per_hour': 1.007788
                }, {
                    'threshold': 0.001,
                    'recall': 0.953333,
                    'fa_per_hour': 1.007788
                }, {
                    'threshold': 0.999,
                    'recall': 0.004444,
                    'fa_per_hour': 0.0
                }]
            }
        },
        'test_run_with_all_models': {
            'checking_item': [OutputKeys.KWS_LIST, 0, 'keyword'],
            'checking_value': '小云小云',
            'example': {
                'kws_type':
                'wav',
                'kws_list': [{
                    'keyword': '小云小云',
                    'offset': 5.76,
                    'length': 9.132938,
                    'confidence': 0.990368
                }],
                'wav_count':
                1
            }
        }
    }

    all_models_info = [{
        'model_id': 'damo/speech_charctc_kws_phone-xiaoyun-commands',
        'wav_path': 'data/test/audios/kws_xiaoyunxiaoyun.wav',
        'keywords': '小云小云'
    }, {
        'model_id': 'damo/speech_charctc_kws_phone-xiaoyun',
        'wav_path': 'data/test/audios/kws_xiaoyunxiaoyun.wav',
        'keywords': '小云小云'
    }, {
        'model_id': 'damo/speech_charctc_kws_phone-speechcommands',
        'wav_path': 'data/test/audios/kws_xiaoyunxiaoyun.wav',
        'keywords': '小云小云'
    }, {
        'model_id': 'damo/speech_charctc_kws_phone-wenwen',
        'wav_path': 'data/test/audios/kws_xiaoyunxiaoyun.wav',
        'keywords': '小云小云'
    }]

    def setUp(self) -> None:
        self.model_id = 'damo/speech_charctc_kws_phone-xiaoyun'
        self.workspace = os.path.join(os.getcwd(), '.tmp')
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

    def tearDown(self) -> None:
        # remove workspace dir (.tmp)
        shutil.rmtree(self.workspace, ignore_errors=True)

    def run_pipeline(self,
                     model_id: str,
                     audio_in: Union[List[str], str, bytes],
                     keywords: List[str] = None) -> Dict[str, Any]:
        kwsbp_16k_pipline = pipeline(
            task=Tasks.keyword_spotting, model=model_id)

        kws_result = kwsbp_16k_pipline(audio_in=audio_in, keywords=keywords)

        return kws_result

    def log_error(self, functions: str, result: Dict[str, Any]) -> None:
        logger.error(ColorCodes.MAGENTA + functions + ': FAILED.'
                     + ColorCodes.END)
        logger.error(ColorCodes.MAGENTA + functions
                     + ' correct result example: ' + ColorCodes.YELLOW
                     + str(self.action_info[functions]['example'])
                     + ColorCodes.END)

        raise ValueError('kws result is mismatched')

    def check_result(self, functions: str, result: Dict[str, Any]) -> None:
        result_item = result
        check_list = self.action_info[functions]['checking_item']
        for check_item in check_list:
            result_item = result_item[check_item]
            if result_item is None or result_item == 'None':
                self.log_error(functions, result)

        if self.action_info[functions].__contains__('checking_value'):
            check_value = self.action_info[functions]['checking_value']
            if result_item != check_value:
                self.log_error(functions, result)

        logger.info(ColorCodes.MAGENTA + functions + ': SUCCESS.'
                    + ColorCodes.END)
        if functions == 'test_run_with_roc':
            find_keyword = result['keywords'][0]
            keyword_list = result[find_keyword]
            for item in iter(keyword_list):
                threshold: float = item['threshold']
                recall: float = item['recall']
                fa_per_hour: float = item['fa_per_hour']
                logger.info(ColorCodes.YELLOW + '  threshold:' + str(threshold)
                            + ' recall:' + str(recall) + ' fa_per_hour:'
                            + str(fa_per_hour) + ColorCodes.END)
        else:
            logger.info(ColorCodes.YELLOW + str(result) + ColorCodes.END)

    def wav2bytes(self, wav_file) -> bytes:
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
        return audio

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav(self):
        kws_result = self.run_pipeline(
            model_id=self.model_id, audio_in=POS_WAV_FILE)
        self.check_result('test_run_with_wav', kws_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_pcm(self):
        audio = self.wav2bytes(os.path.join(os.getcwd(), POS_WAV_FILE))

        kws_result = self.run_pipeline(model_id=self.model_id, audio_in=audio)
        self.check_result('test_run_with_pcm', kws_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav_by_customized_keywords(self):
        keywords = '播放音乐'

        kws_result = self.run_pipeline(
            model_id=self.model_id,
            audio_in=BOFANGYINYUE_WAV_FILE,
            keywords=keywords)
        self.check_result('test_run_with_wav_by_customized_keywords',
                          kws_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_url(self):
        kws_result = self.run_pipeline(
            model_id=self.model_id, audio_in=URL_FILE)
        self.check_result('test_run_with_url', kws_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_pos_testsets(self):
        wav_file_path = download_and_untar(
            os.path.join(self.workspace, POS_TESTSETS_FILE), POS_TESTSETS_URL,
            self.workspace)
        audio_list = [wav_file_path, None]

        kws_result = self.run_pipeline(
            model_id=self.model_id, audio_in=audio_list)
        self.check_result('test_run_with_pos_testsets', kws_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_neg_testsets(self):
        wav_file_path = download_and_untar(
            os.path.join(self.workspace, NEG_TESTSETS_FILE), NEG_TESTSETS_URL,
            self.workspace)
        audio_list = [None, wav_file_path]

        kws_result = self.run_pipeline(
            model_id=self.model_id, audio_in=audio_list)
        self.check_result('test_run_with_neg_testsets', kws_result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_roc(self):
        pos_file_path = download_and_untar(
            os.path.join(self.workspace, POS_TESTSETS_FILE), POS_TESTSETS_URL,
            self.workspace)
        neg_file_path = download_and_untar(
            os.path.join(self.workspace, NEG_TESTSETS_FILE), NEG_TESTSETS_URL,
            self.workspace)
        audio_list = [pos_file_path, neg_file_path]

        kws_result = self.run_pipeline(
            model_id=self.model_id, audio_in=audio_list)
        self.check_result('test_run_with_roc', kws_result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_all_models(self):
        logger.info('test_run_with_all_models')
        for item in self.all_models_info:
            model_id = item['model_id']
            wav_path = item['wav_path']
            keywords = item['keywords']

            logger.info('run with model_id:' + model_id + ' with keywords:'
                        + keywords)
            kws_result = self.run_pipeline(
                model_id=model_id, audio_in=wav_path, keywords=keywords)
            logger.info(ColorCodes.YELLOW + str(kws_result) + ColorCodes.END)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
