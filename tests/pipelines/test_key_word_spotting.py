# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tarfile
import unittest
from typing import Any, Dict, List, Union

import requests

from modelscope.pipelines import pipeline
from modelscope.utils.constant import ColorCodes, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import download_and_untar, test_level

logger = get_logger()

POS_WAV_FILE = 'data/test/audios/kws_xiaoyunxiaoyun.wav'
BOFANGYINYUE_WAV_FILE = 'data/test/audios/kws_bofangyinyue.wav'

POS_TESTSETS_FILE = 'pos_testsets.tar.gz'
POS_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testsets.tar.gz'

NEG_TESTSETS_FILE = 'neg_testsets.tar.gz'
NEG_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/neg_testsets.tar.gz'


class KeyWordSpottingTest(unittest.TestCase):
    action_info = {
        'test_run_with_wav': {
            'checking_item': 'kws_list',
            'checking_value': '小云小云',
            'example': {
                'wav_count':
                1,
                'kws_set':
                'wav',
                'kws_list': [{
                    'keyword': '小云小云',
                    'offset': 5.76,
                    'length': 9.132938,
                    'confidence': 0.990368
                }]
            }
        },
        'test_run_with_wav_by_customized_keywords': {
            'checking_item': 'kws_list',
            'checking_value': '播放音乐',
            'example': {
                'wav_count':
                1,
                'kws_set':
                'wav',
                'kws_list': [{
                    'keyword': '播放音乐',
                    'offset': 0.87,
                    'length': 2.158313,
                    'confidence': 0.646237
                }]
            }
        },
        'test_run_with_pos_testsets': {
            'checking_item': 'recall',
            'example': {
                'wav_count': 450,
                'kws_set': 'pos_testsets',
                'wav_time': 3013.75925,
                'keywords': ['小云小云'],
                'recall': 0.953333,
                'detected_count': 429,
                'rejected_count': 21,
                'rejected': ['yyy.wav', 'zzz.wav']
            }
        },
        'test_run_with_neg_testsets': {
            'checking_item': 'fa_rate',
            'example': {
                'wav_count':
                751,
                'kws_set':
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
            'checking_item': 'keywords',
            'checking_value': '小云小云',
            'example': {
                'kws_set':
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
        }
    }

    def setUp(self) -> None:
        self.model_id = 'damo/speech_charctc_kws_phone-xiaoyunxiaoyun'
        self.workspace = os.path.join(os.getcwd(), '.tmp')
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

    def tearDown(self) -> None:
        # remove workspace dir (.tmp)
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)

    def run_pipeline(self,
                     model_id: str,
                     wav_path: Union[List[str], str],
                     keywords: List[str] = None) -> Dict[str, Any]:
        kwsbp_16k_pipline = pipeline(
            task=Tasks.auto_speech_recognition, model=model_id)

        kws_result = kwsbp_16k_pipline(wav_path=wav_path, keywords=keywords)

        return kws_result

    def print_error(self, functions: str, result: Dict[str, Any]) -> None:
        logger.error(ColorCodes.MAGENTA + functions + ': FAILED.'
                     + ColorCodes.END)
        logger.error(ColorCodes.MAGENTA + functions
                     + ' correct result example: ' + ColorCodes.YELLOW
                     + str(self.action_info[functions]['example'])
                     + ColorCodes.END)

        raise ValueError('kws result is mismatched')

    def check_and_print_result(self, functions: str,
                               result: Dict[str, Any]) -> None:
        if result.__contains__(self.action_info[functions]['checking_item']):
            checking_item = result[self.action_info[functions]
                                   ['checking_item']]
            if functions == 'test_run_with_roc':
                if checking_item[0] != self.action_info[functions][
                        'checking_value']:
                    self.print_error(functions, result)

            elif functions == 'test_run_with_wav':
                if checking_item[0]['keyword'] != self.action_info[functions][
                        'checking_value']:
                    self.print_error(functions, result)

            elif functions == 'test_run_with_wav_by_customized_keywords':
                if checking_item[0]['keyword'] != self.action_info[functions][
                        'checking_value']:
                    self.print_error(functions, result)

            logger.info(ColorCodes.MAGENTA + functions + ': SUCCESS.'
                        + ColorCodes.END)
            if functions == 'test_run_with_roc':
                find_keyword = result['keywords'][0]
                keyword_list = result[find_keyword]
                for item in iter(keyword_list):
                    threshold: float = item['threshold']
                    recall: float = item['recall']
                    fa_per_hour: float = item['fa_per_hour']
                    logger.info(ColorCodes.YELLOW + '  threshold:'
                                + str(threshold) + ' recall:' + str(recall)
                                + ' fa_per_hour:' + str(fa_per_hour)
                                + ColorCodes.END)
            else:
                logger.info(ColorCodes.YELLOW + str(result) + ColorCodes.END)
        else:
            self.print_error(functions, result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav(self):
        kws_result = self.run_pipeline(
            model_id=self.model_id, wav_path=POS_WAV_FILE)
        self.check_and_print_result('test_run_with_wav', kws_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav_by_customized_keywords(self):
        keywords = [{'keyword': '播放音乐'}]

        kws_result = self.run_pipeline(
            model_id=self.model_id,
            wav_path=BOFANGYINYUE_WAV_FILE,
            keywords=keywords)
        self.check_and_print_result('test_run_with_wav_by_customized_keywords',
                                    kws_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_pos_testsets(self):
        wav_file_path = download_and_untar(
            os.path.join(self.workspace, POS_TESTSETS_FILE), POS_TESTSETS_URL,
            self.workspace)
        wav_path = [wav_file_path, None]

        kws_result = self.run_pipeline(
            model_id=self.model_id, wav_path=wav_path)
        self.check_and_print_result('test_run_with_pos_testsets', kws_result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_neg_testsets(self):
        wav_file_path = download_and_untar(
            os.path.join(self.workspace, NEG_TESTSETS_FILE), NEG_TESTSETS_URL,
            self.workspace)
        wav_path = [None, wav_file_path]

        kws_result = self.run_pipeline(
            model_id=self.model_id, wav_path=wav_path)
        self.check_and_print_result('test_run_with_neg_testsets', kws_result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_roc(self):
        pos_file_path = download_and_untar(
            os.path.join(self.workspace, POS_TESTSETS_FILE), POS_TESTSETS_URL,
            self.workspace)
        neg_file_path = download_and_untar(
            os.path.join(self.workspace, NEG_TESTSETS_FILE), NEG_TESTSETS_URL,
            self.workspace)
        wav_path = [pos_file_path, neg_file_path]

        kws_result = self.run_pipeline(
            model_id=self.model_id, wav_path=wav_path)
        self.check_and_print_result('test_run_with_roc', kws_result)


if __name__ == '__main__':
    unittest.main()
