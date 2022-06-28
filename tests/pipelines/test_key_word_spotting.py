# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tarfile
import unittest

import requests

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import build_preprocessor
from modelscope.utils.constant import Fields, InputFields, Tasks
from modelscope.utils.test_utils import test_level

KWSBP_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/tools/kwsbp'

POS_WAV_FILE = '20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_6.wav'
POS_WAV_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/' + POS_WAV_FILE

POS_TESTSETS_FILE = 'pos_testsets.tar.gz'
POS_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testsets.tar.gz'

NEG_TESTSETS_FILE = 'neg_testsets.tar.gz'
NEG_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/neg_testsets.tar.gz'


def un_tar_gz(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)


class KeyWordSpottingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/speech_charctc_kws_phone-xiaoyunxiaoyun'
        self.workspace = os.path.join(os.getcwd(), '.tmp')
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

    def tearDown(self) -> None:
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav(self):
        # wav, neg_testsets, pos_testsets, roc
        kws_set = 'wav'

        # downloading wav file
        wav_file_path = os.path.join(self.workspace, POS_WAV_FILE)
        if not os.path.exists(wav_file_path):
            r = requests.get(POS_WAV_URL)
            with open(wav_file_path, 'wb') as f:
                f.write(r.content)

        # downloading kwsbp
        kwsbp_file_path = os.path.join(self.workspace, 'kwsbp')
        if not os.path.exists(kwsbp_file_path):
            r = requests.get(KWSBP_URL)
            with open(kwsbp_file_path, 'wb') as f:
                f.write(r.content)

        model = Model.from_pretrained(self.model_id)
        self.assertTrue(model is not None)

        cfg_preprocessor = dict(
            type=Preprocessors.wav_to_lists, workspace=self.workspace)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        kwsbp_16k_pipline = pipeline(
            pipeline_name=Pipelines.kws_kwsbp,
            model=model,
            preprocessor=preprocessor)
        self.assertTrue(kwsbp_16k_pipline is not None)

        kws_result = kwsbp_16k_pipline(
            kws_type=kws_set, wav_path=[wav_file_path, None])
        self.assertTrue(kws_result.__contains__('detected'))
        """
        kws result json format example:
            {
                'wav_count': 1,
                'kws_set': 'wav',
                'wav_time': 9.132938,
                'keywords': ['小云小云'],
                'detected': True,
                'confidence': 0.990368
            }
        """
        if kws_result.__contains__('keywords'):
            print('test_run_with_wav keywords: ', kws_result['keywords'])
        print('test_run_with_wav detected result: ', kws_result['detected'])
        print('test_run_with_wav wave time(seconds): ', kws_result['wav_time'])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_pos_testsets(self):
        # wav, neg_testsets, pos_testsets, roc
        kws_set = 'pos_testsets'

        # downloading pos_testsets file
        testsets_file_path = os.path.join(self.workspace, POS_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(POS_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(POS_TESTSETS_FILE))[0]
        testsets_dir_name = os.path.splitext(
            os.path.basename(testsets_dir_name))[0]
        # wav_file_path = <cwd>/.tmp_pos_testsets/pos_testsets/
        wav_file_path = os.path.join(self.workspace, testsets_dir_name)

        # untar the pos_testsets file
        if not os.path.exists(wav_file_path):
            un_tar_gz(testsets_file_path, self.workspace)

        # downloading kwsbp -- a kws batch processing tool
        kwsbp_file_path = os.path.join(self.workspace, 'kwsbp')
        if not os.path.exists(kwsbp_file_path):
            r = requests.get(KWSBP_URL)
            with open(kwsbp_file_path, 'wb') as f:
                f.write(r.content)

        model = Model.from_pretrained(self.model_id)
        self.assertTrue(model is not None)

        cfg_preprocessor = dict(
            type=Preprocessors.wav_to_lists, workspace=self.workspace)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        kwsbp_16k_pipline = pipeline(
            pipeline_name=Pipelines.kws_kwsbp,
            model=model,
            preprocessor=preprocessor)
        self.assertTrue(kwsbp_16k_pipline is not None)

        kws_result = kwsbp_16k_pipline(
            kws_type=kws_set, wav_path=[wav_file_path, None])
        self.assertTrue(kws_result.__contains__('recall'))
        """
        kws result json format example:
            {
                'wav_count': 450,
                'kws_set': 'pos_testsets',
                'wav_time': 3013.759254,
                'keywords': ["小云小云"],
                'recall': 0.953333,
                'detected_count': 429,
                'rejected_count': 21,
                'rejected': [
                    'yyy.wav',
                    'zzz.wav',
                    ......
                ]
            }
        """
        if kws_result.__contains__('keywords'):
            print('test_run_with_pos_testsets keywords: ',
                  kws_result['keywords'])
        print('test_run_with_pos_testsets recall: ', kws_result['recall'])
        print('test_run_with_pos_testsets wave time(seconds): ',
              kws_result['wav_time'])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_neg_testsets(self):
        # wav, neg_testsets, pos_testsets, roc
        kws_set = 'neg_testsets'

        # downloading neg_testsets file
        testsets_file_path = os.path.join(self.workspace, NEG_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(NEG_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(NEG_TESTSETS_FILE))[0]
        testsets_dir_name = os.path.splitext(
            os.path.basename(testsets_dir_name))[0]
        # wav_file_path = <cwd>/.tmp_neg_testsets/neg_testsets/
        wav_file_path = os.path.join(self.workspace, testsets_dir_name)

        # untar the neg_testsets file
        if not os.path.exists(wav_file_path):
            un_tar_gz(testsets_file_path, self.workspace)

        # downloading kwsbp -- a kws batch processing tool
        kwsbp_file_path = os.path.join(self.workspace, 'kwsbp')
        if not os.path.exists(kwsbp_file_path):
            r = requests.get(KWSBP_URL)
            with open(kwsbp_file_path, 'wb') as f:
                f.write(r.content)

        model = Model.from_pretrained(self.model_id)
        self.assertTrue(model is not None)

        cfg_preprocessor = dict(
            type=Preprocessors.wav_to_lists, workspace=self.workspace)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        kwsbp_16k_pipline = pipeline(
            pipeline_name=Pipelines.kws_kwsbp,
            model=model,
            preprocessor=preprocessor)
        self.assertTrue(kwsbp_16k_pipline is not None)

        kws_result = kwsbp_16k_pipline(
            kws_type=kws_set, wav_path=[None, wav_file_path])
        self.assertTrue(kws_result.__contains__('fa_rate'))
        """
        kws result json format example:
            {
                'wav_count': 751,
                'kws_set': 'neg_testsets',
                'wav_time': 3572.180812,
                'keywords': ['小云小云'],
                'fa_rate': 0.001332,
                'fa_per_hour': 1.007788,
                'detected_count': 1,
                'rejected_count': 750,
                'detected': [
                    {
                        '6.wav': {
                            'confidence': '0.321170'
                        }
                    }
                ]
            }
        """
        if kws_result.__contains__('keywords'):
            print('test_run_with_neg_testsets keywords: ',
                  kws_result['keywords'])
        print('test_run_with_neg_testsets fa rate: ', kws_result['fa_rate'])
        print('test_run_with_neg_testsets fa per hour: ',
              kws_result['fa_per_hour'])
        print('test_run_with_neg_testsets wave time(seconds): ',
              kws_result['wav_time'])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_roc(self):
        # wav, neg_testsets, pos_testsets, roc
        kws_set = 'roc'

        # downloading neg_testsets file
        testsets_file_path = os.path.join(self.workspace, NEG_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(NEG_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(NEG_TESTSETS_FILE))[0]
        testsets_dir_name = os.path.splitext(
            os.path.basename(testsets_dir_name))[0]
        # neg_file_path = <workspace>/.tmp_roc/neg_testsets/
        neg_file_path = os.path.join(self.workspace, testsets_dir_name)

        # untar the neg_testsets file
        if not os.path.exists(neg_file_path):
            un_tar_gz(testsets_file_path, self.workspace)

        # downloading pos_testsets file
        testsets_file_path = os.path.join(self.workspace, POS_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(POS_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(POS_TESTSETS_FILE))[0]
        testsets_dir_name = os.path.splitext(
            os.path.basename(testsets_dir_name))[0]
        # pos_file_path = <workspace>/.tmp_roc/pos_testsets/
        pos_file_path = os.path.join(self.workspace, testsets_dir_name)

        # untar the pos_testsets file
        if not os.path.exists(pos_file_path):
            un_tar_gz(testsets_file_path, self.workspace)

        # downloading kwsbp -- a kws batch processing tool
        kwsbp_file_path = os.path.join(self.workspace, 'kwsbp')
        if not os.path.exists(kwsbp_file_path):
            r = requests.get(KWSBP_URL)
            with open(kwsbp_file_path, 'wb') as f:
                f.write(r.content)

        model = Model.from_pretrained(self.model_id)
        self.assertTrue(model is not None)

        cfg_preprocessor = dict(
            type=Preprocessors.wav_to_lists, workspace=self.workspace)
        preprocessor = build_preprocessor(cfg_preprocessor, Fields.audio)
        self.assertTrue(preprocessor is not None)

        kwsbp_16k_pipline = pipeline(
            pipeline_name=Pipelines.kws_kwsbp,
            model=model,
            preprocessor=preprocessor)
        self.assertTrue(kwsbp_16k_pipline is not None)

        kws_result = kwsbp_16k_pipline(
            kws_type=kws_set, wav_path=[pos_file_path, neg_file_path])
        """
        kws result json format example:
            {
                'kws_set': 'roc',
                'keywords': ['小云小云'],
                '小云小云': [
                    {'threshold': 0.0, 'recall': 0.953333, 'fa_per_hour': 1.007788},
                    {'threshold': 0.001, 'recall': 0.953333, 'fa_per_hour': 1.007788},
                    ......
                    {'threshold': 0.999, 'recall': 0.004444, 'fa_per_hour': 0.0}
                ]
            }
        """
        if kws_result.__contains__('keywords'):
            find_keyword = kws_result['keywords'][0]
            print('test_run_with_roc keywords: ', find_keyword)
            keyword_list = kws_result[find_keyword]
            for item in iter(keyword_list):
                threshold: float = item['threshold']
                recall: float = item['recall']
                fa_per_hour: float = item['fa_per_hour']
                print('  threshold:', threshold, ' recall:', recall,
                      ' fa_per_hour:', fa_per_hour)


if __name__ == '__main__':
    unittest.main()
