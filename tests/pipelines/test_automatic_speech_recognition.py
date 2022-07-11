# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tarfile
import unittest

import requests

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

WAV_FILE = 'data/test/audios/asr_example.wav'

LITTLE_TESTSETS_FILE = 'data_aishell.tar.gz'
LITTLE_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/datasets/data_aishell.tar.gz'

AISHELL1_TESTSETS_FILE = 'aishell1.tar.gz'
AISHELL1_TESTSETS_URL = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/datasets/aishell1.tar.gz'


def un_tar_gz(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)


class AutomaticSpeechRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self._am_model_id = 'damo/speech_paraformer_asr_nat-aishell1-pytorch'
        # this temporary workspace dir will store waveform files
        self._workspace = os.path.join(os.getcwd(), '.tmp')
        if not os.path.exists(self._workspace):
            os.mkdir(self._workspace)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_wav(self):
        '''run with single waveform file
        '''

        wav_file_path = os.path.join(os.getcwd(), WAV_FILE)

        inference_16k_pipline = pipeline(
            task=Tasks.auto_speech_recognition, model=[self._am_model_id])
        self.assertTrue(inference_16k_pipline is not None)

        rec_result = inference_16k_pipline(wav_file_path)
        self.assertTrue(len(rec_result['rec_result']) > 0)
        self.assertTrue(rec_result['rec_result'] != 'None')
        '''
           result structure:
           {
               'rec_result': '每一天都要快乐喔'
           }
           or
           {
               'rec_result': 'None'
           }
        '''
        print('test_run_with_wav rec result: ' + rec_result['rec_result'])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_wav_dataset(self):
        '''run with datasets, and audio format is waveform
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
        '''

        # downloading pos_testsets file
        testsets_file_path = os.path.join(self._workspace,
                                          LITTLE_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(LITTLE_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(
                os.path.splitext(
                    os.path.basename(LITTLE_TESTSETS_FILE))[0]))[0]
        # dataset_path = <cwd>/.tmp/data_aishell/wav/test
        dataset_path = os.path.join(self._workspace, testsets_dir_name, 'wav',
                                    'test')

        # untar the dataset_path file
        if not os.path.exists(dataset_path):
            un_tar_gz(testsets_file_path, self._workspace)

        inference_16k_pipline = pipeline(
            task=Tasks.auto_speech_recognition, model=[self._am_model_id])
        self.assertTrue(inference_16k_pipline is not None)

        rec_result = inference_16k_pipline(wav_path=dataset_path)
        self.assertTrue(len(rec_result['datasets_result']) > 0)
        self.assertTrue(rec_result['datasets_result']['Wrd'] > 0)
        '''
           result structure:
           {
               'rec_result': 'None',
               'datasets_result':
                   {
                       'Wrd': 1654,           # the number of words
                       'Snt': 128,            # the number of sentences
                       'Corr': 1573,          # the number of correct words
                       'Ins': 1,              # the number of insert words
                       'Del': 1,              # the number of delete words
                       'Sub': 80,             # the number of substitution words
                       'wrong_words': 82,     # the number of wrong words
                       'wrong_sentences': 47, # the number of wrong sentences
                       'Err': 4.96,           # WER/CER
                       'S.Err': 36.72         # SER
                   }
            }
        '''
        print('test_run_with_wav_dataset datasets result: ')
        print(rec_result['datasets_result'])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_ark_dataset(self):
        '''run with datasets, and audio format is kaldi_ark
           datasets directory:
             <dataset_path>
               test   # testsets
                 data.ark
                 data.scp
                 data.text
               dev    # devsets
                 data.ark
                 data.scp
                 data.text
               train  # trainsets
                 data.ark
                 data.scp
                 data.text
        '''

        # downloading pos_testsets file
        testsets_file_path = os.path.join(self._workspace,
                                          AISHELL1_TESTSETS_FILE)
        if not os.path.exists(testsets_file_path):
            r = requests.get(AISHELL1_TESTSETS_URL)
            with open(testsets_file_path, 'wb') as f:
                f.write(r.content)

        testsets_dir_name = os.path.splitext(
            os.path.basename(
                os.path.splitext(
                    os.path.basename(AISHELL1_TESTSETS_FILE))[0]))[0]
        # dataset_path = <cwd>/.tmp/aishell1/test
        dataset_path = os.path.join(self._workspace, testsets_dir_name, 'test')

        # untar the dataset_path file
        if not os.path.exists(dataset_path):
            un_tar_gz(testsets_file_path, self._workspace)

        inference_16k_pipline = pipeline(
            task=Tasks.auto_speech_recognition, model=[self._am_model_id])
        self.assertTrue(inference_16k_pipline is not None)

        rec_result = inference_16k_pipline(wav_path=dataset_path)
        self.assertTrue(len(rec_result['datasets_result']) > 0)
        self.assertTrue(rec_result['datasets_result']['Wrd'] > 0)
        '''
           result structure:
           {
               'rec_result': 'None',
               'datasets_result':
                   {
                       'Wrd': 104816,           # the number of words
                       'Snt': 7176,             # the number of sentences
                       'Corr': 99327,           # the number of correct words
                       'Ins': 104,              # the number of insert words
                       'Del': 155,              # the number of delete words
                       'Sub': 5334,             # the number of substitution words
                       'wrong_words': 5593,     # the number of wrong words
                       'wrong_sentences': 2898, # the number of wrong sentences
                       'Err': 5.34,             # WER/CER
                       'S.Err': 40.38           # SER
                   }
            }
        '''
        print('test_run_with_ark_dataset datasets result: ')
        print(rec_result['datasets_result'])


if __name__ == '__main__':
    unittest.main()
