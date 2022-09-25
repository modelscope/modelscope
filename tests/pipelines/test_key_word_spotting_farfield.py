# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

TEST_SPEECH_FILE = 'data/test/audios/3ch_nihaomiya.wav'
TEST_SPEECH_URL = 'https://modelscope.cn/api/v1/models/damo/' \
                  'speech_dfsmn_kws_char_farfield_16k_nihaomiya/repo' \
                  '?Revision=master&FilePath=examples/3ch_nihaomiya.wav'


class KWSFarfieldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_normal(self):
        kws = pipeline(Tasks.keyword_spotting, model=self.model_id)
        inputs = {'input_file': os.path.join(os.getcwd(), TEST_SPEECH_FILE)}
        result = kws(inputs)
        self.assertEqual(len(result['kws_list']), 5)
        print(result['kws_list'][-1])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_url(self):
        kws = pipeline(Tasks.keyword_spotting, model=self.model_id)
        result = kws(TEST_SPEECH_URL)
        self.assertEqual(len(result['kws_list']), 5)
        print(result['kws_list'][-1])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_output(self):
        kws = pipeline(Tasks.keyword_spotting, model=self.model_id)
        inputs = {
            'input_file': os.path.join(os.getcwd(), TEST_SPEECH_FILE),
            'output_file': 'output.wav'
        }
        result = kws(inputs)
        self.assertEqual(len(result['kws_list']), 5)
        print(result['kws_list'][-1])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_input_bytes(self):
        with open(os.path.join(os.getcwd(), TEST_SPEECH_FILE), 'rb') as f:
            data = f.read()
        kws = pipeline(Tasks.keyword_spotting, model=self.model_id)
        result = kws(data)
        self.assertEqual(len(result['kws_list']), 5)
        print(result['kws_list'][-1])


if __name__ == '__main__':
    unittest.main()
