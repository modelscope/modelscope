# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class HifiSSRTestTask(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.speech_super_resolution
        self.model_id = 'ACoderPassBy/HifiSSR'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_face_compare(self):
        ref_wav = 'data/test/audios/speaker1_a_en_16k.wav'
        source_wav = 'data/test/audios/speaker1_a_en_16k.wav'
        # out_wav= ''
        inp_data = {
            'ref_wav': ref_wav,
            'source_wav': source_wav,
            'out_wav': ''
        }
        pipe = pipeline(Tasks.speech_super_resolution, model=self.model_id)
        pipe(inp_data)  # 输出结果将保存为"out.wav"
        print('ssr success!')


if __name__ == '__main__':
    unittest.main()
