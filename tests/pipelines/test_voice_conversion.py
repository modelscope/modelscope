# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class UnetVCTestTask(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.voice_conversion
        self.model_id = 'ACoderPassBy/UnetVC'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_face_compare(self):
        ref_wav = 'data/test/audios/speaker1_a_en_16k.wav'
        source_wav = 'data/test/audios/speaker1_a_en_16k.wav'
        inp_data = {
            'source_wav': ref_wav,
            'target_wav': source_wav,
            'save_path': '',
        }
        pipe = pipeline(
            Tasks.voice_conversion,
            model=self.model_id,
            model_revision='v1.0.0')
        pipe(inp_data)  # 输出结果将保存为"out.wav"
        print('speech vc success!')


if __name__ == '__main__':
    unittest.main()
