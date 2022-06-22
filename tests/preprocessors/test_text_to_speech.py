import shutil
import unittest

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import build_preprocessor
from modelscope.utils.constant import Fields, InputFields
from modelscope.utils.logger import get_logger

logger = get_logger()


class TtsPreprocessorTest(unittest.TestCase):

    def test_preprocess(self):
        lang_type = 'pinyin'
        text = '今天天气不错，我们去散步吧。'
        cfg = dict(
            type=Preprocessors.text_to_tacotron_symbols,
            model_name='damo/speech_binary_tts_frontend_resource',
            lang_type=lang_type)
        preprocessor = build_preprocessor(cfg, Fields.audio)
        output = preprocessor(text)
        self.assertTrue(output)
        for line in output['texts']:
            print(line)


if __name__ == '__main__':
    unittest.main()
