# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class InverseTextProcessingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.inverse_text_processing,
        self.model_dict = {
            'en':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-en',
            'de':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-de',
            'es':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-es',
            'fr':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-fr',
            'id':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-id',
            'ko':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-ko',
            'ja':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-ja',
            'pt':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-pt',
            'ru':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-ru',
            'vi':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-vi',
            'tl':
            'damo/speech_inverse_text_processing_fun-text-processing-itn-tl',
        }
        self.text_in_dict = {
            'en':
            'on december second, we paid one hundred and twenty three dollars for christmas tree.',
            'de': 'einhundertdreiundzwanzig',
            'es': 'ciento veintitrés',
            'fr': 'cent vingt-trois',
            'id': 'seratus dua puluh tiga',
            'ko': '삼백오 독일 마',
            'ja': '百二十三',
            'pt': 'cento e vinte e três',
            'ru': 'сто двадцать три',
            'vi': 'một trăm hai mươi ba',
            'tl': "ika-lima mayo dalawang libo dalawampu't dalawa",
        }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_multi_language_itn(self):
        for key, value in self.model_dict.items():
            lang = key
            model_name = value
            itn_inference_pipline = pipeline(
                task=Tasks.inverse_text_processing, model=model_name)
            lang_text_in = self.text_in_dict[lang]
            itn_result = itn_inference_pipline(text_in=lang_text_in)
            print(itn_result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
