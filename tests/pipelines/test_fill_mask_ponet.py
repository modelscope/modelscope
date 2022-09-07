# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.metainfo import Pipelines
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FillMaskPonetTest(unittest.TestCase):
    model_id_ponet = {
        'zh': 'damo/nlp_ponet_fill-mask_chinese-base',
        'en': 'damo/nlp_ponet_fill-mask_english-base'
    }

    ori_texts = {
        'zh':
        '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。'
        '你师父差得动你，你师父可差不动我。',
        'en':
        'Everything in what you call reality is really just a reflection of your '
        'consciousness. Your whole universe is just a mirror reflection of your story.'
    }

    test_inputs = {
        'zh':
        '段誉轻[MASK]折扇，摇了摇[MASK]，[MASK]道：“你师父是你的[MASK][MASK]，你'
        '师父可不是[MASK]的师父。你师父差得动你，你师父可[MASK]不动我。',
        'en':
        'Everything in [MASK] you call reality is really [MASK] a reflection of your '
        '[MASK]. Your [MASK] universe is just a mirror [MASK] of your story.'
    }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_ponet_model(self):
        for language in ['zh', 'en']:
            ori_text = self.ori_texts[language]
            test_input = self.test_inputs[language]

            pipeline_ins = pipeline(
                task=Tasks.fill_mask, model=self.model_id_ponet[language])

            print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline: '
                  f'{pipeline_ins(test_input)}\n')


if __name__ == '__main__':
    unittest.main()
