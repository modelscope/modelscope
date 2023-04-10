# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class FidDialogueTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.fid_dialogue
        # 240M
        self.model_id_240m = 'damo/ChatPLUG-240M'
        self.model_revision_240m = 'v1.0.0'
        # 3.7B
        self.model_id_3_7b = 'damo/ChatPLUG-3.7B'
        self.model_revision_3_7b = 'v1.0.0'
        # sample
        know_list = [
            '李白（701年—762年），字太白，号青莲居士，又号“谪仙人”。是唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。与杜甫并称为“李杜”，为了与另两位诗人李商隐与杜牧即“小李杜”区别，杜甫与',
            '李白（701年2月28日－762），字太白，号青莲居士，唐朝诗人，有“诗仙”之称，最伟大的浪漫主义诗人。汉族，出生于西域碎叶城（今吉尔吉斯斯坦托克马克），5岁随父迁至剑南道之绵州（巴西郡）',
            '李白（701─762），字太白，号青莲居士，祖籍陇西成纪（今甘肃省天水县附近）。先世于隋末流徙中亚。李白即生于中亚的碎叶城（今吉尔吉斯斯坦境内）。五岁时随其父迁居绵州彰明县（今四川省江油'
        ]
        self.input = {
            'history': '你好[SEP]你好，我是娜娜，很高兴认识你！[SEP]李白是谁',
            'bot_profile': '我是娜娜;我是女生;我是单身',
            'knowledge': '[SEP]'.join(know_list),
            'user_profile': '你是小明'
        }

        preprocess_params = {'max_encoder_length': 300, 'context_turn': 3}
        forward_params = {
            'min_length': 10,
            'max_length': 512,
            'num_beams': 1,
            'temperature': 0.8,
            'do_sample': True,
            'early_stopping': True,
            'top_k': 50,
            'top_p': 0.8,
            'repetition_penalty': 1.2,
            'length_penalty': 1.2,
            'no_repeat_ngram_size': 6
        }
        self.kwargs = {
            'preprocess_params': preprocess_params,
            'forward_params': forward_params
        }

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_240m_pipeline(self):
        pipeline_ins = pipeline(
            task=self.task,
            model=self.model_id_240m,
            model_revision=self.model_revision_240m)
        result = pipeline_ins(self.input, **self.kwargs)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_3_7b_pipeline(self):
        pipeline_ins = pipeline(
            task=self.task,
            model=self.model_id_3_7b,
            model_revision=self.model_revision_3_7b)
        result = pipeline_ins(self.input, **self.kwargs)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
