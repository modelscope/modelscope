# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class PlugDialogueTest(unittest.TestCase, DemoCompatibilityCheck):

    know_list = [
        '唐代诗人李白（701年—762年12月）,▂字太白,▂号青莲居士,▂又号“谪仙人”,▂唐代伟大的浪漫主义诗人,▂被后人誉为“诗仙”,▂与杜甫并称为“李杜”,▂为了与另两位诗人李商隐与杜牧即“小李杜”区别,▂杜甫与李白',
        '白词”享有极为崇高的地位。李白▂主要成就▂创造了古代积极浪漫主义文学高峰、为唐诗的繁荣与发展打开了新局面、开创了中国古典诗歌的黄金时代',
        '李白（701年—762年），字太白，号青莲居士，又号“谪仙人”。是唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。与杜甫并称为“李杜”，为了与另两位诗人李商隐与杜牧即“小李杜”区别，杜甫与',
    ]
    input = {
        'history': '你好[SEP]你好，我是小达，很高兴认识你！[SEP]李白是谁',
        'knowledge': '[SEP]'.join(know_list),
        'bot_profile':
        '我是小达;我是女生;我是单身;我今年21岁;我生日是2001年11月11日;我是天蝎座;我现在在复旦大学上学;我家现在常住上海',
        'user_profile': '你是小明'
    }

    def setUp(self) -> None:
        self.task = Tasks.fid_dialogue
        self.model_id = 'damo/plug-dialogue'
        self.model_revision = 'v1.0.1'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_pipeline(self):
        pipeline_ins = pipeline(
            task=self.task,
            model=self.model_id,
            model_revision=self.model_revision)
        result = pipeline_ins(self.input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
