# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import BertForTextRanking
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextRankingPipeline
from modelscope.preprocessors import TextRankingTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextRankingTest(unittest.TestCase):
    models = [
        'damo/nlp_corom_passage-ranking_english-base',
        'damo/nlp_rom_passage-ranking_chinese-base'
    ]

    inputs = {
        'source_sentence': ["how long it take to get a master's degree"],
        'sentences_to_compare': [
            "On average, students take about 18 to 24 months to complete a master's degree.",
            'On the other hand, some students prefer to go at a slower pace and choose to take '
            'several years to complete their studies.',
            'It can take anywhere from two semesters'
        ]
    }

    el_model_id = 'damo/nlp_bert_entity-matching_chinese-base'
    el_inputs = {
        'source_sentence': ['我是猫》([日]夏目漱石)【摘要 [ENT_S] 书评 [ENT_E]  试读】'],
        'sentences_to_compare': [
            '书评； 类型： Other； 别名： Book review; 三元组: 书评 # 外文名 # Book review $ 书评 # 摘要 # '
            '书评，即评论并介绍书籍的文章，是以“书”为对象，实事求是的、有见识的分析书籍的形式和内容，探求创作的思想性、学术性、知识性和艺术性，从而在作者、读者和出版商之间构建信息交流的渠道。 $ 书评 # 定义 # '
            '评论并介绍书籍的文章 $ 书评 # 中文名 # 书评 $ 书评 # 义项描述 # 书评 $ 书评 # 类型 # 应用写作的一种重要文体 $ 书评 # 标签 # 文学作品、文化、出版物、小说、书籍 $',
            '摘要； 类型： Other； 别名： 摘， abstract， 书评; 三元组: 摘要 # 读音 # zhāi yào $ 摘要 # 外文名 # abstract $ 摘要 # 摘要 # '
            '摘要又称概要、内容提要，意思是摘录要点或摘录下来的要点。 $  摘要 # 词目 # 摘要 $ 摘要 # 词性 # 动词，名词 $ 摘要 # 中文名 # 摘要 $ 摘要 # 别称 # 概要、内容提要 $ 摘要 '
            '# 义项描述 # 摘要 $ 摘要 # 标签 # 文化、文学家、行业人物、法律术语、小说 $',
        ]
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        for model_id in self.models:
            cache_path = snapshot_download(model_id)
            tokenizer = TextRankingTransformersPreprocessor(cache_path)
            model = BertForTextRanking.from_pretrained(cache_path)
            pipeline1 = TextRankingPipeline(model, preprocessor=tokenizer)
            pipeline2 = pipeline(
                Tasks.text_ranking, model=model, preprocessor=tokenizer)
            print(f'sentence: {self.inputs}\n'
                  f'pipeline1:{pipeline1(input=self.inputs)}')
            print()
            print(f'pipeline2: {pipeline2(input=self.inputs)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        for model_id in self.models:
            model = Model.from_pretrained(model_id)
            tokenizer = TextRankingTransformersPreprocessor(model.model_dir)
            pipeline_ins = pipeline(
                task=Tasks.text_ranking, model=model, preprocessor=tokenizer)
            print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        for model_id in self.models:
            pipeline_ins = pipeline(task=Tasks.text_ranking, model=model_id)
            print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_ranking)
        print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_el_model(self):
        pipeline_ins = pipeline(
            task=Tasks.text_ranking, model=self.el_model_id)
        print(pipeline_ins(input=self.el_inputs))


if __name__ == '__main__':
    unittest.main()
