# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.import_utils import exists
from modelscope.utils.test_utils import test_level


class DomainClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_classification
        self.model_id = 'damo/nlp_domain_classification_chinese'

    @unittest.skipUnless(
        exists('fasttext'), 'Skip because fasttext is not installed')
    def test_run_with_model_name_for_zh_domain(self):
        inputs = '通过这种方式产生的离子吸收大地水分之后,可以通过潮解作用,将活性电解离子有效释放到周围土壤中,使接地极成为一个离子发生装置,' \
                 '从而改善周边土质使之达到接地要求。'
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(
        exists('fasttext'), 'Skip because fasttext is not installed')
    def test_run_with_model_name_for_zh_style(self):
        model_id = 'damo/nlp_style_classification_chinese'
        inputs = '通过这种方式产生的离子吸收大地水分之后,可以通过潮解作用,将活性电解离子有效释放到周围土壤中,使接地极成为一个离子发生装置,' \
                 '从而改善周边土质使之达到接地要求。'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(
        exists('fasttext'), 'Skip because fasttext is not installed')
    def test_run_with_model_name_for_en_style(self):
        model_id = 'damo/nlp_style_classification_english'
        inputs = 'High Power 11.1V 5200mAh Lipo Battery For RC Car Robot Airplanes ' \
                 'Helicopter RC Drone Parts 3s Lithium battery 11.1v Battery'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))


if __name__ == '__main__':
    unittest.main()
