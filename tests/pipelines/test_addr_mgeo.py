# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextClassificationPipeline
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class MGeoTest(unittest.TestCase, DemoCompatibilityCheck):

    multi_modal_inputs = {
        'source_sentence': ['杭州余杭东方未来学校附近世纪华联商场(金家渡北苑店)'],
        'first_sequence_gis': [[
            [
                13159, 13295, 13136, 13157, 13158, 13291, 13294, 74505, 74713,
                75387, 75389, 75411
            ],
            [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # noqa: E126
            [[1254, 1474, 1255, 1476], [1253, 1473, 1256, 1476],
             [1247, 1473, 1255, 1480], [1252, 1475, 1253, 1476],
             [1253, 1475, 1253, 1476], [1252, 1471, 1254, 1475],
             [1254, 1473, 1256, 1475], [1238, 1427, 1339, 1490],
             [1238, 1427, 1339, 1490], [1252, 1474, 1255, 1476],
             [1252, 1474, 1255, 1476], [1249, 1472, 1255, 1479]],
            [[24, 23, 15, 23], [24, 28, 15, 18], [31, 24, 22, 22],
             [43, 13, 37, 13], [43, 6, 35, 6], [31, 32, 22, 14],
             [19, 30, 9, 16], [24, 30, 15, 16], [24, 30, 15, 16],
             [29, 24, 20, 22], [28, 25, 19, 21], [31, 26, 22, 20]],
            '120.08802231437534,30.343853313981505'
        ]],
        'sentences_to_compare': [
            '良渚街道金家渡北苑42号世纪华联超市(金家渡北苑店)', '金家渡路金家渡中苑南区70幢金家渡中苑70幢',
            '金家渡路140-142号附近家家福足道(金家渡店)'
        ],
        'second_sequence_gis':
        [[[13083, 13081, 13084, 13085, 13131, 13134, 13136, 13147, 13148],
          [3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 4, 4, 4, 4, 4, 4, 4, 4],
          [[1248, 1477, 1250, 1479], [1248, 1475, 1250, 1476],
           [1247, 1478, 1249, 1481], [1249, 1479, 1249, 1480],
           [1249, 1476, 1250, 1476], [1250, 1474, 1252, 1478],
           [1247, 1473, 1255, 1480], [1250, 1478, 1251, 1479],
           [1249, 1478, 1250, 1481]],
          [[30, 26, 21, 20], [32, 43, 23, 43], [33, 23, 23, 23],
           [31, 13, 22, 13], [25, 43, 16, 43], [20, 33, 10, 33],
           [26, 29, 17, 17], [18, 21, 8, 21], [26, 23, 17, 23]],
          '120.08075205680345,30.34697777462197'],
         [[13291, 13159, 13295, 74713, 75387, 75389, 75411],
          [3, 3, 3, 4, 4, 4, 4], [3, 4, 4, 4, 4, 4, 4],
          [[1252, 1471, 1254, 1475], [1254, 1474, 1255, 1476],
           [1253, 1473, 1256, 1476], [1238, 1427, 1339, 1490],
           [1252, 1474, 1255, 1476], [1252, 1474, 1255, 1476],
           [1249, 1472, 1255, 1479]],
          [[28, 28, 19, 18], [22, 16, 12, 16], [23, 24, 13, 22],
           [24, 30, 15, 16], [27, 20, 18, 20], [27, 21, 18, 21],
           [30, 24, 21, 22]], '120.0872539617001,30.342783672056953'],
         [[13291, 13290, 13294, 13295, 13298], [3, 3, 3, 3, 3],
          [3, 4, 4, 4, 4],
          [[1252, 1471, 1254, 1475], [1253, 1469, 1255, 1472],
           [1254, 1473, 1256, 1475], [1253, 1473, 1256, 1476],
           [1255, 1467, 1258, 1472]],
          [[32, 25, 23, 21], [26, 33, 17, 33], [21, 19, 11, 19],
           [25, 21, 16, 21], [21, 33, 11,
                              33]], '120.08839673752281,30.34156156893651']]
    }
    single_modal_inputs = {
        'source_sentence': ['杭州余杭东方未来学校附近世纪华联商场(金家渡北苑店)'],
        'sentences_to_compare': [
            '良渚街道金家渡北苑42号世纪华联超市(金家渡北苑店)', '金家渡路金家渡中苑南区70幢金家渡中苑70幢',
            '金家渡路140-142号附近家家福足道(金家渡店)'
        ]
    }

    pipe_input = [
        [
            Tasks.text_ranking,
            'damo/mgeo_geographic_textual_similarity_rerank_chinese_base',
            multi_modal_inputs
        ],
        [
            Tasks.text_ranking,
            'damo/mgeo_geographic_textual_similarity_rerank_chinese_base',
            single_modal_inputs
        ],
        [
            Tasks.token_classification,
            'damo/mgeo_geographic_elements_tagging_chinese_base',
            '浙江省杭州市余杭区阿里巴巴西溪园区'
        ],
        [
            Tasks.token_classification,
            'damo/mgeo_geographic_composition_analysis_chinese_base',
            '浙江省杭州市余杭区阿里巴巴西溪园区'
        ],
        [
            Tasks.token_classification,
            'damo/mgeo_geographic_where_what_cut_chinese_base',
            '浙江省杭州市余杭区阿里巴巴西溪园区'
        ],
        [
            Tasks.sentence_similarity,
            'damo/mgeo_geographic_entity_alignment_chinese_base',
            ('后湖金桥大道绿色新都116—120栋116号（诺雅广告）', '金桥大道46号宏宇·绿色新都120幢')
        ],
    ]

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        for task, model, inputs in self.pipe_input:
            pipeline_ins = pipeline(task=task, model=model)
            print(pipeline_ins(input=inputs))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
