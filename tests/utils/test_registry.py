# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.utils.constant import Tasks
from modelscope.utils.registry import Registry, build_from_cfg, default_group


class RegistryTest(unittest.TestCase):

    def test_register_class_no_task(self):
        MODELS = Registry('models')
        self.assertTrue(MODELS.name == 'models')
        self.assertTrue(default_group in MODELS.modules)
        self.assertTrue(MODELS.modules[default_group] == {})

        self.assertEqual(len(MODELS.modules), 1)

        @MODELS.register_module(module_name='cls-resnet')
        class ResNetForCls(object):
            pass

        self.assertTrue(default_group in MODELS.modules)
        self.assertTrue(MODELS.get('cls-resnet') is ResNetForCls)

    def test_register_class_with_task(self):
        MODELS = Registry('models')

        @MODELS.register_module(Tasks.image_classification, 'SwinT')
        class SwinTForCls(object):
            pass

        self.assertTrue(Tasks.image_classification in MODELS.modules)
        self.assertTrue(
            MODELS.get('SwinT', Tasks.image_classification) is SwinTForCls)

        @MODELS.register_module(Tasks.sentiment_analysis, 'Bert')
        class BertForSentimentAnalysis(object):
            pass

        self.assertTrue(Tasks.sentiment_analysis in MODELS.modules)
        self.assertTrue(
            MODELS.get('Bert', Tasks.sentiment_analysis) is
            BertForSentimentAnalysis)

        @MODELS.register_module(Tasks.object_detection)
        class DETR(object):
            pass

        self.assertTrue(Tasks.object_detection in MODELS.modules)
        self.assertTrue(MODELS.get('DETR', Tasks.object_detection) is DETR)

        self.assertEqual(len(MODELS.modules), 4)

    def test_list(self):
        MODELS = Registry('models')

        @MODELS.register_module(Tasks.image_classification, 'SwinT')
        class SwinTForCls(object):
            pass

        @MODELS.register_module(Tasks.sentiment_analysis, 'Bert')
        class BertForSentimentAnalysis(object):
            pass

        MODELS.list()
        print(MODELS)

    def test_build(self):
        MODELS = Registry('models')

        @MODELS.register_module(Tasks.image_classification, 'SwinT')
        class SwinTForCls(object):
            pass

        @MODELS.register_module(Tasks.sentiment_analysis, 'Bert')
        class BertForSentimentAnalysis(object):
            pass

        cfg = dict(type='SwinT')
        model = build_from_cfg(cfg, MODELS, Tasks.image_classification)
        self.assertTrue(isinstance(model, SwinTForCls))

        cfg = dict(type='Bert')
        model = build_from_cfg(cfg, MODELS, Tasks.sentiment_analysis)
        self.assertTrue(isinstance(model, BertForSentimentAnalysis))

        with self.assertRaises(KeyError):
            cfg = dict(type='Bert')
            model = build_from_cfg(cfg, MODELS, Tasks.image_classification)


if __name__ == '__main__':
    unittest.main()
