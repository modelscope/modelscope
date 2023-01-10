# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level


class TestFinetuneFaqQuestionAnswering(unittest.TestCase):
    param = {
        'query_set': ['给妈买的，挺好的，妈妈喜欢。'],
        'support_set': [{
            'text': '挺好的，质量和服务都蛮好',
            'label': '1'
        }, {
            'text': '内容较晦涩，小孩不感兴趣',
            'label': '0'
        }, {
            'text': '贵且于我无用，买亏了',
            'label': '0'
        }, {
            'text': '挺好，不错，喜欢，，',
            'label': '1'
        }]
    }
    model_id = 'damo/nlp_structbert_faq-question-answering_chinese-base'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def build_trainer(self):
        train_dataset = MsDataset.load(
            'jd', namespace='DAMO_NLP',
            split='train').remap_columns({'sentence': 'text'})
        eval_dataset = MsDataset.load(
            'jd', namespace='DAMO_NLP',
            split='validation').remap_columns({'sentence': 'text'})

        cfg: Config = read_config(self.model_id, revision='v1.0.1')
        cfg.train.train_iters_per_epoch = 50
        cfg.evaluation.val_iters_per_epoch = 2
        cfg.train.seed = 1234
        cfg.train.hooks = [{
            'type': 'CheckpointHook',
            'by_epoch': False,
            'interval': 50
        }, {
            'type': 'EvaluationHook',
            'by_epoch': False,
            'interval': 50
        }, {
            'type': 'TextLoggerHook',
            'by_epoch': False,
            'rounding_digits': 5,
            'interval': 10
        }]
        cfg_file = os.path.join(self.tmp_dir, 'config.json')
        cfg.dump(cfg_file)

        trainer = build_trainer(
            Trainers.faq_question_answering_trainer,
            default_args=dict(
                model=self.model_id,
                work_dir=self.tmp_dir,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                cfg_file=cfg_file))
        return trainer

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_faq_model_finetune(self):
        trainer = self.build_trainer()
        trainer.train()
        evaluate_result = trainer.evaluate()
        self.assertAlmostEqual(evaluate_result['accuracy'], 0.95, delta=0.1)

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(ModelFile.TRAIN_OUTPUT_DIR, results_files)

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        pipeline_ins = pipeline(
            task=Tasks.faq_question_answering, model=self.model_id)
        result_before = pipeline_ins(self.param)
        self.assertEqual(result_before['output'][0][0]['label'], '1')
        self.assertAlmostEqual(
            result_before['output'][0][0]['score'], 0.2, delta=0.2)
        pipeline_ins = pipeline(
            task=Tasks.faq_question_answering, model=output_dir)
        result_after = pipeline_ins(self.param)
        self.assertEqual(result_after['output'][0][0]['label'], '1')
        self.assertAlmostEqual(
            result_after['output'][0][0]['score'], 0.8, delta=0.2)


if __name__ == '__main__':
    unittest.main()
