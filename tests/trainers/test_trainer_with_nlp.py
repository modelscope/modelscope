# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Metrics
from modelscope.models.base import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level


class TestTrainerWithNlp(unittest.TestCase):
    sentence1 = '今天气温比昨天高么？'
    sentence2 = '今天湿度比昨天高么？'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.dataset = MsDataset.load(
            'clue', subset_name='afqmc',
            split='train').to_hf_dataset().select(range(2))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
        kwargs = dict(
            model=model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(10):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

        output_files = os.listdir(
            os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR))
        self.assertIn(ModelFile.CONFIGURATION, output_files)
        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE, output_files)
        copy_src_files = os.listdir(trainer.model_dir)

        print(f'copy_src_files are {copy_src_files}')
        print(f'output_files are {output_files}')
        for item in copy_src_files:
            if not item.startswith('.'):
                self.assertIn(item, output_files)

        def pipeline_sentence_similarity(model_dir):
            model = Model.from_pretrained(model_dir)
            pipeline_ins = pipeline(
                task=Tasks.sentence_similarity, model=model)
            print(pipeline_ins(input=(self.sentence1, self.sentence2)))

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        pipeline_sentence_similarity(output_dir)

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_trainer_with_backbone_head(self):
        model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
        kwargs = dict(
            model=model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(10):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

        eval_results = trainer.evaluate(
            checkpoint_path=os.path.join(self.tmp_dir, 'epoch_10.pth'))
        self.assertTrue(Metrics.accuracy in eval_results)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_user_defined_config(self):
        model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
        cfg = read_config(model_id)
        cfg.train.max_epochs = 20
        cfg.preprocessor.train['label2id'] = {'0': 0, '1': 1}
        cfg.preprocessor.val['label2id'] = {'0': 0, '1': 1}
        cfg.train.work_dir = self.tmp_dir
        cfg_file = os.path.join(self.tmp_dir, 'config.json')
        cfg.dump(cfg_file)
        kwargs = dict(
            model=model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            cfg_file=cfg_file)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(20):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

        eval_results = trainer.evaluate(
            checkpoint_path=os.path.join(self.tmp_dir, 'epoch_10.pth'))
        self.assertTrue(Metrics.accuracy in eval_results)

    @unittest.skip('skip for now before test is re-configured')
    def test_trainer_with_configured_datasets(self):
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
        cfg: Config = read_config(model_id)
        cfg.train.max_epochs = 20
        cfg.preprocessor.train['label2id'] = {'0': 0, '1': 1}
        cfg.preprocessor.val['label2id'] = {'0': 0, '1': 1}
        cfg.train.work_dir = self.tmp_dir
        cfg.dataset = {
            'train': {
                'name': 'clue',
                'subset_name': 'afqmc',
                'split': 'train',
            },
            'val': {
                'name': 'clue',
                'subset_name': 'afqmc',
                'split': 'train',
            },
        }
        cfg_file = os.path.join(self.tmp_dir, 'config.json')
        cfg.dump(cfg_file)
        kwargs = dict(model=model_id, cfg_file=cfg_file)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(cfg.train.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

        eval_results = trainer.evaluate(
            checkpoint_path=os.path.join(self.tmp_dir, 'epoch_10.pth'))
        self.assertTrue(Metrics.accuracy in eval_results)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_continue_train(self):
        from modelscope.utils.regress_test_utils import MsRegressTool
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
        cfg: Config = read_config(model_id)
        cfg.train.max_epochs = 3
        cfg.preprocessor.first_sequence = 'sentence1'
        cfg.preprocessor.second_sequence = 'sentence2'
        cfg.preprocessor.label = 'label'
        cfg.preprocessor.train['label2id'] = {'0': 0, '1': 1}
        cfg.preprocessor.val['label2id'] = {'0': 0, '1': 1}
        cfg.train.dataloader.batch_size_per_gpu = 2
        cfg.train.hooks = [{
            'type': 'CheckpointHook',
            'interval': 3,
            'by_epoch': False,
        }, {
            'type': 'TextLoggerHook',
            'interval': 1
        }, {
            'type': 'IterTimerHook'
        }, {
            'type': 'EvaluationHook',
            'interval': 1
        }]
        cfg.train.work_dir = self.tmp_dir
        cfg_file = os.path.join(self.tmp_dir, 'config.json')
        cfg.dump(cfg_file)
        dataset = MsDataset.load('clue', subset_name='afqmc', split='train')
        dataset = dataset.to_hf_dataset().select(range(4))
        kwargs = dict(
            model=model_id,
            train_dataset=dataset,
            eval_dataset=dataset,
            cfg_file=cfg_file)

        regress_tool = MsRegressTool(baseline=True)
        trainer: EpochBasedTrainer = build_trainer(default_args=kwargs)

        def lazy_stop_callback():
            from modelscope.trainers.hooks.hook import Hook, Priority

            class EarlyStopHook(Hook):
                PRIORITY = Priority.VERY_LOW

                def after_iter(self, trainer):
                    if trainer.iter == 3:
                        raise MsRegressTool.EarlyStopError('Test finished.')

            if 'EarlyStopHook' not in [
                    hook.__class__.__name__ for hook in trainer.hooks
            ]:
                trainer.register_hook(EarlyStopHook())

        with regress_tool.monitor_ms_train(
                trainer,
                'trainer_continue_train',
                level='strict',
                lazy_stop_callback=lazy_stop_callback):
            trainer.train()

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        trainer = build_trainer(default_args=kwargs)
        regress_tool = MsRegressTool(baseline=False)
        with regress_tool.monitor_ms_train(
                trainer, 'trainer_continue_train', level='strict'):
            trainer.train(os.path.join(self.tmp_dir, 'iter_3.pth'))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_evaluation(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
        cache_path = snapshot_download(model_id)
        model = SbertForSequenceClassification.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        print(trainer.evaluate(cache_path + '/pytorch_model.bin'))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
        cache_path = snapshot_download(model_id)
        model = SbertForSequenceClassification.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            max_epochs=2,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(2):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
