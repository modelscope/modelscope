# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from functools import reduce

from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level


class TestFinetuneTokenClassification(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def finetune(self,
                 model_id,
                 train_dataset,
                 eval_dataset,
                 name='NlpEpochBasedTrainer',
                 cfg_modify_fn=None,
                 **kwargs):
        kwargs = dict(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn,
            **kwargs)

        os.environ['LOCAL_RANK'] = '0'
        trainer = build_trainer(name=name, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(10):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skip
    def test_token_classification(self):
        # WS task
        os.system(
            f'curl http://dingkun.oss-cn-hangzhou-zmf.aliyuncs.com/atemp/train.txt > {self.tmp_dir}/train.txt'
        )
        os.system(
            f'curl http://dingkun.oss-cn-hangzhou-zmf.aliyuncs.com/atemp/dev.txt > {self.tmp_dir}/dev.txt'
        )
        from datasets import load_dataset
        dataset = load_dataset(
            'text',
            data_files={
                'train': f'{self.tmp_dir}/train.txt',
                'test': f'{self.tmp_dir}/dev.txt'
            })

        def split_to_dict(examples):
            text, label = examples['text'].split('\t')
            return {
                'first_sequence': text.split(' '),
                'labels': label.split(' ')
            }

        dataset = dataset.map(split_to_dict, batched=False)

        def reducer(x, y):
            x = x.split(' ') if isinstance(x, str) else x
            y = y.split(' ') if isinstance(y, str) else y
            return x + y

        label_enumerate_values = list(
            set(reduce(reducer, dataset['train'][:1000]['labels'])))
        label_enumerate_values.sort()

        def cfg_modify_fn(cfg):
            cfg.task = 'token-classification'
            cfg['preprocessor'] = {'type': 'token-cls-tokenizer'}
            cfg['dataset'] = {
                'train': {
                    'labels': label_enumerate_values,
                    'first_sequence': 'first_sequence',
                    'label': 'labels',
                }
            }
            cfg.train.max_epochs = 3
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters':
                int(len(dataset['train']) / 32) * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 300
            }]
            return cfg

        self.finetune(
            'damo/nlp_structbert_backbone_tiny_std',
            dataset['train'],
            dataset['test'],
            cfg_modify_fn=cfg_modify_fn)

    @unittest.skip
    def test_word_segmentation(self):
        os.system(
            f'curl http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip > {self.tmp_dir}/icwb2-data.zip'
        )
        shutil.unpack_archive(f'{self.tmp_dir}/icwb2-data.zip', self.tmp_dir)
        from datasets import load_dataset
        from modelscope.preprocessors.nlp import WordSegmentationBlankSetToLabelPreprocessor
        preprocessor = WordSegmentationBlankSetToLabelPreprocessor()
        dataset = load_dataset(
            'text',
            data_files=f'{self.tmp_dir}/icwb2-data/training/pku_training.utf8')

        def split_to_dict(examples):
            return preprocessor(examples['text'])

        dataset = dataset.map(split_to_dict, batched=False)

        def reducer(x, y):
            x = x.split(' ') if isinstance(x, str) else x
            y = y.split(' ') if isinstance(y, str) else y
            return x + y

        label_enumerate_values = list(
            set(reduce(reducer, dataset['train'][:1000]['labels'])))
        label_enumerate_values.sort()

        train_len = int(len(dataset['train']) * 0.7)
        train_dataset = dataset['train'].select(range(train_len))
        dev_dataset = dataset['train'].select(
            range(train_len, len(dataset['train'])))

        def cfg_modify_fn(cfg):
            cfg.task = 'token-classification'
            cfg['dataset'] = {
                'train': {
                    'labels': label_enumerate_values,
                    'first_sequence': 'first_sequence',
                    'label': 'labels',
                }
            }
            cfg['preprocessor'] = {'type': 'token-cls-tokenizer'}
            cfg.train.max_epochs = 3
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters':
                int(len(train_dataset) / 32) * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 50
            }]
            return cfg

        self.finetune(
            'damo/nlp_structbert_backbone_tiny_std',
            train_dataset,
            dev_dataset,
            cfg_modify_fn=cfg_modify_fn)


if __name__ == '__main__':
    unittest.main()
