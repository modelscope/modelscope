# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import DistributedTestCase, test_level


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class TestGPT3BaseTrainDDP(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_poetry(self):
        self.start(finetune_poetry, num_gpus=4)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt3_base_evaluate_poetry(self):
        self.start(evaluate_poetry, num_gpus=4)

    # TODO: add gpt3 trainer predict unittest

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt3_base_predict_poetry(self):
        self.start(predict_poetry, num_gpus=4)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt3_base_output_pipeline(self):
        self.start(pipeline_gpt3_base_output, num_gpus=4)


def finetune_poetry(work_dir='./gpt3_poetry'):
    dataset_dict = MsDataset.load('chinese-poetry-collection')
    train_dataset = dataset_dict['train'].remap_columns({
        'text1': 'src_txt'
    }).select(range(32))
    eval_dataset = dataset_dict['test'].remap_columns({
        'text1': 'src_txt'
    }).select(range(32))
    max_epochs = 2
    tmp_dir = './gpt3_poetry'

    num_warmup_steps = 100

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step**(-0.5),
                   current_step * num_warmup_steps**(-1.5))

    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
        cfg.train.dataloader = {'batch_size_per_gpu': 16, 'workers_per_gpu': 1}
        cfg.evaluation.dataloader = {
            'batch_size_per_gpu': 2,
            'workers_per_gpu': 1
        }
        cfg.evaluation.metrics = 'ppl'
        cfg.model.strict = False
        return cfg

    kwargs = dict(
        model='damo/nlp_gpt3_text-generation_chinese-base',
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_epochs=max_epochs,
        work_dir=tmp_dir,
        cfg_modify_fn=cfg_modify_fn,
        launcher='pytorch')

    # Construct trainer and train
    trainer = build_trainer(name=Trainers.gpt3_trainer, default_args=kwargs)
    trainer.train()


def evaluate_poetry(work_dir='./gpt3_poetry'):
    dataset_dict = MsDataset.load('chinese-poetry-collection')
    eval_dataset = dataset_dict['test'].remap_columns({
        'text1': 'src_txt'
    }).select(range(20))
    tmp_dir = './gpt3_poetry'
    kwargs = dict(
        model=fr'{tmp_dir}/output',
        eval_dataset=eval_dataset,
        work_dir=tmp_dir,
        launcher='pytorch')
    trainer = build_trainer(default_args=kwargs)
    trainer.evaluate()


def predict_poetry(work_dir='./gpt3_poetry'):
    dataset_dict = MsDataset.load('chinese-poetry-collection')
    eval_dataset = dataset_dict['test'].remap_columns({
        'text1': 'src_txt'
    }).select(range(20))
    tmp_dir = './gpt3_poetry'

    kwargs = dict(
        model=fr'{tmp_dir}/output',
        predict_datasets=eval_dataset,
        work_dir=tmp_dir,
        launcher='pytorch')
    trainer = build_trainer(default_args=kwargs)
    trainer.predict()


def pipeline_gpt3_base_output(work_dir='./gpt3_poetry'):
    input = '窗含西岭千秋雪'
    tmp_dir = './gpt3_poetry'
    pipeline_ins = pipeline(
        Tasks.text_generation, model=fr'{tmp_dir}/output', work_dir=tmp_dir)
    gen_content = pipeline_ins(input, max_length=128)
    with open(
            fr'{work_dir}/nlp_gpt3_text-generation_chinese-base_pipeline_gen_text.txt',
            'w',
            encoding='utf-8') as f:
        f.write(gen_content)


class CustomTestLoader(unittest.TestLoader):

    def getTestCaseNames(self, testcase_class):
        test_names = super().getTestCaseNames(testcase_class)
        testcase_methods = list(testcase_class.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


if __name__ == '__main__':
    unittest.main(testLoader=CustomTestLoader())
