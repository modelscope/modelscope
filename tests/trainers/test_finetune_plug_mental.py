# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modelscope.metainfo import Preprocessors, Trainers
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestFinetunePlugMental(unittest.TestCase):

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
                 name=Trainers.nlp_base_trainer,
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
        for i in range(self.epoch_num):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)

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

    def pipeline_sentence_similarity(self, model_dir):
        sentence1 = '今天气温比昨天高么？'
        sentence2 = '今天湿度比昨天高么？'
        model = Model.from_pretrained(model_dir)
        pipeline_ins = pipeline(task=Tasks.sentence_similarity, model=model)
        print(pipeline_ins(input=(sentence1, sentence2)))

    @unittest.skip
    def test_finetune_afqmc(self):
        """This unittest is used to reproduce the clue:afqmc dataset + plug meantal model training results.

        User can train a custom dataset by modifying this piece of code and comment the @unittest.skip.
        """

        def cfg_modify_fn(cfg):
            cfg.task = Tasks.sentence_similarity
            cfg['preprocessor'] = {'type': Preprocessors.sen_sim_tokenizer}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'labels': ['0', '1'],
                    'first_sequence': 'sentence1',
                    'second_sequence': 'sentence2',
                    'label': 'label',
                }
            }
            cfg.train.lr_scheduler.total_iters = int(
                len(dataset['train']) / 32) * cfg.train.max_epochs
            return cfg

        dataset = MsDataset.load('clue', subset_name='afqmc')
        self.finetune(
            model_id='damo/nlp_plug-mental_backbone_base',
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            cfg_modify_fn=cfg_modify_fn)
        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        self.pipeline_sentence_similarity(output_dir)


if __name__ == '__main__':
    unittest.main()
