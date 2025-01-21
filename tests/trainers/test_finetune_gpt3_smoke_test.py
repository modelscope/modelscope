# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.hub import Config, read_config, snapshot_download
from modelscope.utils.test_utils import DistributedTestCase, test_level


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class TestFinetuneGPT3Smoke(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_dir = snapshot_download(
            'damo/nlp_gpt3_text-generation_1.3B')
        config: Config = read_config(
            os.path.join(self.model_dir, 'configuration.json'))
        config.megatron.world_size = 2
        config.megatron.tensor_model_parallel_size = 2
        config.dump(os.path.join(self.model_dir, 'configuration.json'))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_multi_finetune_portry(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(finetune_poetry, num_gpus=2, dist_start_cmd=dist_start_cmd)

    # TODO: add gpt3 trainer predict unittest


def finetune_poetry(work_dir='./gpt3_poetry'):
    dataset_dict = MsDataset.load('chinese-poetry-collection')
    train_dataset = dataset_dict['train'].remap_columns({
        'text1': 'src_txt'
    }).select(range(20))
    eval_dataset = dataset_dict['test'].remap_columns({
        'text1': 'src_txt'
    }).select(range(20))
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
        cfg.train.dataloader = {'batch_size_per_gpu': 2, 'workers_per_gpu': 1}
        cfg.train.hooks.append({'type': 'MegatronHook'})
        cfg.evaluation.dataloader = {
            'batch_size_per_gpu': 2,
            'workers_per_gpu': 1
        }
        cfg.evaluation.metrics = 'ppl'
        cfg.num_hidden_layers = 1
        cfg.model.strict = False
        return cfg

    kwargs = dict(
        model='damo/nlp_gpt3_text-generation_1.3B',
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_epochs=max_epochs,
        work_dir=tmp_dir,
        cfg_modify_fn=cfg_modify_fn)

    # Construct trainer and train
    trainer = build_trainer(name=Trainers.gpt3_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    unittest.main()
