# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.megatron_utils import (convert_megatron_checkpoint,
                                             init_megatron_util,
                                             is_megatron_initialized)
from modelscope.utils.test_utils import DistributedTestCase, test_level


class MegatronTest(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skip
    def test_init_megatron_util(self):
        dummy_megatron_cfg = {
            'tensor_model_parallel_size': 1,
            'world_size': 1,
            'distributed_backend': 'nccl',
            'seed': 42,
        }
        os.environ['MASTER_PORT'] = '39500'
        init_megatron_util(dummy_megatron_cfg)
        self.assertTrue(is_megatron_initialized())

    @unittest.skip
    def test_convert_megatron_checkpoint(self):
        cache_path = snapshot_download('damo/nlp_gpt3_text-generation_1.3B')
        splited_dir = os.path.join(self.tmp_dir, 'splited')
        merged_dir = os.path.join(self.tmp_dir, 'merged')

        self._start(
            'torchrun --nproc_per_node=2 --master_port=39501',
            convert_gpt3_checkpoint,
            num_gpus=2,
            model_dir=cache_path,
            origin_dir=cache_path,
            target_dir=splited_dir)

        splited_files = os.listdir(splited_dir)
        self.assertIn('mp_rank_00_model_states.pt', splited_files)
        self.assertIn('mp_rank_01_model_states.pt', splited_files)

        self._start(
            'torchrun --nproc_per_node=1 --master_port=39502',
            convert_gpt3_checkpoint,
            num_gpus=1,
            model_dir=cache_path,
            origin_dir=splited_dir,
            target_dir=merged_dir)

        merged_files = os.listdir(merged_dir)
        self.assertIn('mp_rank_00_model_states.pt', merged_files)


def convert_gpt3_checkpoint(model_dir, origin_dir, target_dir):
    from modelscope.models.nlp.gpt3 import GPT3Config
    from modelscope.models.nlp.gpt3.distributed_gpt3 import GPT3Model

    init_megatron_util(
        {'tensor_model_parallel_size': int(os.getenv('WORLD_SIZE'))})
    config = GPT3Config.from_pretrained(model_dir)
    model = GPT3Model(config)

    convert_megatron_checkpoint(model, origin_dir, target_dir)


if __name__ == '__main__':
    unittest.main()
