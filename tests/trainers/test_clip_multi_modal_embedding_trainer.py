import os
import tempfile
import unittest

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


def clip_train_worker(local_rank, ngpus, node_size, node_rank):
    global_rank = local_rank + node_rank * ngpus
    dist_world_size = node_size * ngpus

    dist.init_process_group(
        backend='nccl', world_size=dist_world_size, rank=global_rank)

    model_id = 'damo/multi-modal_clip-vit-large-patch14-chinese_multi-modal-embedding'
    local_model_dir = snapshot_download(model_id)

    default_args = dict(
        cfg_file='{}/{}'.format(local_model_dir, ModelFile.CONFIGURATION),
        model=model_id,
        device_id=local_rank)
    trainer = build_trainer(
        name=Trainers.clip_multi_modal_embedding, default_args=default_args)

    trainer.train()
    trainer.evaluate()


class CLIPMultiModalEmbeddingTrainerTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '2001'
        NODE_SIZE, NODE_RANK = 1, 0
        logger.info('Train clip with {} machines'.format(NODE_SIZE))
        ngpus = torch.cuda.device_count()
        logger.info('Machine: {} has {} GPUs'.format(NODE_RANK, ngpus))
        mp.spawn(
            clip_train_worker,
            nprocs=ngpus,
            args=(ngpus, NODE_SIZE, NODE_RANK))
        logger.info('Training done')


if __name__ == '__main__':
    unittest.main()
    ...
