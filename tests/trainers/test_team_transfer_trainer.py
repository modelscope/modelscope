import os
import unittest

import json
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.trainers.multi_modal.team.team_trainer_utils import (
    collate_fn, train_mapping, val_mapping)
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModeKeys, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


def train_worker(device_id):
    model_id = 'damo/multi-modal_team-vit-large-patch14_multi-modal-similarity'
    ckpt_dir = './ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)
    # Use epoch=1 for faster training here
    cfg = Config({
        'framework': 'pytorch',
        'task': 'multi-modal-similarity',
        'pipeline': {
            'type': 'multi-modal-similarity'
        },
        'model': {
            'type': 'team-multi-modal-similarity'
        },
        'dataset': {
            'name': 'Caltech101',
            'class_num': 101
        },
        'preprocessor': {},
        'train': {
            'epoch': 1,
            'batch_size': 32,
            'ckpt_dir': ckpt_dir
        },
        'evaluation': {
            'batch_size': 64
        }
    })
    cfg_file = '{}/{}'.format(ckpt_dir, ModelFile.CONFIGURATION)
    cfg.dump(cfg_file)

    train_dataset = MsDataset.load(
        cfg.dataset.name,
        namespace='modelscope',
        split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset()
    train_dataset = train_dataset.with_transform(train_mapping)
    val_dataset = MsDataset.load(
        cfg.dataset.name,
        namespace='modelscope',
        split='validation',
        download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset()
    val_dataset = val_dataset.with_transform(val_mapping)

    default_args = dict(
        cfg_file=cfg_file,
        model=model_id,
        device_id=device_id,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset)

    trainer = build_trainer(
        name=Trainers.image_classification_team, default_args=default_args)
    trainer.train()
    trainer.evaluate()


class TEAMTransferTrainerTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        if torch.cuda.device_count() > 0:
            train_worker(device_id=0)
        else:
            train_worker(device_id=-1)
        logger.info('Training done')


if __name__ == '__main__':
    unittest.main()
