import os
from os import path as osp
from typing import Dict, Optional

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModeKeys, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import init_dist
from .ofa_trainer_utils import (AdjustLabelSmoothedCrossEntropyCriterion,
                                OFADataset, get_schedule)

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.ofa_tasks)
class OFATrainer(BaseTrainer):

    def __init__(self, model: str, *args, **kwargs):
        model = Model.from_pretrained(model)
        super().__init__(osp.join(model.model_dir, ModelFile.CONFIGURATION))
        self.model_dir = model.model_dir
        self.model = model.model
        self.device_id = 0
        self.total_epoch = self.cfg.train.epoch
        self.train_batch_size = self.cfg.train.batch_size
        self.val_batch_size = self.cfg.evaluation.batch_size
        self.save_dir = self.cfg.train.save_dir
        init_dist(launcher='pytorch')
        self.train_dataset = OFADataset(
            file_path=self.cfg.dataset.train_set,
            selected_id_keys=self.cfg.dataset.selected_id_keys,
            preprocessor=OfaPreprocessor(
                model_dir=self.model_dir, split=ModeKeys.TRAIN),
        )
        self.val_dataset = OFADataset(
            file_path=self.cfg.dataset.valid_set,
            selected_id_keys=self.cfg.dataset.selected_id_keys,
            preprocessor=OfaPreprocessor(
                model_dir=self.model_dir, split=ModeKeys.EVAL),
        )
        epoch_steps = len(
            self.train_dataset) // self.cfg.train.gradient_accumulation_steps
        self.cfg.train.num_train_steps = epoch_steps * self.cfg.train.epoch
        self.criterion = AdjustLabelSmoothedCrossEntropyCriterion(
            self.cfg.train.criterion)

    def train(self, *args, **kwargs):
        assert dist.is_initialized()

        self.model.train()
        self.model.to(self.device_id)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[
                self.device_id,
            ])

        optimizer = transformers.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            correct_bias=False,
        )
        scheduler_class, scheduler_args = get_schedule(self.cfg.train)
        if scheduler_class is not None:
            lr_scheduler = scheduler_class(**{'optimizer': optimizer},
                                           **scheduler_args)
        else:
            lr_scheduler = None
        for epoch in range(self.total_epoch):
            train_sampler = DistributedSampler(
                dataset=self.train_dataset, shuffle=True)
            train_sampler.set_epoch(epoch)

            train_params = {
                'pin_memory': True,
                'collate_fn': collate_fn,
                'batch_size': self.train_batch_size,
                'shuffle': False,
                'drop_last': True,
                'sampler': train_sampler,
                'num_workers': 2,
            }

            train_loader = DataLoader(self.train_dataset, **train_params)

            for idx, batch in enumerate(train_loader, start=1):
                model_outputs = ddp_model(**batch)
                loss, sample_size, logging_output = self.criterion(
                    model_outputs, batch)
                loss.backward()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                if idx % 10 == 0:
                    logger.info(
                        'epoch: {}, train batch {}/{}, loss={:.5f}'.format(
                            epoch, idx, len(train_loader), loss.item()))
            if dist.get_rank() == 0:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                torch.save(ddp_model.module.state_dict(),
                           f'{self.ckpt_dir}/epoch{epoch}.bin')

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        pass
