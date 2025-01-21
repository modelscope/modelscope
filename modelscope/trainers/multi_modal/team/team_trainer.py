# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import OrderedDict
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.multi_modal.team.team_trainer_utils import \
    get_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import Invoke
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.image_classification_team)
class TEAMImgClsTrainer(BaseTrainer):

    def __init__(self, cfg_file: str, model: str, device_id: int,
                 data_collator: Callable, train_dataset: Dataset,
                 val_dataset: Dataset, *args, **kwargs):
        super().__init__(cfg_file)

        self.cfg = Config.from_file(cfg_file)
        team_model = Model.from_pretrained(model, invoked_by=Invoke.TRAINER)
        image_model = team_model.model.image_model.vision_transformer
        classification_model = nn.Sequential(
            OrderedDict([('encoder', image_model),
                         ('classifier',
                          nn.Linear(768, self.cfg.dataset.class_num))]))
        self.model = classification_model

        for pname, param in self.model.named_parameters():
            if 'encoder' in pname:
                param.requires_grad = False

        self.device_id = device_id
        self.total_epoch = self.cfg.train.epoch
        self.train_batch_size = self.cfg.train.batch_size
        self.val_batch_size = self.cfg.evaluation.batch_size
        self.ckpt_dir = self.cfg.train.ckpt_dir

        self.collate_fn = data_collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.criterion = nn.CrossEntropyLoss().to(self.device_id)

    def train(self, *args, **kwargs):
        self.model.train()
        self.model.to(self.device_id)

        optimizer = get_optimizer(self.model)

        for epoch in range(self.total_epoch):
            train_params = {
                'pin_memory': True,
                'collate_fn': self.collate_fn,
                'batch_size': self.train_batch_size,
                'shuffle': True,
                'drop_last': True,
                'num_workers': 8
            }

            train_loader = DataLoader(self.train_dataset, **train_params)

            for batch_idx, data in enumerate(train_loader):
                img_tensor, label_tensor = data['pixel_values'], data['labels']
                img_tensor = img_tensor.to(self.device_id, non_blocking=True)
                label_tensor = label_tensor.to(
                    self.device_id, non_blocking=True)

                pred_logits = self.model(img_tensor)
                loss = self.criterion(pred_logits, label_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    logger.info(
                        'epoch: {}, train batch {}/{}, loss={:.5f}'.format(
                            epoch, batch_idx, len(train_loader), loss.item()))

            os.makedirs(self.ckpt_dir, exist_ok=True)
            torch.save(self.model.state_dict(),
                       '{}/epoch{}.pth'.format(self.ckpt_dir, epoch))
            self.evaluate()

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        if checkpoint_path is not None:
            checkpoint_params = torch.load(checkpoint_path, 'cpu')
            self.model.load_state_dict(checkpoint_params)
        self.model.eval()
        self.model.to(self.device_id)

        val_params = {
            'collate_fn': self.collate_fn,
            'batch_size': self.val_batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 8
        }
        val_loader = DataLoader(self.val_dataset, **val_params)

        tp_cnt, processed_cnt = 0, 0
        all_pred_labels, all_gt_labels = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                img_tensor, label_tensor = data['pixel_values'], data['labels']
                img_tensor = img_tensor.to(self.device_id, non_blocking=True)
                label_tensor = label_tensor.to(
                    self.device_id, non_blocking=True)

                pred_logits = self.model(img_tensor)
                pred_labels = torch.max(pred_logits, dim=1)[1]
                tp_cnt += torch.sum(pred_labels == label_tensor).item()
                processed_cnt += img_tensor.shape[0]
                logger.info('Accuracy: {:.3f}'.format(tp_cnt / processed_cnt))

                all_pred_labels.extend(pred_labels.tolist())
                all_gt_labels.extend(label_tensor.tolist())
        conf_mat = confusion_matrix(all_gt_labels, all_pred_labels)
        acc_mean_per_class = np.mean(conf_mat.diagonal()
                                     / conf_mat.sum(axis=1))
        logger.info(
            'Accuracy mean per class: {:.3f}'.format(acc_mean_per_class))
