import datetime
import math
import os
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim

from modelscope.metainfo import Trainers
from modelscope.models import Model, TorchModel
from modelscope.msdatasets.task_datasets.audio import KWSDataLoader, KWSDataset
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.audio.audio_utils import update_conf
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.data_utils import to_device
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import (get_dist_info, get_local_rank,
                                          init_dist, is_master)

logger = get_logger()

BASETRAIN_CONF_EASY = 'basetrain_easy'
BASETRAIN_CONF_NORMAL = 'basetrain_normal'
BASETRAIN_CONF_HARD = 'basetrain_hard'
FINETUNE_CONF_EASY = 'finetune_easy'
FINETUNE_CONF_NORMAL = 'finetune_normal'
FINETUNE_CONF_HARD = 'finetune_hard'

EASY_RATIO = 0.1
NORMAL_RATIO = 0.6
HARD_RATIO = 0.3
BASETRAIN_RATIO = 0.5


@TRAINERS.register_module(module_name=Trainers.speech_dfsmn_kws_char_farfield)
class KWSFarfieldTrainer(BaseTrainer):
    DEFAULT_WORK_DIR = './work_dir'
    conf_keys = (BASETRAIN_CONF_EASY, FINETUNE_CONF_EASY,
                 BASETRAIN_CONF_NORMAL, FINETUNE_CONF_NORMAL,
                 BASETRAIN_CONF_HARD, FINETUNE_CONF_HARD)

    def __init__(self,
                 model: str,
                 work_dir: str,
                 cfg_file: Optional[str] = None,
                 arg_parse_fn: Optional[Callable] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 custom_conf: Optional[dict] = None,
                 **kwargs):

        if isinstance(model, str):
            self.model_dir = self.get_or_download_model_dir(
                model, model_revision)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is not from pretrained!'
            self.model_dir = os.path.dirname(cfg_file)

        super().__init__(cfg_file, arg_parse_fn)

        # the number of model output dimension
        # should update config outside the trainer, if user need more wake word
        num_syn = kwargs.get('num_syn', None)
        if num_syn:
            self.cfg.model.num_syn = num_syn
        self._num_classes = self.cfg.model.num_syn
        self.model = self.build_model()
        self.work_dir = work_dir

        if kwargs.get('launcher', None) is not None:
            init_dist(kwargs['launcher'])

        _, world_size = get_dist_info()
        self._dist = world_size > 1

        device_name = kwargs.get('device', 'gpu')
        if self._dist:
            local_rank = get_local_rank()
            device_name = f'cuda:{local_rank}'

        self.device = create_device(device_name)
        # model placement
        if self.device.type == 'cuda':
            self.model.to(self.device)

        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train, 'max_epochs'
            ), 'max_epochs is missing from the configuration file'
            self._max_epochs = self.cfg.train.max_epochs
        else:
            self._max_epochs = kwargs['max_epochs']
        self._train_iters = kwargs.get('train_iters_per_epoch', None)
        self._val_iters = kwargs.get('val_iters_per_epoch', None)
        if self._train_iters is None:
            self._train_iters = self.cfg.train.train_iters_per_epoch
        if self._val_iters is None:
            self._val_iters = self.cfg.evaluation.val_iters_per_epoch
        dataloader_config = self.cfg.train.dataloader
        self._threads = kwargs.get('workers', None)
        if self._threads is None:
            self._threads = dataloader_config.workers_per_gpu
        self._single_rate = BASETRAIN_RATIO
        if 'single_rate' in kwargs:
            self._single_rate = kwargs['single_rate']
        self._batch_size = dataloader_config.batch_size_per_gpu
        if 'model_bin' in kwargs:
            model_bin_file = os.path.join(self.model_dir, kwargs['model_bin'])
            self.model = torch.load(model_bin_file)
        # build corresponding optimizer and loss function
        lr = self.cfg.train.optimizer.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.data_val = None
        self.json_log_path = os.path.join(self.work_dir,
                                          '{}.log.json'.format(self.timestamp))
        self.conf_files = []
        for conf_key in self.conf_keys:
            template_file = os.path.join(self.model_dir, conf_key)
            conf_file = os.path.join(self.model_dir, f'{conf_key}.conf')
            update_conf(template_file, conf_file, custom_conf[conf_key])
            self.conf_files.append(conf_file)
        self._current_epoch = 0
        self.stages = (math.floor(self._max_epochs * EASY_RATIO),
                       math.floor(self._max_epochs * NORMAL_RATIO),
                       math.floor(self._max_epochs * HARD_RATIO))

    def build_model(self) -> nn.Module:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model = Model.from_pretrained(
            self.model_dir, cfg_dict=self.cfg, training=True)
        if isinstance(model, TorchModel) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def train(self, *args, **kwargs):
        if not self.data_val:
            self.gen_val()
        logger.info('Start training...')
        totaltime = datetime.datetime.now()

        for stage, num_epoch in enumerate(self.stages):
            self.run_stage(stage, num_epoch)

        # total time spent
        totaltime = datetime.datetime.now() - totaltime
        logger.info('Total time spent: {:.2f} hours\n'.format(
            totaltime.total_seconds() / 3600.0))

    def run_stage(self, stage, num_epoch):
        """
        Run training stages with correspond data

        Args:
            stage: id of stage
            num_epoch: the number of epoch to run in this stage
        """
        if num_epoch <= 0:
            logger.warning(f'Invalid epoch number, stage {stage} exit!')
            return
        logger.info(f'Starting stage {stage}...')
        dataset, dataloader = self.create_dataloader(
            self.conf_files[stage * 2], self.conf_files[stage * 2 + 1])
        it = iter(dataloader)
        for _ in range(num_epoch):
            self._current_epoch += 1
            epochtime = datetime.datetime.now()
            logger.info('Start epoch %d...', self._current_epoch)
            loss_train_epoch = 0.0
            validbatchs = 0
            for bi in range(self._train_iters):
                # prepare data
                feat, label = next(it)
                label = torch.reshape(label, (-1, ))
                feat = to_device(feat, self.device)
                label = to_device(label, self.device)
                # apply model
                self.optimizer.zero_grad()
                predict = self.model(feat)
                # calculate loss
                loss = self.loss_fn(
                    torch.reshape(predict, (-1, self._num_classes)), label)
                if not np.isnan(loss.item()):
                    loss.backward()
                    self.optimizer.step()
                    loss_train_epoch += loss.item()
                    validbatchs += 1
                train_result = 'Epoch: {:04d}/{:04d}, batch: {:04d}/{:04d}, loss: {:.4f}'.format(
                    self._current_epoch, self._max_epochs, bi + 1,
                    self._train_iters, loss.item())
                logger.info(train_result)
                self._dump_log(train_result)

            # average training loss in one epoch
            loss_train_epoch /= validbatchs
            loss_val_epoch = self.evaluate('')
            val_result = 'Evaluate epoch: {:04d}, loss_train: {:.4f}, loss_val: {:.4f}'.format(
                self._current_epoch, loss_train_epoch, loss_val_epoch)
            logger.info(val_result)
            self._dump_log(val_result)
            # check point
            ckpt_name = 'checkpoint_{:04d}_loss_train_{:.4f}_loss_val_{:.4f}.pth'.format(
                self._current_epoch, loss_train_epoch, loss_val_epoch)
            save_path = os.path.join(self.work_dir, ckpt_name)
            logger.info(f'Save model to {save_path}')
            torch.save(self.model, save_path)
            # time spent per epoch
            epochtime = datetime.datetime.now() - epochtime
            logger.info('Epoch {:04d} time spent: {:.2f} hours'.format(
                self._current_epoch,
                epochtime.total_seconds() / 3600.0))
        dataloader.stop()
        dataset.release()
        logger.info(f'Stage {stage} is finished.')

    def gen_val(self):
        """
        generate validation set
        """
        logger.info('Start generating validation set...')
        dataset, dataloader = self.create_dataloader(self.conf_files[2],
                                                     self.conf_files[3])
        it = iter(dataloader)

        self.data_val = []
        for bi in range(self._val_iters):
            logger.info('Iterating validation data %d', bi)
            feat, label = next(it)
            label = torch.reshape(label, (-1, ))
            self.data_val.append([feat, label])

        dataloader.stop()
        dataset.release()
        logger.info('Finish generating validation set!')

    def create_dataloader(self, base_path, finetune_path):
        dataset = KWSDataset(base_path, finetune_path, self._threads,
                             self._single_rate, self._num_classes)
        dataloader = KWSDataLoader(
            dataset, batchsize=self._batch_size, numworkers=self._threads)
        dataloader.start()
        return dataset, dataloader

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        logger.info('Start validation...')
        loss_val_epoch = 0.0

        with torch.no_grad():
            for feat, label in self.data_val:
                feat = to_device(feat, self.device)
                label = to_device(label, self.device)
                # apply model
                predict = self.model(feat)
                # calculate loss
                loss = self.loss_fn(
                    torch.reshape(predict, (-1, self._num_classes)), label)
                loss_val_epoch += loss.item()
        logger.info('Finish validation.')
        return loss_val_epoch / self._val_iters

    def _dump_log(self, msg):
        if is_master():
            with open(self.json_log_path, 'a+') as f:
                f.write(msg)
                f.write('\n')
