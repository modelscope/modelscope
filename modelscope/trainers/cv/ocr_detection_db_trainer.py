# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import datetime
import math
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from easydict import EasyDict as easydict
from tqdm import tqdm

from modelscope.metainfo import Trainers
from modelscope.models.cv.ocr_detection.modules.dbnet import (DBModel,
                                                              DBModel_v2)
from modelscope.models.cv.ocr_detection.utils import (boxes_from_bitmap,
                                                      polygons_from_bitmap)
from modelscope.msdatasets.dataset_cls.custom_datasets.ocr_detection import (
    DataLoader, ImageDataset, QuadMeasurer)
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import get_rank, synchronize


@TRAINERS.register_module(module_name=Trainers.ocr_detection_db)
class OCRDetectionDBTrainer(BaseTrainer):

    def __init__(self,
                 model: str = None,
                 cfg_file: str = None,
                 load_pretrain: bool = True,
                 cache_path: str = None,
                 model_revision: str = DEFAULT_MODEL_REVISION,
                 *args,
                 **kwargs):
        """ High-level finetune api for dbnet.

        Args:
            model: Model id of modelscope models.
            cfg_file: Path to configuration file.
            load_pretrain: Whether load pretrain model for finetune.
                if False, means training from scratch.
            cache_path: cache path of model files.
        """
        if model is not None:
            self.cache_path = self.get_or_download_model_dir(
                model, model_revision)
            if cfg_file is None:
                self.cfg_file = os.path.join(self.cache_path,
                                             ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None and cache_path is not None, \
                'cfg_file and cache_path is needed, if model is not provided'

        if cfg_file is not None:
            self.cfg_file = cfg_file
            if cache_path is not None:
                self.cache_path = cache_path
        super().__init__(self.cfg_file)
        cfg = self.cfg
        if load_pretrain:
            if 'pretrain_model' in kwargs:
                cfg.train.finetune_path = kwargs['pretrain_model']
            else:
                cfg.train.finetune_path = os.path.join(self.cache_path,
                                                       self.cfg.model.weights)

        if 'framework' in self.cfg:
            cfg = self._config_transform(cfg)

        if 'gpu_ids' in kwargs:
            cfg.train.gpu_ids = kwargs['gpu_ids']
        if 'batch_size' in kwargs:
            cfg.train.batch_size = kwargs['batch_size']
        if 'max_epochs' in kwargs:
            cfg.train.total_epochs = kwargs['max_epochs']
        if 'base_lr' in kwargs:
            cfg.train.base_lr = kwargs['base_lr']
        if 'train_data_dir' in kwargs:
            cfg.dataset.train_data_dir = kwargs['train_data_dir']
        if 'val_data_dir' in kwargs:
            cfg.dataset.val_data_dir = kwargs['val_data_dir']
        if 'train_data_list' in kwargs:
            cfg.dataset.train_data_list = kwargs['train_data_list']
        if 'val_data_list' in kwargs:
            cfg.dataset.val_data_list = kwargs['val_data_list']

        self.gpu_ids = cfg.train.gpu_ids
        self.world_size = len(self.gpu_ids)

        self.cfg = cfg

    def train(self):
        trainer = DBTrainer(self.cfg)
        trainer.train(local_rank=0)

    def evaluate(self,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        if checkpoint_path is not None:
            self.cfg.test.checkpoint_path = checkpoint_path
        evaluater = DBTrainer(self.cfg)
        evaluater.evaluate(local_rank=0)

    def _config_transform(self, config):
        new_config = easydict({})
        new_config.miscs = config.train.miscs
        new_config.miscs.output_dir = config.train.work_dir
        new_config.model = config.model
        new_config.dataset = config.dataset
        new_config.train = config.train
        new_config.test = config.evaluation

        new_config.train.dataloader.num_gpus = len(config.train.gpu_ids)
        new_config.train.dataloader.batch_size = len(
            config.train.gpu_ids) * config.train.dataloader.batch_size_per_gpu
        new_config.train.dataloader.num_workers = len(
            config.train.gpu_ids) * config.train.dataloader.workers_per_gpu
        new_config.train.total_epochs = config.train.max_epochs

        new_config.test.dataloader.num_gpus = 1
        new_config.test.dataloader.num_workers = 4
        new_config.test.dataloader.collect_fn = config.evaluation.transform.collect_fn

        return new_config


class DBTrainer:

    def __init__(self, cfg):
        self.init_device()

        self.cfg = cfg
        self.dir_path = cfg.miscs.output_dir
        self.lr = cfg.train.base_lr
        self.current_lr = 0

        self.total = 0

        if len(cfg.train.gpu_ids) > 1:
            self.distributed = True
        else:
            self.distributed = False

        self.file_name = os.path.join(cfg.miscs.output_dir, cfg.miscs.exp_name)

        # setup logger
        if get_rank() == 0:
            os.makedirs(self.file_name, exist_ok=True)

        self.logger = get_logger(os.path.join(self.file_name, 'train_log.txt'))

        # logger
        self.logger.info('cfg value:\n{}'.format(self.cfg))

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self, local_rank):
        model = DBModel_v2(self.device, self.distributed, local_rank)
        return model

    def get_learning_rate(self, epoch, step=None):
        # DecayLearningRate
        factor = 0.9
        rate = np.power(1.0 - epoch / float(self.cfg.train.total_epochs + 1),
                        factor)
        return rate * self.lr

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.get_learning_rate(epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def restore_model(self, model, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    def create_optimizer(self, lr=0.007, momentum=0.9, weight_decay=0.0001):
        bn_group, weight_group, bias_group = [], [], []

        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                bias_group.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                bn_group.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                weight_group.append(v.weight)

        optimizer = torch.optim.SGD(
            bn_group, lr=lr, momentum=momentum, nesterov=True)
        optimizer.add_param_group({
            'params': weight_group,
            'weight_decay': weight_decay
        })
        optimizer.add_param_group({'params': bias_group})
        return optimizer

    def maybe_save_model(self, model, epoch, step):
        if step % self.cfg.miscs.save_interval == 0:
            self.logger.info('save interval model for step ' + str(step))
            self.save_model(model, epoch, step)

    def save_model(self, model, epoch=None, step=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch, step)
                self.save_checkpoint(net, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name('model', epoch, step)
            self.save_checkpoint(model, checkpoint_name)

    def save_checkpoint(self, net, name):
        os.makedirs(self.dir_path, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(self.dir_path, name))
        self.logger.info('save_checkpoint to: '
                         + os.path.join(self.dir_path, name))

    def convert_model_for_inference(self, finetune_model_name,
                                    infer_model_name):
        # Convert finetuned model to model for inference,
        # remove some param for training.
        infer_model = DBModel().to(self.device)
        model_state_dict = infer_model.state_dict()
        model_keys = list(model_state_dict.keys())
        saved_dict = torch.load(
            os.path.join(self.dir_path, finetune_model_name),
            map_location=self.device)
        saved_keys = set(saved_dict.keys())
        prefix = 'model.module.'
        for i in range(len(model_keys)):
            if prefix + model_keys[i] in saved_keys:
                model_state_dict[model_keys[i]] = (
                    saved_dict[prefix + model_keys[i]].cpu().float())
        infer_model.load_state_dict(model_state_dict)
        torch.save(infer_model.state_dict(),
                   os.path.join(self.dir_path, infer_model_name))

    def make_checkpoint_name(self, name, epoch=None, step=None):
        if epoch is None or step is None:
            c_name = name + '_latest.pt'
        else:
            c_name = '{}_epoch_{}_minibatch_{}.pt'.format(name, epoch, step)
        return c_name

    def get_data_loader(self, cfg, distributed=False):
        train_dataset = ImageDataset(cfg, cfg.dataset.train_data_dir,
                                     cfg.dataset.train_data_list)
        train_dataloader = DataLoader(
            train_dataset,
            cfg.train.dataloader,
            is_train=True,
            distributed=distributed)
        test_dataset = ImageDataset(cfg, cfg.dataset.val_data_dir,
                                    cfg.dataset.val_data_list)
        test_dataloader = DataLoader(
            test_dataset,
            cfg.test.dataloader,
            is_train=False,
            distributed=distributed)
        return train_dataloader, test_dataloader

    def train(self, local_rank):
        # Build model for training
        self.model = self.init_model(local_rank)

        # Build dataloader
        self.train_data_loader, self.validation_loaders = self.get_data_loader(
            self.cfg, self.distributed)
        # Resume model from finetune_path
        self.steps = 0
        if self.cfg.train.finetune_path is not None:
            self.logger.info(f'finetune from {self.cfg.train.finetune_path}')
            self.restore_model(self.model, self.cfg.train.finetune_path,
                               self.device)
        epoch = 0

        # Build optimizer
        optimizer = self.create_optimizer(self.lr)

        self.logger.info('Start Training...')

        self.model.train()

        # Training loop
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.total = len(self.train_data_loader)
            for batch in self.train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                self.train_step(
                    self.model, optimizer, batch, epoch=epoch, step=self.steps)

                # Save interval model
                self.maybe_save_model(self.model, epoch, self.steps)

                self.steps += 1

            epoch += 1
            if epoch > self.cfg.train.total_epochs:
                self.save_checkpoint(self.model, 'final.pt')
                self.convert_model_for_inference('final.pt',
                                                 'pytorch_model.pt')
                self.logger.info('Training done')
                break

    def train_step(self, model, optimizer, batch, epoch, step):
        optimizer.zero_grad()

        results = model.forward(batch, training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        if step % self.cfg.train.miscs.print_interval_iters == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(
                    ['step:{:6d}', 'epoch:{:3d}', '{}',
                     'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' %
                                 (step, epoch, loss.item(), self.current_lr))
            for name, metric in metrics.items():
                self.logger.info('%s: %6f' % (name, metric.mean()))

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def represent(self, batch, _pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, 1, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, 1, H, W)
        '''
        images = batch['image']
        if isinstance(_pred, dict):
            pred = _pred['binary']
        else:
            pred = _pred
        segmentation = pred > self.cfg.test.thresh
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            height, width = batch['shape'][batch_index]
            if is_output_polygon:
                boxes, scores = polygons_from_bitmap(pred[batch_index],
                                                     segmentation[batch_index],
                                                     width, height)
            else:
                boxes, scores = boxes_from_bitmap(pred[batch_index],
                                                  segmentation[batch_index],
                                                  width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def evaluate(self, local_rank):
        self.init_torch_tensor()
        # Build model for evaluation
        model = self.init_model(local_rank)

        # Restore model from checkpoint_path
        self.restore_model(model, self.cfg.test.checkpoint_path, self.device)

        # Build dataloader for evaluation
        self.train_data_loader, self.validation_loaders = self.get_data_loader(
            self.cfg, self.distributed)
        # Build evaluation metric
        quad_measurer = QuadMeasurer()
        model.eval()

        with torch.no_grad():
            raw_metrics = []
            for i, batch in tqdm(
                    enumerate(self.validation_loaders),
                    total=len(self.validation_loaders)):
                pred = model.forward(batch, training=False)
                output = self.represent(batch, pred,
                                        self.cfg.test.return_polygon)
                raw_metric = quad_measurer.validate_measure(
                    batch,
                    output,
                    is_output_polygon=self.cfg.test.return_polygon,
                    box_thresh=0.3)
                raw_metrics.append(raw_metric)
            metrics = quad_measurer.gather_measure(raw_metrics)
            for key, metric in metrics.items():
                self.logger.info('%s : %f (%d)' %
                                 (key, metric.avg, metric.count))

        self.logger.info('Evaluation done')
