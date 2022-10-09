# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from modelscope.metainfo import Trainers
from modelscope.models.nlp.space.model.generator import SpaceGenerator
from modelscope.models.nlp.space.model.model_base import SpaceModelBase
from modelscope.preprocessors.space.data_loader import \
    get_sequential_data_loader
from modelscope.preprocessors.space.fields.intent_field import \
    IntentBPETextField
from modelscope.preprocessors.space.preprocess import intent_preprocess
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp.space.trainer.intent_trainer import IntentTrainer
from modelscope.utils.config import Config
from modelscope.utils.logger import get_logger

PATH = None
logger = get_logger(PATH)


@TRAINERS.register_module(module_name=Trainers.dialog_intent_trainer)
class DialogIntentTrainer(BaseTrainer):

    def __init__(self,
                 cfg_file: Optional[str] = None,
                 cfg_modify_fn: Optional[Callable] = None,
                 *args,
                 **kwargs):
        super().__init__(os.path.join(kwargs['model_dir'], kwargs['cfg_name']))

        def to_tensor(array):
            """
            numpy array -> tensor
            """
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.cfg.use_gpu else array

        def setup_seed(seed):
            import random
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        self.cfg_modify_fn = cfg_modify_fn
        self.cfg = self.rebuild_config(self.cfg)

        setup_seed(self.cfg.Trainer.seed)

        # preprocess data
        intent_preprocess(self.cfg.Model.init_checkpoint, self.cfg)
        # set reader and evaluator
        bpe = IntentBPETextField(self.cfg.Model.init_checkpoint, self.cfg)

        self.cfg.Model.num_token_embeddings = bpe.vocab_size
        self.cfg.Model.num_turn_embeddings = bpe.max_ctx_turn + 1
        dataset_paths = [
            os.path.join(self.cfg.Dataset.data_dir,
                         self.cfg.Dataset.trigger_data)
        ]
        # set data and data status
        collate_fn = bpe.collate_fn_multi_turn
        self.train_label_loader = get_sequential_data_loader(
            batch_size=self.cfg.Trainer.batch_size_label,
            reader=bpe,
            hparams=self.cfg,
            data_paths=dataset_paths,
            collate_fn=collate_fn,
            data_type='train')
        self.valid_label_loader = get_sequential_data_loader(
            batch_size=self.cfg.Trainer.batch_size_label,
            reader=bpe,
            hparams=self.cfg,
            data_paths=dataset_paths,
            collate_fn=collate_fn,
            data_type='valid')
        self.test_label_loader = get_sequential_data_loader(
            batch_size=self.cfg.Trainer.batch_size_label,
            reader=bpe,
            hparams=self.cfg,
            data_paths=dataset_paths,
            collate_fn=collate_fn,
            data_type='test')

        # set generator
        generator = SpaceGenerator.create(self.cfg, reader=bpe)
        # construct model
        self.model = SpaceModelBase.create(
            self.cfg.Model.init_checkpoint,
            self.cfg,
            reader=bpe,
            generator=generator)

        import torch

        # multi-gpu
        if self.cfg.Trainer.gpu > 1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # construct trainer
        self.trainer = IntentTrainer(
            self.model, to_tensor, self.cfg, reader=bpe)
        num_batches = len(self.train_label_loader)
        self.trainer.set_optimizers(num_training_steps_per_epoch=num_batches)
        # load model, optimizer and lr_scheduler
        self.trainer.load()

    def rebuild_config(self, cfg: Config):
        if self.cfg_modify_fn is not None:
            return self.cfg_modify_fn(cfg)
        return cfg

    def train(self, *args, **kwargs):
        logger.info('Train')

        self.trainer.train(
            train_label_iter=self.train_label_loader,
            valid_label_iter=self.valid_label_loader)

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        logger.info('Evaluate')
        self.trainer.infer(
            data_iter=self.test_label_loader,
            ex_data_iter=self.train_label_loader)
