# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import Callable, Dict, Optional

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import SimpleTrainer, hooks, launch
from detectron2.engine.defaults import create_ddp_model, default_writers
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.solver import LRMultiplier, WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from fvcore.common.param_scheduler import CosineParamScheduler

from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.metrics.action_detection_evaluator import DetEvaluator
from modelscope.models.cv.action_detection.modules.action_detection_pytorch import \
    build_action_detection_model
from modelscope.preprocessors.cv.action_detection_mapper import VideoDetMapper
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import Invoke, ModelFile, Tasks


@TRAINERS.register_module(module_name=Trainers.action_detection)
class ActionDetectionTrainer(BaseTrainer):

    def __init__(self,
                 model_id,
                 train_dataset,
                 test_dataset,
                 cfg_file: str = None,
                 cfg_modify_fn: Optional[Callable] = None,
                 *args,
                 **kwargs):
        model_cache_dir = self.get_or_download_model_dir(model_id)
        if cfg_file is None:
            cfg_file = os.path.join(model_cache_dir, ModelFile.CONFIGURATION)

        super().__init__(cfg_file)
        if cfg_modify_fn is not None:
            self.cfg = cfg_modify_fn(self.cfg)
        self.total_step = self.cfg.train.max_iter
        self.warmup_step = self.cfg.train.lr_scheduler['warmup_step']
        self.lr = self.cfg.train.optimizer.lr
        self.total_batch_size = max(
            1, self.cfg.train.num_gpus
        ) * self.cfg.train.dataloader['batch_size_per_gpu']
        self.num_classes = len(self.cfg.train.classes_id_map)
        self.resume = kwargs.get('resume', False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.pretrained_model = kwargs.get(
            'pretrained_model',
            osp.join(model_cache_dir, ModelFile.TORCH_MODEL_FILE))

    def start(self, output_dir):
        if comm.is_main_process() and output_dir:
            PathManager.mkdirs(output_dir)
            self.cfg.dump(osp.join(output_dir, 'config.py'))
        rank = comm.get_rank()
        setup_logger(output_dir, distributed_rank=rank, name='fvcore')
        logger = setup_logger(output_dir, distributed_rank=rank)
        logger.info('Rank of current process: {}. World size: {}'.format(
            rank, comm.get_world_size()))

    def train(self, *args, **kwargs):
        if self.cfg.train.num_gpus <= 1:
            self.do_train()
        else:
            launch(
                self.do_train,
                self.cfg.train.num_gpus,
                1,
                machine_rank=0,
                dist_url='auto',
                args=())

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        if self.cfg.train.num_gpus <= 1:
            self.do_train(just_eval=True, checkpoint_path=checkpoint_path)
        else:
            launch(
                self.do_train,
                self.cfg.train.num_gpus,
                1,
                machine_rank=0,
                dist_url='auto',
                args=(True, checkpoint_path))

    def do_train(
        self,
        just_eval=False,
        checkpoint_path=None,
    ):
        self.start(self.cfg.train.work_dir)
        model = build_action_detection_model(num_classes=self.num_classes)
        if self.cfg.train.num_gpus > 0:
            model.cuda()
            model = create_ddp_model(model, broadcast_buffers=False)
        if just_eval:
            checkpoint = DetectionCheckpointer(model)
            checkpoint.load(checkpoint_path)
            result = self.do_test(model)
            return result
        optim = torch.optim.AdamW(
            params=get_default_optimizer_params(model, base_lr=self.lr),
            lr=self.lr,
            weight_decay=0.1)
        lr_scheduler = LRMultiplier(
            optim,
            WarmupParamScheduler(
                CosineParamScheduler(1, 1e-3),
                warmup_factor=0,
                warmup_length=self.warmup_step / self.total_step),
            max_iter=self.total_step,
        )
        train_loader = build_detection_train_loader(
            self.train_dataset,
            mapper=VideoDetMapper(
                self.cfg.train.classes_id_map, is_train=True),
            total_batch_size=self.total_batch_size,
            num_workers=self.cfg.train.dataloader.workers_per_gpu)
        trainer = SimpleTrainer(model, train_loader, optim)
        checkpointer = DetectionCheckpointer(
            model, self.cfg.train.work_dir, trainer=trainer)

        trainer.register_hooks([
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=lr_scheduler),
            hooks.PeriodicCheckpointer(
                checkpointer, period=self.cfg.train.checkpoint_interval)
            if comm.is_main_process() else None,
            hooks.EvalHook(
                eval_period=self.cfg.evaluation.interval,
                eval_function=lambda: self.do_test(model)),
            hooks.PeriodicWriter(
                default_writers(checkpointer.save_dir, self.total_step),
                period=20) if comm.is_main_process() else None,
        ])
        checkpointer.resume_or_load(self.pretrained_model, resume=False)
        if self.resume:
            checkpointer.resume_or_load(resume=self.resume)
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
        trainer.train(start_iter, self.total_step)

    def do_test(self, model):
        evaluator = DetEvaluator(
            list(self.cfg.train.classes_id_map.keys()),
            self.cfg.train.work_dir,
            distributed=self.cfg.train.num_gpus > 1)
        test_loader = build_detection_test_loader(
            self.test_dataset,
            mapper=VideoDetMapper(
                self.cfg.train.classes_id_map, is_train=False),
            num_workers=self.cfg.evaluation.dataloader.workers_per_gpu)

        result = inference_on_dataset(model, test_loader, evaluator)
        print_csv_format(result)
        return result

    def get_or_download_model_dir(self, model, model_revision=None):
        if os.path.exists(model):
            model_cache_dir = model if os.path.isdir(
                model) else os.path.dirname(model)
            check_local_model_is_latest(
                model_cache_dir, user_agent={Invoke.KEY: Invoke.LOCAL_TRAINER})
        else:
            model_cache_dir = snapshot_download(
                model,
                revision=model_revision,
                user_agent={Invoke.KEY: Invoke.TRAINER})
        return model_cache_dir
