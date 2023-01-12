# Part of the implementation is borrowed and modified from mmclassification,
# publicly available at https://github.com/open-mmlab/mmclassification
import copy
import os
import os.path as osp
import time
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke, ModelFile
from modelscope.utils.logger import get_logger


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                val_dataset=None,
                timestamp=None,
                device=None,
                meta=None):
    import torch
    import warnings
    from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook,
                             build_optimizer, build_runner, get_dist_info)
    from mmcls.core import DistEvalHook, DistOptimizerHook, EvalHook
    from mmcls.datasets import build_dataloader
    from mmcls.utils import (wrap_distributed_model,
                             wrap_non_distributed_model)
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

    logger = get_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    sampler_cfg = cfg.train.get('sampler', None)

    data_loaders = [
        build_dataloader(
            ds,
            cfg.train.dataloader.batch_size_per_gpu,
            cfg.train.dataloader.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            seed=cfg.seed,
            sampler_cfg=sampler_cfg) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cpu':
            logger.warning(
                'The argument `device` is deprecated. To use cpu to train, '
                'please refers to https://mmclassification.readthedocs.io/en'
                '/latest/getting_started.html#train-a-model')
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                from mmcv import __version__, digit_version
                assert digit_version(__version__) >= (1, 4, 4), \
                    'To train with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'

    # build runner
    optimizer = build_optimizer(model, cfg.train.optimizer)

    if cfg.train.get('runner') is None:
        cfg.train.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.train.max_epochs
        }
        logger.warning(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.train.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.train.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.train.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.train.optimizer_config)
    else:
        optimizer_config = cfg.train.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.train.lr_config,
        optimizer_config,
        cfg.train.checkpoint_config,
        cfg.train.log_config,
        cfg.train.get('momentum_config', None),
        custom_hooks_config=cfg.train.get('custom_hooks', None))
    if distributed and cfg.train.runner['type'] == 'EpochBasedRunner':
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.evaluation.dataloader.batch_size_per_gpu,
            workers_per_gpu=cfg.evaluation.dataloader.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            round_up=True)
        eval_cfg = cfg.train.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.train.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.train.resume_from:
        runner.resume(cfg.train.resume_from, map_location='cpu')
    elif cfg.train.load_from:
        runner.load_checkpoint(cfg.train.load_from)

    cfg.train.workflow = [tuple(flow) for flow in cfg.train.workflow]
    runner.run(data_loaders, cfg.train.workflow)


@TRAINERS.register_module(module_name=Trainers.image_classification)
class ImageClassifitionTrainer(BaseTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Union[Callable, Dict[str,
                                                         Callable]]] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            seed: int = 0,
            cfg_modify_fn: Optional[Callable] = None,
            **kwargs):
        """ High-level finetune api for Image Classifition.

        Args:
            model: model id
            model_version: model version, default is None.
            cfg_modify_fn: An input fn which is used to modify the cfg read out of the file.
        """
        import torch
        import mmcv
        from modelscope.models.cv.image_classification.utils import get_ms_dataset_root, get_classes
        from mmcls.models import build_classifier
        from mmcv.runner import get_dist_info, init_dist
        from mmcls.apis import set_random_seed
        from mmcls.utils import collect_env
        from mmcv.utils import get_logger as mmcv_get_logger
        import modelscope.models.cv.image_classification.backbones

        self._seed = seed
        set_random_seed(self._seed)
        if isinstance(model, str):
            self.model_dir = self.get_or_download_model_dir(
                model, model_revision=model_revision)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is not from pretrained!'
            self.model_dir = os.path.dirname(cfg_file)

        super().__init__(cfg_file, arg_parse_fn)
        cfg = self.cfg

        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        else:
            self.work_dir = self.cfg.train.get('work_dir', './work_dir')
        mmcv.mkdir_or_exist(osp.abspath(self.work_dir))
        cfg.work_dir = self.work_dir

        # evaluate config seting
        self.eval_checkpoint_path = os.path.join(self.model_dir,
                                                 ModelFile.TORCH_MODEL_FILE)

        # train config seting
        if 'resume_from' in kwargs:
            cfg.train.resume_from = kwargs['resume_from']
        else:
            cfg.train.resume_from = cfg.train.get('resume_from', None)

        if 'load_from' in kwargs:
            cfg.train.load_from = kwargs['load_from']
        else:
            if cfg.train.get('resume_from', None) is None:
                cfg.train.load_from = os.path.join(self.model_dir,
                                                   ModelFile.TORCH_MODEL_FILE)

        if 'device' in kwargs:
            cfg.device = kwargs['device']
        else:
            cfg.device = cfg.get('device', 'cuda')

        if 'gpu_ids' in kwargs:
            cfg.gpu_ids = kwargs['gpu_ids'][0:1]
        else:
            cfg.gpu_ids = [0]

        if 'fp16' in kwargs:
            cfg.fp16 = None if kwargs['fp16'] is None else kwargs['fp16']
        else:
            cfg.fp16 = None

        # no_validate=True will not evaluate checkpoint during training
        cfg.no_validate = kwargs.get('no_validate', False)

        if cfg_modify_fn is not None:
            cfg = cfg_modify_fn(cfg)

        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train,
                'max_epochs'), 'max_epochs is missing in configuration file'
            self.max_epochs = self.cfg.train.max_epochs
        else:
            self.max_epochs = kwargs['max_epochs']
        cfg.train.max_epochs = self.max_epochs
        if cfg.train.get('runner', None) is not None:
            cfg.train.runner.max_epochs = self.max_epochs

        if 'launcher' in kwargs:
            distributed = True
            dist_params = kwargs['dist_params'] \
                if 'dist_params' in kwargs else {'backend': 'nccl'}
            init_dist(kwargs['launcher'], **dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = list(range(world_size))
        else:
            distributed = False

        # init the logger before other steps
        mmcv_get_logger('modelscope')  # set name of mmcv logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.work_dir, f'{timestamp}.log')
        logger = get_logger(log_file=log_file)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n'
                    + dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        cfg.seed = self._seed
        _deterministic = kwargs.get('deterministic', False)
        logger.info(f'Set random seed to {cfg.seed}, '
                    f'deterministic: {_deterministic}')
        set_random_seed(cfg.seed, deterministic=_deterministic)

        meta['seed'] = cfg.seed
        meta['exp_name'] = osp.basename(cfg_file)

        # dataset
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # set img_prefix for image data path in csv files.
        if cfg.dataset.get('data_prefix', None) is None:
            self.data_prefix = ''
        else:
            self.data_prefix = cfg.dataset.data_prefix

        # model
        model = build_classifier(self.cfg.model.mm_model)
        model.init_weights()

        self.cfg = cfg
        self.device = cfg.device
        self.cfg_file = cfg_file
        self.model = model
        self.distributed = distributed
        self.timestamp = timestamp
        self.meta = meta
        self.logger = logger

    def train(self, *args, **kwargs):
        from mmcls import __version__
        from modelscope.models.cv.image_classification.utils import get_ms_dataset_root, MmDataset, preprocess_transform
        from mmcls.utils import setup_multi_processes

        if self.train_dataset is None:
            raise ValueError(
                "Not found train dataset, please set the 'train_dataset' parameter!"
            )

        self.cfg.model.mm_model.pretrained = None

        # dump config
        self.cfg.dump(osp.join(self.work_dir, osp.basename(self.cfg_file)))

        # build the dataloader
        if self.cfg.dataset.classes is None:
            data_root = get_ms_dataset_root(self.train_dataset)
            classname_path = osp.join(data_root, 'classname.txt')
            classes = classname_path if osp.exists(classname_path) else None
        else:
            classes = self.cfg.dataset.classes

        datasets = [
            MmDataset(
                self.train_dataset,
                pipeline=self.cfg.preprocessor.train,
                classes=classes,
                data_prefix=self.data_prefix)
        ]

        if len(self.cfg.train.workflow) == 2:
            if self.eval_dataset is None:
                raise ValueError(
                    "Not found evaluate dataset, please set the 'eval_dataset' parameter!"
                )
            val_data_pipeline = self.cfg.preprocessor.train
            val_dataset = MmDataset(
                self.eval_dataset,
                pipeline=val_data_pipeline,
                classes=classes,
                data_prefix=self.data_prefix)
            datasets.append(val_dataset)

        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        self.meta.update(
            dict(
                mmcls_version=__version__,
                config=self.cfg.pretty_text,
                CLASSES=datasets[0].CLASSES))

        val_dataset = None
        if not self.cfg.no_validate:
            val_dataset = MmDataset(
                self.eval_dataset,
                pipeline=preprocess_transform(self.cfg.preprocessor.val),
                classes=classes,
                data_prefix=self.data_prefix)

        # add an attribute for visualization convenience
        train_model(
            self.model,
            datasets,
            self.cfg,
            distributed=self.distributed,
            val_dataset=val_dataset,
            timestamp=self.timestamp,
            device='cpu' if self.device == 'cpu' else 'cuda',
            meta=self.meta)

    def evaluate(self,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        import warnings
        import torch
        from modelscope.models.cv.image_classification.utils import (
            get_ms_dataset_root, MmDataset, preprocess_transform,
            get_trained_checkpoints_name)
        from mmcls.datasets import build_dataloader
        from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
        from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
        from mmcls.apis import multi_gpu_test, single_gpu_test
        from mmcls.utils import setup_multi_processes

        if self.eval_dataset is None:
            raise ValueError(
                "Not found evaluate dataset, please set the 'eval_dataset' parameter!"
            )

        self.cfg.model.mm_model.pretrained = None

        # build the dataloader
        if self.cfg.dataset.classes is None:
            data_root = get_ms_dataset_root(self.eval_dataset)
            classname_path = osp.join(data_root, 'classname.txt')
            classes = classname_path if osp.exists(classname_path) else None
        else:
            classes = self.cfg.dataset.classes
        dataset = MmDataset(
            self.eval_dataset,
            pipeline=preprocess_transform(self.cfg.preprocessor.val),
            classes=classes,
            data_prefix=self.data_prefix)
        # the extra round_up data will be removed during gpu/cpu collect
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=self.cfg.evaluation.dataloader.batch_size_per_gpu,
            workers_per_gpu=self.cfg.evaluation.dataloader.workers_per_gpu,
            dist=self.distributed,
            shuffle=False,
            round_up=True)

        model = copy.deepcopy(self.model)
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if checkpoint_path is None:
            trained_checkpoints = get_trained_checkpoints_name(self.work_dir)
            if trained_checkpoints is not None:
                checkpoint = load_checkpoint(
                    model,
                    os.path.join(self.work_dir, trained_checkpoints),
                    map_location='cpu')
            else:
                checkpoint = load_checkpoint(
                    model, self.eval_checkpoint_path, map_location='cpu')
        else:
            checkpoint = load_checkpoint(
                model, checkpoint_path, map_location='cpu')

        if 'CLASSES' in checkpoint.get('meta', {}):
            CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            self.logger.warning(
                'Class names are not saved in the checkpoint\'s '
                'meta data, use imagenet by default.')
            CLASSES = ImageNet.CLASSES

        if not self.distributed:
            if self.device == 'cpu':
                model = model.cpu()
            else:
                model = MMDataParallel(model, device_ids=self.cfg.gpu_ids)
                if not model.device_ids:
                    assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                        'To test with CPU, please confirm your mmcv version ' \
                        'is not lower than v1.4.4'
            model.CLASSES = CLASSES
            show_kwargs = {}
            outputs = single_gpu_test(model, data_loader, False, None,
                                      **show_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, None, True)

        rank, _ = get_dist_info()
        if rank == 0:
            results = {}
            logger = get_logger()
            metric_options = self.cfg.evaluation.get('metric_options', {})
            if 'topk' in metric_options.keys():
                metric_options['topk'] = tuple(metric_options['topk'])
            if self.cfg.evaluation.metrics:
                eval_results = dataset.evaluate(
                    results=outputs,
                    metric=self.cfg.evaluation.metrics,
                    metric_options=metric_options,
                    logger=logger)
                results.update(eval_results)

            return results

        return None
