# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
import os.path as osp
import time
from typing import Callable, Dict, Optional

from modelscope.metainfo import Trainers
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS


@TRAINERS.register_module(module_name=Trainers.face_detection_scrfd)
class FaceDetectionScrfdTrainer(BaseTrainer):

    def __init__(self,
                 cfg_file: str,
                 cfg_modify_fn: Optional[Callable] = None,
                 *args,
                 **kwargs):
        """ High-level finetune api for SCRFD.

        Args:
            cfg_file: Path to configuration file.
            cfg_modify_fn: An input fn which is used to modify the cfg read out of the file.
        """
        import mmcv
        from mmcv.runner import get_dist_info, init_dist
        from mmcv.utils import get_git_hash
        from mmdet.utils import collect_env, get_root_logger
        from mmdet.apis import set_random_seed
        from mmdet.models import build_detector
        from mmdet.datasets import build_dataset
        from mmdet import __version__
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets import RetinaFaceDataset
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets.pipelines import DefaultFormatBundleV2
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets.pipelines import LoadAnnotationsV2
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets.pipelines import RotateV2
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets.pipelines import RandomSquareCrop
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.backbones import ResNetV1e
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.dense_heads import SCRFDHead
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.detectors import SCRFD
        super().__init__(cfg_file)
        cfg = self.cfg
        if 'work_dir' in kwargs:
            cfg.work_dir = kwargs['work_dir']
        else:
            # use config filename as default work_dir if work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(cfg_file))[0])
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        if 'resume_from' in kwargs:  # pretrain model for finetune
            cfg.resume_from = kwargs['resume_from']
        cfg.device = 'cuda'
        if 'gpu_ids' in kwargs:
            cfg.gpu_ids = kwargs['gpu_ids']
        else:
            cfg.gpu_ids = range(1)
        labelfile_name = kwargs.pop('labelfile_name', 'labelv2.txt')
        imgdir_name = kwargs.pop('imgdir_name', 'images/')
        if 'train_root' in kwargs:
            cfg.data.train.ann_file = kwargs['train_root'] + labelfile_name
            cfg.data.train.img_prefix = kwargs['train_root'] + imgdir_name
        if 'val_root' in kwargs:
            cfg.data.val.ann_file = kwargs['val_root'] + labelfile_name
            cfg.data.val.img_prefix = kwargs['val_root'] + imgdir_name
        if 'total_epochs' in kwargs:
            cfg.total_epochs = kwargs['total_epochs']
        if cfg_modify_fn is not None:
            cfg = cfg_modify_fn(cfg)
        if 'launcher' in kwargs:
            distributed = True
            init_dist(kwargs['launcher'], **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        else:
            distributed = False
        # no_validate=True will not evaluate checkpoint during training
        cfg.no_validate = kwargs.get('no_validate', False)
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
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
        if 'seed' in kwargs:
            cfg.seed = kwargs['seed']
            _deterministic = kwargs.get('deterministic', False)
            logger.info(f'Set random seed to {kwargs["seed"]}, '
                        f'deterministic: {_deterministic}')
            set_random_seed(kwargs['seed'], deterministic=_deterministic)
        else:
            cfg.seed = None
        meta['seed'] = cfg.seed
        meta['exp_name'] = osp.basename(cfg_file)

        model = build_detector(cfg.model)
        model.init_weights()
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES

        self.cfg = cfg
        self.datasets = datasets
        self.model = model
        self.distributed = distributed
        self.timestamp = timestamp
        self.meta = meta
        self.logger = logger

    def train(self, *args, **kwargs):
        from mmdet.apis import train_detector
        train_detector(
            self.model,
            self.datasets,
            self.cfg,
            distributed=self.distributed,
            validate=(not self.cfg.no_validate),
            timestamp=self.timestamp,
            meta=self.meta)

    def evaluate(self,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        cfg = self.cfg.evaluation
        logger.info(f'eval cfg {cfg}')
        logger.info(f'checkpoint_path {checkpoint_path}')
