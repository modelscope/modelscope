# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/engine/defaults.py
# https://github.com/er-muyue/DeFRCN/blob/main/tools/model_surgery.py

import os
from typing import Callable, Optional, Union

import torch
from detectron2.engine import SimpleTrainer, hooks
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.utils import comm
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.logger import get_logger


class DefaultTrainer(SimpleTrainer):

    def __init__(self, model, cfg):

        from collections import OrderedDict
        from fvcore.nn.precise_bn import get_bn_modules
        from torch.nn.parallel import DistributedDataParallel

        from detectron2.data.build import build_detection_train_loader, build_detection_test_loader
        from detectron2.solver.build import build_optimizer, build_lr_scheduler
        from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
        from detectron2.utils.logger import setup_logger

        setup_logger()

        optimizer = build_optimizer(cfg, model)
        data_loader = build_detection_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True)
        super().__init__(model, data_loader, optimizer)

        self.scheduler = build_lr_scheduler(cfg, optimizer)

        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.checkpointer.resume_or_load(
                self.cfg.MODEL.WEIGHTS, resume=resume).get('iteration', -1)
            + 1)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(self.checkpointer,
                                           cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, 'metrics.json')),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if hasattr(self, '_last_eval_results') and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.data import MetadataCatalog

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == 'coco':
            from detectron2.evaluation import COCOEvaluator
            evaluator_list.append(
                COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == 'pascal_voc':
            from detectron2.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                'no Evaluator for the dataset {} with the type {}'.format(
                    dataset_name, evaluator_type))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        from detectron2.engine.defaults import DefaultTrainer as _DefaultTrainer
        _DefaultTrainer.build_evaluator = cls.build_evaluator

        return _DefaultTrainer.test(cfg, model, evaluators)


@TRAINERS.register_module(module_name=Trainers.image_fewshot_detection)
class ImageDefrcnFewshotTrainer(BaseTrainer):

    def __init__(self,
                 model: Optional[Union[TorchModel, nn.Module, str]] = None,
                 cfg_file: Optional[str] = None,
                 arg_parse_fn: Optional[Callable] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 seed: int = 0,
                 cfg_modify_fn: Optional[Callable] = None,
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

        if cfg_modify_fn is not None:
            self.cfg = cfg_modify_fn(self.cfg)

        self.logger = get_logger(log_level=self.cfg.get('log_level', 'INFO'))

        if isinstance(model, (TorchModel, nn.Module)):
            self.model = model
        else:
            self.model = self.build_model(**kwargs)

        self.model_cfg = self.model.get_model_cfg()

        if 'datasets_train' in kwargs:
            self.model_cfg.merge_from_list(
                ['DATASETS.TRAIN', kwargs['datasets_train']])
        if 'datasets_test' in kwargs:
            self.model_cfg.merge_from_list(
                ['DATASETS.TEST', kwargs['datasets_test']])
        if 'work_dir' in kwargs:
            self.model_cfg.merge_from_list(['OUTPUT_DIR', kwargs['work_dir']])

        if not os.path.exists(self.model_cfg.OUTPUT_DIR):
            os.makedirs(self.model_cfg.OUTPUT_DIR)

        self.model_cfg.freeze()

        self.data_dir = kwargs.get('data_dir', None)
        self.data_type = kwargs.get('data_type', 'pascal_voc')

        self.register_data(self.data_type, self.data_dir)

        self.trainer = DefaultTrainer(self.model, self.model_cfg)

    def train(self, *args, **kwargs):
        self.trainer.resume_or_load()
        self.trainer.train()

    def evaluate(self, checkpoint_path: str, *args, **kwargs):
        from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer

        DetectionCheckpointer(
            self.model,
            save_dir=self.model_cfg.OUTPUT_DIR).resume_or_load(checkpoint_path)
        metric_values = DefaultTrainer.test(self.model_cfg, self.model)
        return metric_values

    def build_model(self, *args, **kwargs) -> Union[nn.Module, TorchModel]:
        model = Model.from_pretrained(self.model_dir, **kwargs)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    @classmethod
    def register_data(cls, data_type='pascal_voc', data_dir=None):

        if data_type == 'pascal_voc':
            from modelscope.models.cv.image_defrcn_fewshot.utils.voc_register import register_all_voc
            if data_dir:
                register_all_voc(data_dir)
            else:
                register_all_voc()
        else:
            raise NotImplementedError(
                'no {} dataset was registered'.format(data_type))

    @classmethod
    def model_surgery(cls,
                      src_path,
                      save_dir,
                      data_type='pascal_voc',
                      method='remove'):

        assert method in ['remove',
                          'randinit'], '{} not implemented'.format(method)

        def _surgery(param_name, is_weight, tar_size, ckpt):
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            pretrained_weight = ckpt['model'][weight_name]
            prev_cls = pretrained_weight.size(0)
            if 'cls_score' in param_name:
                prev_cls -= 1
            if is_weight:
                feat_size = pretrained_weight.size(1)
                new_weight = torch.rand((tar_size, feat_size))
                torch.nn.init.normal_(new_weight, 0, 0.01)
            else:
                new_weight = torch.zeros(tar_size)

            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
            if 'cls_score' in param_name:
                new_weight[-1] = pretrained_weight[-1]  # bg class
            ckpt['model'][weight_name] = new_weight

        if data_type == 'pascal_voc':
            TAR_SIZE = 20
            params_name = [
                'model.roi_heads.box_predictor.cls_score',
                'model.roi_heads.box_predictor.bbox_pred'
            ]

            save_name = 'model_reset_' + ('remove' if method == 'remove' else
                                          'surgery') + '.pth'
            save_path = os.path.join(save_dir, save_name)
            os.makedirs(save_dir, exist_ok=True)

            ckpt = torch.load(src_path)

            if 'scheduler' in ckpt:
                del ckpt['scheduler']
            if 'optimizer' in ckpt:
                del ckpt['optimizer']
            if 'iteration' in ckpt:
                ckpt['iteration'] = 0

            if method == 'remove':
                for param_name in params_name:
                    del ckpt['model'][param_name + '.weight']
                    if param_name + '.bias' in ckpt['model']:
                        del ckpt['model'][param_name + '.bias']
            else:
                tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
                for idx, (param_name,
                          tar_size) in enumerate(zip(params_name, tar_sizes)):
                    _surgery(param_name, True, tar_size, ckpt)
                    _surgery(param_name, False, tar_size, ckpt)

            torch.save(ckpt, save_path)
        else:
            NotImplementedError(
                '{} dataset does not supported'.format(data_type))
