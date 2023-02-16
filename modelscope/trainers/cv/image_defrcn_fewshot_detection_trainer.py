# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/engine/defaults.py
# https://github.com/er-muyue/DeFRCN/blob/main/tools/model_surgery.py

import os
from collections import OrderedDict
from typing import Callable, Optional, Union

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.build import (build_detection_test_loader,
                                   build_detection_train_loader)
from detectron2.engine import SimpleTrainer, hooks
from detectron2.evaluation import (DatasetEvaluator, DatasetEvaluators,
                                   verify_results)
from detectron2.evaluation.testing import print_csv_format
from detectron2.solver.build import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.models.cv.image_defrcn_fewshot.evaluation.evaluator import \
    inference_on_dataset
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.logger import get_logger


class DefaultTrainer(SimpleTrainer):
    """
    Trainer inherit from detectron2 SimpleTrainer, use detectron2 framework to train.
    """

    def __init__(self, model, cfg):
        """ initialize model with cfg

        Args:
            model: torch.nn.Module
            cfg: model config with detectron2 format
        """
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

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == 'coco':
            from modelscope.models.cv.image_defrcn_fewshot.evaluation.coco_evaluation import COCOEvaluator
            evaluator_list.append(
                COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == 'pascal_voc':
            from modelscope.models.cv.image_defrcn_fewshot.evaluation.pascal_voc_evaluation import PascalVOCEvaluator
            return PascalVOCEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                'no Evaluator for the dataset {} with the type {}'.format(
                    dataset_name, evaluator_type))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = get_logger()

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(
                cfg.DATASETS.TEST) == len(evaluators), '{} != {}'.format(
                    len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = build_detection_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        'No evaluator found. Use `DefaultTrainer.test(evaluators=)`, '
                        'or implement its `build_evaluator` method.')
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator,
                                             cfg)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), 'Evaluator must return a dict on the main process. Got {} instead.'.format(
                    results_i)
                logger.info('Evaluation results for {} in csv format:'.format(
                    dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


@TRAINERS.register_module(module_name=Trainers.image_fewshot_detection)
class ImageDefrcnFewshotTrainer(BaseTrainer):
    """
    Defrcn model trainer, used to train base model and fsod/gfsod model.
    And model_surgery function used to modify model outputs arch, to train fsod & gfsod.
    """

    def __init__(self,
                 model: Optional[Union[TorchModel, nn.Module, str]] = None,
                 cfg_file: Optional[str] = None,
                 arg_parse_fn: Optional[Callable] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 seed: int = 0,
                 cfg_modify_fn: Optional[Callable] = None,
                 **kwargs):
        """ init model

        Args:
            model:  used to init model
            cfg_file: model config file path, if none, will init from model_dir by ModelFile.CONFIGURATION
            arg_parse_fn: Same as ``parse_fn`` in :obj:`Config.to_args`.
            model_revision: model version. Use latest if model_revision is none.
            seed: random seed
            cfg_modify_fn: modify model config, should be callable
        """

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

        kwargs['_cfg_dict'] = self.cfg

        if isinstance(model, (TorchModel, nn.Module)):
            self.model = model
        else:
            self.model = self.build_model(**kwargs)

        self.model_cfg = self.model.get_model_cfg()

        if not os.path.exists(self.model_cfg.OUTPUT_DIR):
            os.makedirs(self.model_cfg.OUTPUT_DIR)

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
        model = Model.from_pretrained(
            model_name_or_path=self.model_dir, cfg_dict=self.cfg, **kwargs)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    @classmethod
    def model_surgery(cls,
                      src_path,
                      save_dir,
                      data_type='pascal_voc',
                      method='remove',
                      params_name=[
                          'model.roi_heads.box_predictor.cls_score',
                          'model.roi_heads.box_predictor.bbox_pred'
                      ]):

        from modelscope.models.cv.image_defrcn_fewshot.utils.model_surgery_op import model_surgery as _model_surgery
        _model_surgery(src_path, save_dir, data_type, method, params_name)
