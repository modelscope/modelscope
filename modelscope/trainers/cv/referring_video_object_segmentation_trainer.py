# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch

from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.constant import ModeKeys


@TRAINERS.register_module(
    module_name=Trainers.referring_video_object_segmentation)
class ReferringVideoObjectSegmentationTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.set_postprocessor(self.cfg.dataset.name)
        self.train_data_collator = self.train_dataset.collator
        self.eval_data_collator = self.eval_dataset.collator

        device_name = kwargs.get('device', 'gpu')
        self.model.set_device(self.device, device_name)

    def train(self, *args, **kwargs):
        self.model.criterion.train()
        super().train(*args, **kwargs)

    def evaluate(self, checkpoint_path=None):
        if checkpoint_path is not None:
            from modelscope.trainers.hooks import LoadCheckpointHook
            LoadCheckpointHook.load_checkpoint(checkpoint_path, self)
        self.model.eval()
        self._mode = ModeKeys.EVAL
        if self.eval_dataset is None:
            self.eval_dataloader = self.get_eval_data_loader()
        else:
            self.eval_dataloader = self._build_dataloader_with_dataset(
                self.eval_dataset,
                dist=self._dist,
                seed=self._seed,
                collate_fn=self.eval_data_collator,
                **self.cfg.evaluation.get('dataloader', {}))
        self.data_loader = self.eval_dataloader

        from modelscope.metrics import build_metric
        ann_file = self.eval_dataset.ann_file
        metric_classes = []
        for metric in self.metrics:
            metric.update({'ann_file': ann_file})
            metric_classes.append(build_metric(metric))

        for m in metric_classes:
            m.trainer = self

        metric_values = self.evaluation_loop(self.eval_dataloader,
                                             metric_classes)

        self._metric_values = metric_values
        return metric_values

    def prediction_step(self, model, inputs):
        pass
