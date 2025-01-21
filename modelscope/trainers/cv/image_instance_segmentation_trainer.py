# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer


@TRAINERS.register_module(module_name=Trainers.image_instance_segmentation)
class ImageInstanceSegmentationTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, data):
        # we skip this func due to some special data type, e.g., BitmapMasks
        return data

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        metric_values = super().evaluate(*args, **kwargs)
        return metric_values

    def prediction_step(self, model, inputs):
        pass
