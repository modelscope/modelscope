# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import torch

from modelscope.models.cv.tinynas_detection.damo.apis.detector_inference import \
    inference
from modelscope.models.cv.tinynas_detection.damo.detectors.detector import \
    build_local_model
from modelscope.msdatasets.task_datasets.damoyolo import (build_dataloader,
                                                          build_dataset)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Evaluater:

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = cfg.miscs.output_dir
        self.exp_name = cfg.miscs.exp_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.ckpt = torch.load(
            self.cfg.test.checkpoint_path, map_location=self.device)
        self.model = build_local_model(self.cfg, self.device)
        self.model.load_state_dict(self.ckpt['model'])
        self.val_loader = self.get_data_loader(self.cfg, False)

    def get_data_loader(self, cfg, distributed=False):

        val_dataset = build_dataset(
            cfg,
            cfg.dataset.val_image_dir,
            cfg.dataset.val_ann,
            is_train=False)

        val_loader = build_dataloader(
            val_dataset,
            cfg.test.augment,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.miscs.num_workers,
            is_train=False,
            size_div=32,
            distributed=distributed)

        return val_loader

    def evaluate(self):

        output_folder = os.path.join(self.output_dir, self.exp_name,
                                     'inference')
        for data_loader_val in self.val_loader:
            inference(
                self.model,
                data_loader_val,
                device=self.device,
                output_folder=output_folder,
            )
