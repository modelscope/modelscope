# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging as logger
import os
import os.path as osp
import time

import cv2
import json
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .yolox.data.data_augment import ValTransform
from .yolox.exp import get_exp_by_name
from .yolox.utils import postprocess


@MODELS.register_module(
    group_key=Tasks.image_object_detection,
    module_name=Models.realtime_object_detection)
class RealtimeDetector(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        # model type
        self.exp = get_exp_by_name(self.config.model_type)

        # build model
        self.model = self.exp.get_model()
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        ckpt = torch.load(model_path, map_location='cpu')

        # load the model state dict
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        # params setting
        self.exp.num_classes = self.config.num_classes
        self.confthre = self.config.conf_thr
        self.num_classes = self.exp.num_classes
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.preproc = ValTransform(legacy=False)
        self.label_mapping = self.config['labels']

    def inference(self, img):
        with torch.no_grad():
            outputs = self.model(img)
        return outputs

    def forward(self, inputs):
        return self.inference(inputs)

    def preprocess(self, img):
        img = LoadImage.convert_to_ndarray(img)
        height, width = img.shape[:2]
        self.ratio = min(self.test_size[0] / img.shape[0],
                         self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        return img

    def postprocess(self, input):
        outputs = postprocess(
            input,
            self.num_classes,
            self.confthre,
            self.nmsthre,
            class_agnostic=True)

        if len(outputs) == 1:
            bboxes = outputs[0][:, 0:4].cpu().numpy() / self.ratio
            scores = outputs[0][:, 5].cpu().numpy()
            labels = outputs[0][:, 6].cpu().int().numpy()
            pred_label_names = []
            for lab in labels:
                pred_label_names.append(self.label_mapping[lab])

        return bboxes, scores, pred_label_names
