# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging as logger
import os
import os.path as osp
import time

import cv2
import json
import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .data.data_augment import ValTransform
from .exp import get_exp_by_name
from .utils import postprocess, timestamp_format


@MODELS.register_module(
    group_key=Tasks.video_object_detection,
    module_name=Models.realtime_video_object_detection)
class RealtimeVideoDetector(TorchModel):

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
        self.current_buffer = None
        self.label_mapping = self.config['labels']

    def inference(self, img):
        with torch.no_grad():
            outputs, self.current_buffer = self.model(
                img, buffer=self.current_buffer, mode='on_pipe')
        return outputs

    def forward(self, inputs):
        return self.inference_video(inputs)

    def preprocess(self, img):
        img = LoadImage.convert_to_ndarray(img)
        height, width = img.shape[:2]
        self.ratio = min(self.test_size[0] / img.shape[0],
                         self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        # Video decoding and preprocessing automatically are not supported by Pipeline/Model
        # Sending preprocessed video frame tensor to GPU buffer self-adaptively
        if next(self.model.parameters()).is_cuda:
            img = img.to(next(self.model.parameters()).device)
        return img

    def postprocess(self, input):
        outputs = postprocess(
            input,
            self.num_classes,
            self.confthre,
            self.nmsthre,
            class_agnostic=True)

        if len(outputs) == 1 and (outputs[0] is not None):
            bboxes = outputs[0][:, 0:4].cpu().numpy() / self.ratio
            scores = outputs[0][:, 5].cpu().numpy()
            labels = outputs[0][:, 6].cpu().int().numpy()
            pred_label_names = []
            for lab in labels:
                pred_label_names.append(self.label_mapping[lab])
        else:
            bboxes = np.asarray([])
            scores = np.asarray([])
            pred_label_names = np.asarray([])

        return bboxes, scores, pred_label_names

    def inference_video(self, v_path):
        outputs = []
        desc = 'Detecting video: {}'.format(v_path)
        for frame_idx, (frame, result) in enumerate(
                tqdm(self.inference_video_iter(v_path), desc=desc)):
            result = result + (timestamp_format(seconds=frame_idx
                                                / self.fps), )
            outputs.append(result)

        return outputs

    def inference_video_iter(self, v_path):
        capture = cv2.VideoCapture(v_path)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            output = self.preprocess(frame)
            output = self.inference(output)
            output = self.postprocess(output)
            yield frame, output
