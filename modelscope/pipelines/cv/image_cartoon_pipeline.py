# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import cv2
import numpy as np
import tensorflow as tf

from modelscope.metainfo import Pipelines
from modelscope.models.cv.cartoon import (FaceAna, get_f5p,
                                          get_reference_facial_points,
                                          padTo16x, resize_size,
                                          warp_and_crop_face)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ...utils.device import device_placement

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_portrait_stylization,
    module_name=Pipelines.person_image_cartoon)
class ImageCartoonPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image cartoon pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.facer = FaceAna(self.model)
        with tf.Graph().as_default():
            self.sess_anime_head = self.load_sess(
                os.path.join(self.model, 'cartoon_h.pb'), 'model_anime_head')
            self.sess_anime_bg = self.load_sess(
                os.path.join(self.model, 'cartoon_bg.pb'), 'model_anime_bg')

        self.box_width = 288
        global_mask = cv2.imread(os.path.join(self.model, 'alpha.jpg'))
        global_mask = cv2.resize(
            global_mask, (self.box_width, self.box_width),
            interpolation=cv2.INTER_AREA)
        self.global_mask = cv2.cvtColor(
            global_mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def load_sess(self, model_path, name):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        logger.info(f'loading model from {model_path}')
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name=name)
            sess.run(tf.global_variables_initializer())
        logger.info(f'load model {model_path} done.')
        return sess

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(float)
        result = {'img': img}
        return result

    def detect_face(self, img):
        src_h, src_w, _ = img.shape
        boxes, landmarks, _ = self.facer.run(img)
        if boxes.shape[0] == 0:
            return None
        else:
            return landmarks

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        img = input['img'].astype(np.uint8)
        ori_h, ori_w, _ = img.shape
        img = resize_size(img, size=720)

        img_brg = img[:, :, ::-1]

        # background process
        pad_bg, pad_h, pad_w = padTo16x(img_brg)

        bg_res = self.sess_anime_bg.run(
            self.sess_anime_bg.graph.get_tensor_by_name(
                'model_anime_bg/output_image:0'),
            feed_dict={'model_anime_bg/input_image:0': pad_bg})
        res = bg_res[:pad_h, :pad_w, :]

        landmarks = self.detect_face(img)
        if landmarks is None:
            print('No face detected!')
            return {OutputKeys.OUTPUT_IMG: res}

        for landmark in landmarks:
            # get facial 5 points
            f5p = get_f5p(landmark, img_brg)

            # face alignment
            head_img, trans_inv = warp_and_crop_face(
                img,
                f5p,
                ratio=0.75,
                reference_pts=get_reference_facial_points(default_square=True),
                crop_size=(self.box_width, self.box_width),
                return_trans_inv=True)

            # head process
            head_res = self.sess_anime_head.run(
                self.sess_anime_head.graph.get_tensor_by_name(
                    'model_anime_head/output_image:0'),
                feed_dict={
                    'model_anime_head/input_image:0': head_img[:, :, ::-1]
                })

            # merge head and background
            head_trans_inv = cv2.warpAffine(
                head_res,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))

            mask = self.global_mask
            mask_trans_inv = cv2.warpAffine(
                mask,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))
            mask_trans_inv = np.expand_dims(mask_trans_inv, 2)

            res = mask_trans_inv * head_trans_inv + (1 - mask_trans_inv) * res

        res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)

        return {OutputKeys.OUTPUT_IMG: res}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
