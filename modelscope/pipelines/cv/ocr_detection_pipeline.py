# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import tensorflow as tf
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.cv.ocr_utils.model_vlpt import VLPTModel
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.logger import get_logger
from .ocr_utils import (SegLinkDetector, cal_width, combine_segments_python,
                        decode_segments_links_python, nms_python,
                        polygons_from_bitmap, rboxes_to_polygons)

if tf.__version__ >= '2.0':
    import tf_slim as slim
else:
    from tensorflow.contrib import slim

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
tf.compat.v1.disable_eager_execution()

logger = get_logger()

# constant
RBOX_DIM = 5
OFFSET_DIM = 6
WORD_POLYGON_DIM = 8
OFFSET_VARIANCE = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('node_threshold', 0.4,
                          'Confidence threshold for nodes')
tf.app.flags.DEFINE_float('link_threshold', 0.6,
                          'Confidence threshold for links')


@PIPELINES.register_module(
    Tasks.ocr_detection, module_name=Pipelines.ocr_detection)
class OCRDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a OCR detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        if 'vlpt' in self.model:
            model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
            logger.info(f'loading model from {model_path}')

            self.thresh = 0.3
            self.image_short_side = 736
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.infer_model = VLPTModel().to(self.device)
            self.infer_model.eval()
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.infer_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.infer_model.load_state_dict(checkpoint)
        else:
            tf.reset_default_graph()
            model_path = osp.join(
                osp.join(self.model, ModelFile.TF_CHECKPOINT_FOLDER),
                'checkpoint-80000')
            self._graph = tf.get_default_graph()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)

            with self._graph.as_default():
                with device_placement(self.framework, self.device_name):
                    self.input_images = tf.placeholder(
                        tf.float32,
                        shape=[1, 1024, 1024, 3],
                        name='input_images')
                    self.output = {}

                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        global_step = tf.get_variable(
                            'global_step', [],
                            initializer=tf.constant_initializer(0),
                            dtype=tf.int64,
                            trainable=False)
                        variable_averages = tf.train.ExponentialMovingAverage(
                            0.997, global_step)

                        # detector
                        detector = SegLinkDetector()
                        all_maps = detector.build_model(
                            self.input_images, is_training=False)

                        # decode local predictions
                        all_nodes, all_links, all_reg = [], [], []
                        for i, maps in enumerate(all_maps):
                            cls_maps, lnk_maps, reg_maps = maps[0], maps[
                                1], maps[2]
                            reg_maps = tf.multiply(reg_maps, OFFSET_VARIANCE)

                            cls_prob = tf.nn.softmax(
                                tf.reshape(cls_maps, [-1, 2]))

                            lnk_prob_pos = tf.nn.softmax(
                                tf.reshape(lnk_maps, [-1, 4])[:, :2])
                            lnk_prob_mut = tf.nn.softmax(
                                tf.reshape(lnk_maps, [-1, 4])[:, 2:])
                            lnk_prob = tf.concat([lnk_prob_pos, lnk_prob_mut],
                                                 axis=1)

                            all_nodes.append(cls_prob)
                            all_links.append(lnk_prob)
                            all_reg.append(reg_maps)

                        # decode segments and links
                        image_size = tf.shape(self.input_images)[1:3]
                        segments, group_indices, segment_counts, _ = decode_segments_links_python(
                            image_size,
                            all_nodes,
                            all_links,
                            all_reg,
                            anchor_sizes=list(detector.anchor_sizes))

                        # combine segments
                        combined_rboxes, combined_counts = combine_segments_python(
                            segments, group_indices, segment_counts)
                        self.output['combined_rboxes'] = combined_rboxes
                        self.output['combined_counts'] = combined_counts

                    with self._session.as_default() as sess:
                        logger.info(f'loading model from {model_path}')
                        # load model
                        model_loader = tf.train.Saver(
                            variable_averages.variables_to_restore())
                        model_loader.restore(sess, model_path)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if 'vlpt' in self.model:
            img = LoadImage.convert_to_ndarray(input)[:, :, ::-1]

            height, width, _ = img.shape
            if height < width:
                new_height = self.image_short_side
                new_width = int(
                    math.ceil(new_height / height * width / 32) * 32)
            else:
                new_width = self.image_short_side
                new_height = int(
                    math.ceil(new_width / width * height / 32) * 32)
            resized_img = cv2.resize(img, (new_width, new_height))

            resized_img = resized_img - np.array([123.68, 116.78, 103.94],
                                                 dtype=np.float32)
            resized_img /= 255.
            resized_img = torch.from_numpy(resized_img).permute(
                2, 0, 1).float().unsqueeze(0)

            result = {'img': resized_img, 'org_shape': [height, width]}
            return result
        else:
            img = LoadImage.convert_to_ndarray(input)

            h, w, c = img.shape
            img_pad = np.zeros((max(h, w), max(h, w), 3), dtype=np.float32)
            img_pad[:h, :w, :] = img

            resize_size = 1024
            img_pad_resize = cv2.resize(img_pad, (resize_size, resize_size))
            img_pad_resize = cv2.cvtColor(img_pad_resize, cv2.COLOR_RGB2BGR)
            img_pad_resize = img_pad_resize - np.array(
                [123.68, 116.78, 103.94], dtype=np.float32)

            with self._graph.as_default():
                resize_size = tf.stack([resize_size, resize_size])
                orig_size = tf.stack([max(h, w), max(h, w)])
                self.output['orig_size'] = orig_size
                self.output['resize_size'] = resize_size

            result = {'img': np.expand_dims(img_pad_resize, axis=0)}
            return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if 'vlpt' in self.model:
            pred = self.infer_model(input['img'])
            return {'results': pred, 'org_shape': input['org_shape']}
        else:
            with self._graph.as_default():
                with self._session.as_default():
                    feed_dict = {self.input_images: input['img']}
                    sess_outputs = self._session.run(
                        self.output, feed_dict=feed_dict)
                    return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'vlpt' in self.model:
            pred = inputs['results'][0]
            height, width = inputs['org_shape']
            segmentation = pred > self.thresh

            boxes, scores = polygons_from_bitmap(pred, segmentation, width,
                                                 height)
            result = {OutputKeys.POLYGONS: np.array(boxes)}
            return result
        else:
            rboxes = inputs['combined_rboxes'][0]
            count = inputs['combined_counts'][0]
            if count == 0 or count < rboxes.shape[0]:
                raise Exception('modelscope error: No text detected')
            rboxes = rboxes[:count, :]

            # convert rboxes to polygons and find its coordinates on the original image
            orig_h, orig_w = inputs['orig_size']
            resize_h, resize_w = inputs['resize_size']
            polygons = rboxes_to_polygons(rboxes)
            scale_y = float(orig_h) / float(resize_h)
            scale_x = float(orig_w) / float(resize_w)

            # confine polygons inside image
            polygons[:, ::2] = np.maximum(
                0, np.minimum(polygons[:, ::2] * scale_x, orig_w - 1))
            polygons[:, 1::2] = np.maximum(
                0, np.minimum(polygons[:, 1::2] * scale_y, orig_h - 1))
            polygons = np.round(polygons).astype(np.int32)

            # nms
            dt_n9 = [o + [cal_width(o)] for o in polygons.tolist()]
            dt_nms = nms_python(dt_n9)
            dt_polygons = np.array([o[:8] for o in dt_nms])

            result = {OutputKeys.POLYGONS: dt_polygons}
            return result
