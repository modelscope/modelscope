import math
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL
import tensorflow as tf
import tf_slim as slim

from modelscope.pipelines.base import Input
from modelscope.preprocessors import load_image
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from ..base import Pipeline
from ..builder import PIPELINES
from .ocr_utils import model_resnet_mutex_v4_linewithchar, ops, utils

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
tf.compat.v1.disable_eager_execution()

logger = get_logger()

# constant
RBOX_DIM = 5
OFFSET_DIM = 6
WORD_POLYGON_DIM = 8
OFFSET_VARIANCE = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


@PIPELINES.register_module(
    Tasks.ocr_detection, module_name=Tasks.ocr_detection)
class OCRDetectionPipeline(Pipeline):

    def __init__(self, model: str):
        super().__init__()
        # print("self.model: ", self.model)
        model_path = osp.join(
            osp.join(self.model, ModelFile.TF_CHECKPOINT_FOLDER),
            'checkpoint-80000')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(config=config)
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            dtype=tf.int64,
            trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        self.input_images = tf.placeholder(
            tf.float32, shape=[1, 1024, 1024, 3], name='input_images')
        self.output = {}

        # detector
        detector = model_resnet_mutex_v4_linewithchar.SegLinkDetector()
        all_maps = detector.build_model(self.input_images, is_training=False)

        # decode local predictions
        all_nodes, all_links, all_reg = [], [], []
        for i, maps in enumerate(all_maps):
            cls_maps, lnk_maps, reg_maps = maps[0], maps[1], maps[2]
            reg_maps = tf.multiply(reg_maps, OFFSET_VARIANCE)

            cls_prob = tf.nn.softmax(tf.reshape(cls_maps, [-1, 2]))
            cls_pos_prob = cls_prob[:, model_resnet_mutex_v4_linewithchar.
                                    POS_LABEL]
            cls_pos_prob_maps = tf.reshape(cls_pos_prob,
                                           tf.shape(cls_maps)[:3])
            node_labels = tf.cast(
                tf.greater_equal(cls_pos_prob_maps, 0.4), tf.int32)

            lnk_prob_pos = tf.nn.softmax(tf.reshape(lnk_maps, [-1, 4])[:, :2])
            lnk_pos_prob_pos = lnk_prob_pos[:,
                                            model_resnet_mutex_v4_linewithchar.
                                            POS_LABEL]
            lnk_shape = tf.shape(lnk_maps)
            lnk_pos_prob_maps = tf.reshape(
                lnk_pos_prob_pos,
                [lnk_shape[0], lnk_shape[1], lnk_shape[2], -1])
            link_labels = tf.cast(
                tf.greater_equal(lnk_pos_prob_maps, 0.6), tf.int32)

            all_nodes.append(node_labels)
            all_links.append(link_labels)
            all_reg.append(reg_maps)

        # decode segments and links
        image_size = tf.shape(self.input_images)[1:3]
        segments, group_indices, segment_counts, _ = ops.decode_segments_links(
            image_size,
            all_nodes,
            all_links,
            all_reg,
            anchor_sizes=list(detector.anchor_sizes))

        # combine segments
        combined_rboxes, combined_counts = ops.combine_segments_filter(
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
        if isinstance(input, str):
            img = np.array(load_image(input))
        elif isinstance(input, PIL.Image.Image):
            img = np.array(input.convert('RGB'))
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = input[:, :, ::-1]  # in rgb order
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        h, w, c = img.shape
        img_pad = np.zeros((max(h, w), max(h, w), 3), dtype=np.float32)
        img_pad[:h, :w, :] = img

        resize_size = 1024
        img_pad_resize = cv2.resize(img_pad, (resize_size, resize_size))
        img_pad_resize = cv2.cvtColor(img_pad_resize, cv2.COLOR_RGB2BGR)
        img_pad_resize = img_pad_resize - np.array([123.68, 116.78, 103.94],
                                                   dtype=np.float32)

        resize_size = tf.stack([resize_size, resize_size])
        orig_size = tf.stack([max(h, w), max(h, w)])
        self.output['orig_size'] = orig_size
        self.output['resize_size'] = resize_size
        self.output['orig_image'] = tf.convert_to_tensor(img)

        result = {'img': np.expand_dims(img_pad_resize, axis=0)}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_images: input['img']}
            sess_outputs = self._session.run(self.output, feed_dict=feed_dict)
            return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        rboxes = inputs['combined_rboxes'][0]
        count = inputs['combined_counts'][0]
        rboxes = rboxes[:count, :]

        # convert rboxes to polygons and find its coordinates on the original image
        orig_h, orig_w = inputs['orig_size']
        resize_h, resize_w = inputs['resize_size']
        polygons = utils.rboxes_to_polygons(rboxes)
        scale_y = float(orig_h) / float(resize_h)
        scale_x = float(orig_w) / float(resize_w)

        # confine polygons inside image
        polygons[:, ::2] = np.maximum(
            0, np.minimum(polygons[:, ::2] * scale_x, orig_w - 1))
        polygons[:, 1::2] = np.maximum(
            0, np.minimum(polygons[:, 1::2] * scale_y, orig_h - 1))
        polygons = np.round(polygons).astype(np.int32)

        # nms
        dt_n9 = [o + [utils.cal_width(o)] for o in polygons.tolist()]
        dt_nms = utils.nms_python(dt_n9)
        dt_polygons = np.array([o[:8] for o in dt_nms])

        # visualize dt_polygons in image
        # image = inputs['orig_image']
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = utils.draw_polygons(image, dt_polygons)
        # cv2.imwrite('./ocr_detection_visu.jpg', image)

        result = {'det_polygons': dt_polygons}
        return result
