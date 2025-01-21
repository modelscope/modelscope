# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.ocr_detection import OCRDetection
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.logger import get_logger
from .ocr_utils import cal_width, nms_python, rboxes_to_polygons

logger = get_logger()

# constant
RBOX_DIM = 5
OFFSET_DIM = 6
WORD_POLYGON_DIM = 8
OFFSET_VARIANCE = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
TF_NODE_THRESHOLD = 0.4
TF_LINK_THRESHOLD = 0.6


@PIPELINES.register_module(
    Tasks.ocr_detection, module_name=Pipelines.ocr_detection)
class OCRDetectionPipeline(Pipeline):
    """ OCR Detection Pipeline.

    Example:

    ```python
    >>> from modelscope.pipelines import pipeline

    >>> ocr_detection = pipeline('ocr-detection', model='damo/cv_resnet18_ocr-detection-line-level_damo')
    >>> result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')

        {'polygons': array([[220,  14, 780,  14, 780,  64, 220,  64],
       [196, 369, 604, 370, 604, 425, 196, 425],
       [ 21, 730, 425, 731, 425, 787,  21, 786],
       [421, 731, 782, 731, 782, 789, 421, 789],
       [  0, 121, 109,   0, 147,  35,  26, 159],
       [697, 160, 773, 160, 773, 197, 697, 198],
       [547, 205, 623, 205, 623, 244, 547, 244],
       [548, 161, 623, 161, 623, 199, 547, 199],
       [698, 206, 772, 206, 772, 244, 698, 244]])}
    ```
    note:
    model = damo/cv_resnet18_ocr-detection-line-level_damo, for general text line detection, based on SegLink++.
    model = damo/cv_resnet18_ocr-detection-word-level_damo, for general text word detection, based on SegLink++.
    model = damo/cv_resnet50_ocr-detection-vlpt, for toaltext dataset, based on VLPT_pretrained DBNet.
    model = damo/cv_resnet18_ocr-detection-db-line-level_damo, for general text line detection, based on DBNet.

    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a OCR detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        assert isinstance(model, str), 'model must be a single str'
        super().__init__(model=model, **kwargs)
        logger.info(f'loading model from dir {model}')
        cfgs = Config.from_file(os.path.join(model, ModelFile.CONFIGURATION))
        if hasattr(cfgs, 'model') and hasattr(cfgs.model, 'model_type'):
            self.model_type = cfgs.model.model_type
        else:
            self.model_type = 'SegLink++'

        if self.model_type == 'DBNet':
            self.ocr_detector = self.model.to(self.device)
            self.ocr_detector.eval()
            logger.info('loading model done')
        else:
            # for model seglink++
            import tensorflow as tf

            if tf.__version__ >= '2.0':
                tf = tf.compat.v1
            tf.compat.v1.disable_eager_execution()

            tf.app.flags.DEFINE_float('node_threshold', TF_NODE_THRESHOLD,
                                      'Confidence threshold for nodes')
            tf.app.flags.DEFINE_float('link_threshold', TF_LINK_THRESHOLD,
                                      'Confidence threshold for links')
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

                        from .ocr_utils import SegLinkDetector, combine_segments_python, decode_segments_links_python
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

    def __call__(self, input, **kwargs):
        """
        Detect text instance in the text image.

        Args:
            input (`Image`):
                The pipeline handles three types of images:

                - A string containing an HTTP link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly

                The pipeline currently supports single image input.

        Return:
            An array of contour polygons of detected N text instances in image,
            every row is [x1, y1, x2, y2, x3, y3, x4, y4, ...].
        """
        return super().__call__(input, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if self.model_type == 'DBNet':
            result = self.preprocessor(input)
            return result
        else:
            # for model seglink++
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
        if self.model_type == 'DBNet':
            outputs = self.ocr_detector(input)
            return outputs
        else:
            with self._graph.as_default():
                with self._session.as_default():
                    feed_dict = {self.input_images: input['img']}
                    sess_outputs = self._session.run(
                        self.output, feed_dict=feed_dict)
                    return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_type == 'DBNet':
            result = {OutputKeys.POLYGONS: inputs['det_polygons']}
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
