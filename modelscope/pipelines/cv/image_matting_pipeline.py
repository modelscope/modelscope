# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.portrait_matting, module_name=Pipelines.portrait_matting)
class ImageMattingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image matting pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        import tensorflow as tf
        if tf.__version__ >= '2.0':
            tf = tf.compat.v1
        model_path = osp.join(self.model, ModelFile.TF_GRAPH_FILE)

        with device_placement(self.framework, self.device_name):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)
            with self._session.as_default():
                logger.info(f'loading model from {model_path}')
                with tf.gfile.FastGFile(model_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    self.output = self._session.graph.get_tensor_by_name(
                        'output_png:0')
                    self.input_name = 'input_image:0'
                logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_name: input['img']}
            output_img = self._session.run(self.output, feed_dict=feed_dict)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2BGRA)
            return {OutputKeys.OUTPUT_IMG: output_img}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
