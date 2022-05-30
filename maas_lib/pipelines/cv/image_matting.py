import os.path as osp
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL
import tensorflow as tf
from cv2 import COLOR_GRAY2RGB

from maas_lib.pipelines.base import Input
from maas_lib.preprocessors import load_image
from maas_lib.utils.constant import Tasks
from maas_lib.utils.logger import get_logger
from ..base import Pipeline
from ..builder import PIPELINES

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_matting, module_name=Tasks.image_matting)
class ImageMatting(Pipeline):

    def __init__(self, model: str):
        super().__init__(model=model)
        model_path = osp.join(self.model, 'matting_person.pb')

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
        img = img.astype(np.float)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_name: input['img']}
            output_png = self._session.run(self.output, feed_dict=feed_dict)
            output_png = cv2.cvtColor(output_png, cv2.COLOR_RGBA2BGRA)
            return {'output_png': output_png}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
