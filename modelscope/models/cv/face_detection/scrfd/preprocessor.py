# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import numpy as np
from PIL import Image

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Fields, ModeKeys


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.object_detection_scrfd)
class SCRFDPreprocessor(Preprocessor):

    def __init__(self, model_dir: str = None, mode: str = ModeKeys.INFERENCE):
        """The base constructor for all the fill-mask preprocessors.

        Args:
            model_dir (str): model directory to initialize some resource
            mode: The mode for the preprocessor.
        """
        super().__init__(mode)
        pre_pipeline = [
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=False),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
        from mmdet.datasets.pipelines import Compose
        self.pipeline = Compose(pre_pipeline)

    def __call__(self, data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """process the raw input data
        Args:
            data (str or dict):  image path or data dict containing following info:
                filename, ori_filename, img, img_shape, ori_shape, img_fields
                example:
                    ```python
                    {
                        "filename": "xxx.jpg"
                        "ori_filename": "xxx.jpg",
                        "img": np.ndarray,
                        "img_shape": (300, 300, 3)
                        "ori_shape": (300, 300, 3)
                        "img_fields": "img"
                    }
                    ```
        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, str):
            img = LoadImage.convert_to_ndarray(data)
            img = img.astype(np.float32)
            data_dict = {}
            data_dict['filename'] = ''
            data_dict['ori_filename'] = ''
            data_dict['img'] = img
            data_dict['img_shape'] = img.shape
            data_dict['ori_shape'] = img.shape
            data_dict['img_fields'] = ['img']
        elif isinstance(data, (np.ndarray, Image.Image)):
            if isinstance(data, Image.Image):
                data = LoadImage.convert_to_ndarray(data)

            data = data.astype(np.float32)
            data_dict = {}
            data_dict['filename'] = ''
            data_dict['ori_filename'] = ''
            data_dict['img'] = data
            data_dict['img_shape'] = data.shape
            data_dict['ori_shape'] = data.shape
            data_dict['img_fields'] = ['img']

        elif isinstance(data, dict):
            data_dict = data

        return self.pipeline(data_dict)
