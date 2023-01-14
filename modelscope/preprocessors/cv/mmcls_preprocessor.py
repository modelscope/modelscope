# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import numpy as np
from numpy import ndarray

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.hub import read_config
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.image_classification_mmcv_preprocessor)
class ImageClassificationMmcvPreprocessor(Preprocessor):

    def __init__(self, model_dir, **kwargs):
        """Preprocess the image.

        What this preprocessor will do:
        1. Remove the `LoadImageFromFile` preprocessor(which will be called in the pipeline).
        2. Compose and instantiate other preprocessors configured in the file.
        3. Call the sub preprocessors one by one.

        This preprocessor supports two types of configuration:
        1. The mmcv config file, configured in a `config.py`
        2. The maas config file, configured in a `configuration.json`
        By default, if the `config.py` exists, the preprocessor will use the mmcv config file.

        Args:
            model_dir (str): The model dir to build the preprocessor from.
        """

        import mmcv
        from mmcls.datasets.pipelines import Compose
        from modelscope.models.cv.image_classification.utils import preprocess_transform
        super().__init__(**kwargs)

        self.config_type = 'ms_config'
        mm_config = os.path.join(model_dir, 'config.py')
        if os.path.exists(mm_config):
            cfg = mmcv.Config.fromfile(mm_config)
            cfg.model.pretrained = None
            config_type = 'mmcv_config'
        else:
            cfg = read_config(model_dir)
            cfg.model.mm_model.pretrained = None
            config_type = 'ms_config'

        if config_type == 'mmcv_config':
            if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                cfg.data.test.pipeline.pop(0)
            self.preprocessors = Compose(cfg.data.test.pipeline)
        else:
            if cfg.preprocessor.val[0]['type'] == 'LoadImageFromFile':
                cfg.preprocessor.val.pop(0)
            data_pipeline = preprocess_transform(cfg.preprocessor.val)
            self.preprocessors = Compose(data_pipeline)

    @type_assert(object, object)
    def __call__(self, data: np.ndarray) -> Dict[str, ndarray]:
        data = dict(img=data)
        data = self.preprocessors(data)
        return data
