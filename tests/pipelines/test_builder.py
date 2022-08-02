# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest
from asyncio import Task
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL

from modelscope.fileio import io
from modelscope.models.base import Model
from modelscope.pipelines import Pipeline, pipeline
from modelscope.pipelines.builder import PIPELINES, add_default_pipeline_info
from modelscope.utils.constant import (ConfigFields, Frameworks, ModelFile,
                                       Tasks)
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import default_group

logger = get_logger()


@PIPELINES.register_module(
    group_key=Tasks.image_classification, module_name='custom_single_model')
class CustomSingleModelPipeline(Pipeline):

    def __init__(self,
                 config_file: str = None,
                 model: List[Union[str, Model]] = None,
                 preprocessor=None,
                 **kwargs):
        super().__init__(config_file, model, preprocessor, **kwargs)
        assert isinstance(model, str), 'model is not str'
        print(model)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super().postprocess(inputs)


@PIPELINES.register_module(
    group_key=Tasks.image_classification, module_name='model1_model2')
class CustomMultiModelPipeline(Pipeline):

    def __init__(self,
                 config_file: str = None,
                 model: List[Union[str, Model]] = None,
                 preprocessor=None,
                 **kwargs):
        super().__init__(config_file, model, preprocessor, **kwargs)
        assert isinstance(model, list), 'model is not list'
        for m in model:
            assert isinstance(m, str), 'submodel is not str'
            print(m)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super().postprocess(inputs)


class PipelineInterfaceTest(unittest.TestCase):

    def prepare_dir(self, dirname, pipeline_name):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cfg_file = os.path.join(dirname, ModelFile.CONFIGURATION)
        cfg = {
            ConfigFields.framework: Frameworks.torch,
            ConfigFields.task: Tasks.image_classification,
            ConfigFields.pipeline: {
                'type': pipeline_name,
            }
        }
        io.dump(cfg, cfg_file)

    def setUp(self) -> None:
        self.prepare_dir('/tmp/custom_single_model', 'custom_single_model')
        self.prepare_dir('/tmp/model1', 'model1_model2')
        self.prepare_dir('/tmp/model2', 'model1_model2')

    def test_single_model(self):
        pipe = pipeline(
            Tasks.image_classification, model='/tmp/custom_single_model')
        assert isinstance(pipe, CustomSingleModelPipeline)

    def test_multi_model(self):
        pipe = pipeline(
            Tasks.image_classification, model=['/tmp/model1', '/tmp/model2'])
        assert isinstance(pipe, CustomMultiModelPipeline)


if __name__ == '__main__':
    unittest.main()
