# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import Pipeline, pipeline
from modelscope.pipelines.builder import PIPELINES, add_default_pipeline_info
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import default_group

logger = get_logger()

Input = Union[str, 'PIL.Image', 'numpy.ndarray']


class CustomPipelineTest(unittest.TestCase):

    def test_abstract(self):

        @PIPELINES.register_module()
        class CustomPipeline1(Pipeline):

            def __init__(self,
                         config_file: str = None,
                         model=None,
                         preprocessor=None,
                         **kwargs):
                super().__init__(config_file, model, preprocessor, **kwargs)

        with self.assertRaises(TypeError):
            CustomPipeline1()

    def test_custom(self):
        dummy_task = 'dummy-task'

        @PIPELINES.register_module(
            group_key=dummy_task, module_name='custom-image')
        class CustomImagePipeline(Pipeline):

            def __init__(self,
                         config_file: str = None,
                         model=None,
                         preprocessor=None,
                         **kwargs):
                super().__init__(config_file, model, preprocessor, **kwargs)

            def preprocess(self, input: Union[str,
                                              'PIL.Image']) -> Dict[str, Any]:
                """ Provide default implementation based on preprocess_cfg and user can reimplement it

                """
                if not isinstance(input, Image.Image):
                    from modelscope.preprocessors import load_image
                    data_dict = {'img': load_image(input), 'url': input}
                else:
                    data_dict = {'img': input}
                return data_dict

            def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """ Provide default implementation using self.model and user can reimplement it
                """
                outputs = {}
                if 'url' in inputs:
                    outputs['filename'] = inputs['url']
                img = inputs['img']
                new_image = img.resize((img.width // 2, img.height // 2))
                outputs[OutputKeys.OUTPUT_IMG] = np.array(new_image)
                return outputs

            def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                return inputs

        self.assertTrue('custom-image' in PIPELINES.modules[dummy_task])
        add_default_pipeline_info(dummy_task, 'custom-image', overwrite=True)
        pipe = pipeline(task=dummy_task, pipeline_name='custom-image')
        pipe2 = pipeline(dummy_task)
        self.assertTrue(type(pipe) is type(pipe2))

        img_url = 'data/test/images/dogs.jpg'
        output = pipe(img_url)
        self.assertEqual(output['filename'], img_url)
        self.assertEqual(output[OutputKeys.OUTPUT_IMG].shape, (318, 512, 3))

        outputs = pipe([img_url for i in range(4)])
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out['filename'], img_url)
            self.assertEqual(out[OutputKeys.OUTPUT_IMG].shape, (318, 512, 3))


if __name__ == '__main__':
    unittest.main()
