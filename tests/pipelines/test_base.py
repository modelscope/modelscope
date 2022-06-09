# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL

from maas_lib.pipelines import Pipeline, pipeline
from maas_lib.pipelines.builder import PIPELINES
from maas_lib.pipelines.default import add_default_pipeline_info
from maas_lib.utils.constant import Tasks
from maas_lib.utils.logger import get_logger
from maas_lib.utils.registry import default_group

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

        @PIPELINES.register_module(
            group_key=Tasks.image_tagging, module_name='custom-image')
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
                if not isinstance(input, PIL.Image.Image):
                    from maas_lib.preprocessors import load_image
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
                outputs['resize_image'] = np.array(new_image)
                outputs['dummy_result'] = 'dummy_result'
                return outputs

            def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                return inputs

        self.assertTrue('custom-image' in PIPELINES.modules[default_group])
        add_default_pipeline_info(Tasks.image_tagging, 'custom-image')
        pipe = pipeline(pipeline_name='custom-image')
        pipe2 = pipeline(Tasks.image_tagging)
        self.assertTrue(type(pipe) is type(pipe2))

        img_url = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.' \
                  'aliyuncs.com/data/test/images/image1.jpg'
        output = pipe(img_url)
        self.assertEqual(output['filename'], img_url)
        self.assertEqual(output['resize_image'].shape, (318, 512, 3))
        self.assertEqual(output['dummy_result'], 'dummy_result')

        outputs = pipe([img_url for i in range(4)])
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out['filename'], img_url)
            self.assertEqual(out['resize_image'].shape, (318, 512, 3))
            self.assertEqual(out['dummy_result'], 'dummy_result')


if __name__ == '__main__':
    unittest.main()
