# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

from modelscope.fileio import io
from modelscope.outputs import OutputKeys
from modelscope.pipelines import Pipeline, pipeline
from modelscope.pipelines.builder import PIPELINES, add_default_pipeline_info
from modelscope.utils.constant import (ConfigFields, Frameworks, ModelFile,
                                       Tasks)
from modelscope.utils.logger import get_logger

logger = get_logger()

Input = Union[str, 'PIL.Image', 'numpy.ndarray']


class CustomPipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = '/tmp/custom-image'
        self.prepare_dir(self.model_dir, 'custom-image')

    def prepare_dir(self, dirname, pipeline_name):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cfg_file = os.path.join(dirname, ModelFile.CONFIGURATION)
        cfg = {
            ConfigFields.framework: 'dummy',
            ConfigFields.task: 'dummy-task',
            ConfigFields.pipeline: {
                'type': pipeline_name,
            }
        }
        io.dump(cfg, cfg_file)

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

    def test_batch(self):
        import torch

        dummy_task = 'dummy-task'
        dummy_module = 'custom-batch'

        @PIPELINES.register_module(
            group_key=dummy_task, module_name=dummy_module)
        class CustomBatchPipeline(Pipeline):

            def __init__(self,
                         config_file: str = None,
                         model=None,
                         preprocessor=None,
                         **kwargs):
                super().__init__(config_file, model, preprocessor, **kwargs)
                self._postprocess_inputs = None

            def _batch(self, sample_list):
                sample_batch = {'img': [], 'url': []}
                for sample in sample_list:
                    sample_batch['img'].append(sample['img'])
                    sample_batch['url'].append(sample['url'])

                sample_batch['img'] = torch.concat(sample_batch['img'])
                return sample_batch

            def preprocess(self, input: Union[str,
                                              'PIL.Image']) -> Dict[str, Any]:
                """ Provide default implementation based on preprocess_cfg and user can reimplement it

                """
                if not isinstance(input, Image.Image):
                    from modelscope.preprocessors import load_image
                    image = load_image(input)
                    resized_img = torch.from_numpy(
                        np.array(image.resize((640, 640))))
                    unsqueezed_img = torch.unsqueeze(resized_img, 0)
                    data_dict = {'img': unsqueezed_img, 'url': input}
                else:
                    data_dict = {'img': input}
                return data_dict

            def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """ Provide default implementation using self.model and user can reimplement it
                """
                inputs['img'] += 1
                return inputs

            def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                if self._postprocess_inputs is None:
                    self._postprocess_inputs = inputs
                else:
                    self._check_postprocess_input(inputs)
                inputs['url'] += 'dummy_end'
                return inputs

            def _check_postprocess_input(self, current_input: Dict[str, Any]):
                for key in current_input:
                    if isinstance(current_input[key], torch.Tensor):
                        assert len(current_input[key].shape) == len(
                            self._postprocess_inputs[key].shape)

        self.assertTrue(dummy_module in PIPELINES.modules[dummy_task])
        add_default_pipeline_info(dummy_task, dummy_module, overwrite=True)
        pipe = pipeline(
            task=dummy_task, pipeline_name=dummy_module, model=self.model_dir)

        img_url = 'data/test/images/dogs.jpg'
        pipe(img_url)
        output = pipe([img_url for _ in range(9)], batch_size=2)
        for out in output:
            self.assertEqual(out['url'], img_url + 'dummy_end')
            self.assertEqual(out['img'].shape, (1, 640, 640, 3))

        pipe_nocollate = pipeline(
            task=dummy_task,
            pipeline_name=dummy_module,
            model=self.model_dir,
            auto_collate=False)

        img_url = 'data/test/images/dogs.jpg'
        output = pipe_nocollate([img_url for _ in range(9)], batch_size=2)
        for out in output:
            self.assertEqual(out['url'], img_url + 'dummy_end')
            self.assertEqual(out['img'].shape, (1, 640, 640, 3))

    def test_chat_task(self):
        dummy_module = 'dummy_module'

        @PIPELINES.register_module(
            group_key=Tasks.chat, module_name=dummy_module)
        class CustomChat(Pipeline):

            def __init__(self,
                         config_file: str = None,
                         model=None,
                         preprocessor=None,
                         **kwargs):

                def f(x):
                    return x

                preprocessor = f
                super().__init__(config_file, model, preprocessor, **kwargs)

            def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """ Provide default implementation using self.model and user can reimplement it
                """
                return inputs

            def postprocess(self, out, **kwargs):
                return {'message': {'role': 'assistant', 'content': 'xxx'}}

        pipe = pipeline(
            task=Tasks.chat, pipeline_name=dummy_module, model=self.model_dir)
        pipe('text')
        inputs = {'text': 'aaa', 'history': [('dfd', 'fds')]}
        inputs = {
            'messages': [{
                'role': 'user',
                'content': 'dfd'
            }, {
                'role': 'assistant',
                'content': 'fds'
            }, {
                'role': 'user',
                'content': 'aaa'
            }]
        }
        pipe(inputs)

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
        pipe = pipeline(
            task=dummy_task,
            pipeline_name='custom-image',
            model=self.model_dir)
        pipe2 = pipeline(dummy_task, model=self.model_dir)
        self.assertTrue(type(pipe) is type(pipe2))

        img_url = 'data/test/images/dogs.jpg'
        output = pipe(img_url)
        self.assertEqual(output['filename'], img_url)
        self.assertEqual(output[OutputKeys.OUTPUT_IMG].shape, (598, 600, 3))

        outputs = pipe([img_url for i in range(4)])
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out['filename'], img_url)
            self.assertEqual(out[OutputKeys.OUTPUT_IMG].shape, (598, 600, 3))


if __name__ == '__main__':
    unittest.main()
