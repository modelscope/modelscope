# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import torch

from modelscope.preprocessors import (PREPROCESSORS, Compose, Filter,
                                      Preprocessor, ToTensor)


class ComposeTest(unittest.TestCase):

    def test_compose(self):

        @PREPROCESSORS.register_module()
        class Tmp1(Preprocessor):

            def __call__(self, input):
                input['tmp1'] = 'tmp1'
                return input

        @PREPROCESSORS.register_module()
        class Tmp2(Preprocessor):

            def __call__(self, input):
                input['tmp2'] = 'tmp2'
                return input

        pipeline = [
            dict(type='Tmp1'),
            dict(type='Tmp2'),
        ]
        trans = Compose(pipeline)

        input = {}
        output = trans(input)
        self.assertEqual(output['tmp1'], 'tmp1')
        self.assertEqual(output['tmp2'], 'tmp2')


class ToTensorTest(unittest.TestCase):

    def test_totensor(self):
        to_tensor_op = ToTensor(keys=['img'])
        inputs = {'img': [1, 2, 3], 'label': 1, 'path': 'test.jpg'}
        inputs = to_tensor_op(inputs)
        self.assertIsInstance(inputs['img'], torch.Tensor)
        self.assertEqual(inputs['label'], 1)
        self.assertEqual(inputs['path'], 'test.jpg')


class FilterTest(unittest.TestCase):

    def test_filter(self):
        filter_op = Filter(reserved_keys=['img', 'label'])
        inputs = {'img': [1, 2, 3], 'label': 1, 'path': 'test.jpg'}
        inputs = filter_op(inputs)
        self.assertIn('img', inputs)
        self.assertIn('label', inputs)
        self.assertNotIn('path', inputs)


if __name__ == '__main__':
    unittest.main()
