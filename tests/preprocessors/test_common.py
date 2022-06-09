# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.preprocessors import PREPROCESSORS, Compose, Preprocessor


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


if __name__ == '__main__':
    unittest.main()
