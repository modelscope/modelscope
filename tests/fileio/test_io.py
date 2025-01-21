# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import numpy as np

from modelscope.fileio.io import dump, dumps, load


class FileIOTest(unittest.TestCase):

    def test_format(self, format='json'):
        obj = [
            1, 2, 3, 'str', {
                'model': 'resnet'
            },
            np.array([[1, 2]], dtype=np.float16),
            np.array([[1, 2]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.float64),
            np.array([[1, 2]], dtype=np.int64), (1, 2)
        ]
        result_str = dumps(obj, format)
        temp_name = tempfile.gettempdir() + '/' + next(
            tempfile._get_candidate_names()) + '.' + format
        dump(obj, temp_name)
        obj_load = load(temp_name)

        self.assertEqual(len(obj), len(obj_load))
        for i, obj_i in enumerate(obj):
            if isinstance(obj_i, list):
                self.assertListEqual(obj_i, obj_load[i])
            elif isinstance(obj_i, np.ndarray):
                self.assertListEqual(obj_i.tolist(), obj_load[i].tolist())
            elif isinstance(obj_i, dict):
                self.assertDictEqual(obj_i, obj_load[i])
            else:
                self.assertEqual(obj_i, obj_load[i])

        with open(temp_name, 'r') as infile:
            self.assertEqual(result_str, infile.read())

        with self.assertRaises(TypeError):
            obj_load = load(temp_name + 's')

        with self.assertRaises(TypeError):
            dump(obj, temp_name + 's')

    def test_yaml(self):
        self.test_format('yaml')


if __name__ == '__main__':
    unittest.main()
