# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

from maas_lib.fileio.io import dump, dumps, load


class FileIOTest(unittest.TestCase):

    def test_format(self, format='json'):
        obj = [1, 2, 3, 'str', {'model': 'resnet'}]
        result_str = dumps(obj, format)
        temp_name = tempfile.gettempdir() + '/' + next(
            tempfile._get_candidate_names()) + '.' + format
        dump(obj, temp_name)
        obj_load = load(temp_name)
        self.assertEqual(obj_load, obj)
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
