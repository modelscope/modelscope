# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from modelscope.utils.ast_utils import AstScaning, FilesAstScaning, load_index

p = Path(__file__)

MODELSCOPE_PATH = p.resolve().parents[2].joinpath('modelscope')


class AstScaningTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        self.test_file = os.path.join(self.tmp_dir, 'test.py')
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_ast_scaning_class(self):
        astScaner = AstScaning()
        pipeline_file = os.path.join(MODELSCOPE_PATH, 'pipelines', 'nlp',
                                     'sequence_classification_pipeline.py')
        output = astScaner.generate_ast(pipeline_file)
        self.assertTrue(output['imports'] is not None)
        self.assertTrue(output['from_imports'] is not None)
        self.assertTrue(output['decorators'] is not None)
        imports, from_imports, decorators = output['imports'], output[
            'from_imports'], output['decorators']
        self.assertIsInstance(imports, dict)
        self.assertIsInstance(from_imports, dict)
        self.assertIsInstance(decorators, list)
        self.assertListEqual(
            list(set(imports.keys()) - set(['typing', 'numpy'])), [])
        self.assertEqual(len(from_imports.keys()), 9)
        self.assertTrue(from_imports['modelscope.metainfo'] is not None)
        self.assertEqual(from_imports['modelscope.metainfo'], ['Pipelines'])
        self.assertEqual(
            decorators,
            [('PIPELINES', 'text-classification', 'sentiment-analysis')])

    def test_files_scaning_method(self):
        fileScaner = FilesAstScaning()
        output = fileScaner.get_files_scan_results()
        self.assertTrue(output['index'] is not None)
        self.assertTrue(output['requirements'] is not None)
        index, requirements = output['index'], output['requirements']
        self.assertIsInstance(index, dict)
        self.assertIsInstance(requirements, dict)
        self.assertIsInstance(list(index.keys())[0], tuple)
        index_0 = list(index.keys())[0]
        self.assertIsInstance(index[index_0], dict)
        self.assertTrue(index[index_0]['imports'] is not None)
        self.assertIsInstance(index[index_0]['imports'], list)
        self.assertTrue(index[index_0]['module'] is not None)
        self.assertIsInstance(index[index_0]['module'], str)
        index_0 = list(requirements.keys())[0]
        self.assertIsInstance(requirements[index_0], list)

    def test_file_mtime_md5_method(self):
        fileScaner = FilesAstScaning()
        # create first file
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write('This is the new test!')

        md5_1 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        md5_2 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertEqual(md5_1, md5_2)
        time.sleep(2)
        # case of revise
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write('test again')
        md5_3 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertNotEqual(md5_1, md5_3)

        # case of create
        self.test_file_new = os.path.join(self.tmp_dir, 'test_1.py')
        time.sleep(2)
        with open(self.test_file_new, 'w', encoding='utf-8') as f:
            f.write('test again')
        md5_4 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertNotEqual(md5_1, md5_4)
        self.assertNotEqual(md5_3, md5_4)


if __name__ == '__main__':
    unittest.main()
