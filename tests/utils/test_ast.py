# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from modelscope.utils.ast_utils import (FILES_MTIME_KEY, INDEX_KEY, MD5_KEY,
                                        MODELSCOPE_PATH_KEY, REQUIREMENT_KEY,
                                        VERSION_KEY, AstScanning,
                                        FilesAstScanning,
                                        generate_ast_template,
                                        load_from_prebuilt, load_index)

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
        astScaner = AstScanning()
        pipeline_file = os.path.join(MODELSCOPE_PATH, 'pipelines', 'nlp',
                                     'fill_mask_pipeline.py')
        output = astScaner.generate_ast(pipeline_file)
        self.assertTrue(output['imports'] is not None)
        self.assertTrue(output['from_imports'] is not None)
        self.assertTrue(output['decorators'] is not None)
        imports, from_imports, decorators = output['imports'], output[
            'from_imports'], output['decorators']
        self.assertIsInstance(imports, dict)
        self.assertIsInstance(from_imports, dict)
        self.assertIsInstance(decorators, list)
        self.assertListEqual(list(set(imports.keys()) - set(['numpy'])), [])
        self.assertEqual(len(from_imports.keys()), 8)
        self.assertTrue(from_imports['modelscope.metainfo'] is not None)
        self.assertEqual(from_imports['modelscope.metainfo'], ['Pipelines'])
        self.assertEqual(decorators,
                         [('PIPELINES', 'fill-mask', 'fill-mask'),
                          ('PIPELINES', 'fill-mask', 'fill-mask-ponet')])

    def test_files_scaning_method(self):
        fileScaner = FilesAstScanning()
        # case of pass in files directly
        pipeline_file = os.path.join(MODELSCOPE_PATH, 'pipelines', 'nlp',
                                     'fill_mask_pipeline.py')
        file_list = [pipeline_file]
        output = fileScaner.get_files_scan_results(file_list)
        self.assertTrue(output[INDEX_KEY] is not None)
        self.assertTrue(output[REQUIREMENT_KEY] is not None)
        index, requirements = output[INDEX_KEY], output[REQUIREMENT_KEY]
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
        fileScaner = FilesAstScanning()
        # create first file
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write('This is the new test!')

        md5_1, mtime_1 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        md5_2, mtime_2 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertEqual(md5_1, md5_2)
        self.assertEqual(mtime_1, mtime_2)
        self.assertIsInstance(mtime_1, dict)
        self.assertEqual(list(mtime_1.keys()), [self.test_file])
        self.assertEqual(mtime_1[self.test_file], mtime_2[self.test_file])

        time.sleep(2)
        # case of revise
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write('test again')
        md5_3, mtime_3 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertNotEqual(md5_1, md5_3)
        self.assertNotEqual(mtime_1[self.test_file], mtime_3[self.test_file])

        # case of create
        self.test_file_new = os.path.join(self.tmp_dir, 'test_1.py')
        time.sleep(2)
        with open(self.test_file_new, 'w', encoding='utf-8') as f:
            f.write('test again')
        md5_4, mtime_4 = fileScaner.files_mtime_md5(self.tmp_dir, [])
        self.assertNotEqual(md5_1, md5_4)
        self.assertNotEqual(md5_3, md5_4)
        self.assertEqual(
            set(mtime_4.keys()) - set([self.test_file, self.test_file_new]),
            set())

    def test_load_index_method(self):
        # test full indexing case
        output = load_index()
        self.assertTrue(output[INDEX_KEY] is not None)
        self.assertTrue(output[REQUIREMENT_KEY] is not None)
        index, requirements = output[INDEX_KEY], output[REQUIREMENT_KEY]
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
        self.assertIsInstance(output[MD5_KEY], str)
        self.assertIsInstance(output[MODELSCOPE_PATH_KEY], str)
        self.assertIsInstance(output[VERSION_KEY], str)
        self.assertIsInstance(output[FILES_MTIME_KEY], dict)

        # generate ast_template
        file_path = os.path.join(self.tmp_dir, 'index_file.py')
        index = generate_ast_template(file_path=file_path, force_rebuild=False)
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(output, index)
        index_from_prebuilt = load_from_prebuilt(file_path)
        self.assertEqual(index, index_from_prebuilt)

    def test_update_load_index_method(self):
        file_number = 20
        file_list = []
        for i in range(file_number):
            filename = os.path.join(self.tmp_dir, f'test_{i}.py')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('import os')
            file_list.append(filename)

        index_file = 'ast_indexer_1'

        index = load_index(
            file_list=file_list,
            indexer_file_dir=self.tmp_dir,
            indexer_file=index_file)
        self.assertEqual(len(index[FILES_MTIME_KEY]), file_number)

        # no changing case, time should be less than original
        index = load_index(
            file_list=file_list,
            indexer_file_dir=self.tmp_dir,
            indexer_file=index_file)
        self.assertEqual(len(index[FILES_MTIME_KEY]), file_number)

        # modify case
        with open(file_list[0], 'a', encoding='utf-8') as f:
            f.write('\n\nimport typing')
        file_list.append(filename)
        index = load_index(
            file_list=file_list,
            indexer_file_dir=self.tmp_dir,
            indexer_file=index_file)
        self.assertEqual(len(index[FILES_MTIME_KEY]), file_number)

        # adding new file, time should be less than original
        test_file_new_2 = os.path.join(self.tmp_dir, 'test_new.py')
        with open(test_file_new_2, 'w', encoding='utf-8') as f:
            f.write('import os')
        file_list.append(test_file_new_2)

        index = load_index(
            file_list=file_list,
            indexer_file_dir=self.tmp_dir,
            indexer_file=index_file)
        self.assertEqual(len(index[FILES_MTIME_KEY]), file_number + 1)

        # deleting one file, time should be less than original
        file_list.pop()
        index = load_index(
            file_list=file_list,
            indexer_file_dir=self.tmp_dir,
            indexer_file=index_file)
        self.assertEqual(len(index[FILES_MTIME_KEY]), file_number)


if __name__ == '__main__':
    unittest.main()
