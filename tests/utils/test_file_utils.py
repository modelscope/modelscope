import os
import tempfile
import unittest

from modelscope.utils.file_utils import copytree_py37


class TestCopyTree(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp2_dir = tempfile.TemporaryDirectory()
        dir_path = self.tmp_dir.name
        print(f'self.tmp_dir: {self.tmp_dir.name}')
        print(f'self.tmp_dir2: {self.tmp2_dir.name}')
        fnames = ['1.py', '2.py', '3.py']
        self.folders = ['.', 'a', 'b', 'c']
        folder_dirs = [
            os.path.join(dir_path, folder) for folder in self.folders
        ]
        for folder in folder_dirs:
            os.makedirs(folder, exist_ok=True)
            for fname in fnames:
                fpath = os.path.join(folder, fname)
                with open(fpath, 'w') as f:
                    f.write('hello world')

        for folder in folder_dirs:
            print(f'folder: {os.listdir(folder)}')

    def tearDown(self):
        self.tmp_dir.cleanup()
        self.tmp2_dir.cleanup()

    def test_copytree_py37_exist_ok_true(self):
        copytree_py37(
            self.tmp_dir.name, self.tmp2_dir.name, dirs_exist_ok=True)
        copytree_py37(
            self.tmp_dir.name, self.tmp2_dir.name, dirs_exist_ok=True)
        dir_path = self.tmp2_dir.name
        new_folder_dirs = [
            os.path.join(dir_path, folder) for folder in self.folders
        ]
        for folder in new_folder_dirs:
            print(f'new_folder: {os.listdir(folder)}')


if __name__ == '__main__':
    unittest.main()
