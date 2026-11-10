import io
import os
import tarfile
import tempfile
import unittest
import warnings

from modelscope.utils.file_utils import copytree_py37, extract_tar_safely


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


class TestExtractTarSafely(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp_dir.name, 'dest')
        self.outside = os.path.join(self.tmp_dir.name, 'outside')
        os.makedirs(self.dest)
        os.makedirs(self.outside)
        self.archive = os.path.join(self.tmp_dir.name, 'archive.tar')

    def tearDown(self):
        self.tmp_dir.cleanup()

    @staticmethod
    def _add_file(tar, name, content=b'payload'):
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))

    def test_relative_traversal_is_rejected(self):
        escape_path = os.path.join(self.tmp_dir.name, 'escape.txt')
        with tarfile.open(self.archive, 'w') as tar:
            self._add_file(tar, os.path.join('..', 'outside', 'escape.txt'))
        with tarfile.open(self.archive) as tar:
            with self.assertRaises(tarfile.TarError):
                extract_tar_safely(tar, self.dest)
        self.assertFalse(os.path.exists(escape_path))

    def test_absolute_member_never_escapes(self):
        escape_path = os.path.join(self.tmp_dir.name, 'absolute.txt')
        with tarfile.open(self.archive, 'w') as tar:
            self._add_file(tar, escape_path)
        try:
            with tarfile.open(self.archive) as tar:
                extract_tar_safely(tar, self.dest)
        except tarfile.TarError:
            pass  # the member fallback rejects it outright
        self.assertFalse(os.path.exists(escape_path))

    def test_link_escape_is_rejected(self):
        with tarfile.open(self.archive, 'w') as tar:
            link = tarfile.TarInfo('link')
            link.type = tarfile.SYMTYPE
            link.linkname = self.outside
            tar.addfile(link)
            self._add_file(tar, os.path.join('link', 'escape.txt'))
        with tarfile.open(self.archive) as tar:
            with self.assertRaises(tarfile.TarError):
                extract_tar_safely(tar, self.dest)
        self.assertFalse(
            os.path.exists(os.path.join(self.outside, 'escape.txt')))

    def test_regular_archive_is_extracted(self):
        with tarfile.open(self.archive, 'w') as tar:
            self._add_file(tar, 'README.md', b'hello')
            self._add_file(tar, os.path.join('nested', 'weights.bin'),
                           b'\x00' * 16)
        with tarfile.open(self.archive) as tar:
            extract_tar_safely(tar, self.dest)
        self.assertTrue(os.path.isfile(os.path.join(self.dest, 'README.md')))
        self.assertTrue(
            os.path.isfile(os.path.join(self.dest, 'nested', 'weights.bin')))

    @unittest.skipUnless(
        hasattr(tarfile, 'data_filter'), 'PEP 706 filters not available')
    def test_member_fallback_blocks_traversal(self):
        data_filter = tarfile.data_filter
        del tarfile.data_filter
        try:
            with warnings.catch_warnings():
                # Simulating an interpreter without PEP 706 filters triggers
                # the extract() deprecation warning of newer versions.
                warnings.simplefilter('ignore', DeprecationWarning)
                with tarfile.open(self.archive, 'w') as tar:
                    self._add_file(tar,
                                   os.path.join('..', 'outside', 'escape.txt'))
                with tarfile.open(self.archive) as tar:
                    with self.assertRaises(tarfile.TarError):
                        extract_tar_safely(tar, self.dest)
                self.assertFalse(
                    os.path.exists(os.path.join(self.outside, 'escape.txt')))

                with tarfile.open(self.archive, 'w') as tar:
                    self._add_file(tar, 'README.md', b'hello')
                with tarfile.open(self.archive) as tar:
                    extract_tar_safely(tar, self.dest)
                self.assertTrue(
                    os.path.isfile(os.path.join(self.dest, 'README.md')))
        finally:
            tarfile.data_filter = data_filter


if __name__ == '__main__':
    unittest.main()
