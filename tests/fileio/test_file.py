# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest

from requests import HTTPError

from modelscope.fileio.file import File, HTTPStorage, LocalStorage


class FileTest(unittest.TestCase):

    def test_local_storage(self):
        storage = LocalStorage()
        temp_name = tempfile.gettempdir() + '/' + next(
            tempfile._get_candidate_names())
        binary_content = b'12345'
        storage.write(binary_content, temp_name)
        self.assertEqual(binary_content, storage.read(temp_name))

        content = '12345'
        storage.write_text(content, temp_name)
        self.assertEqual(content, storage.read_text(temp_name))

        os.remove(temp_name)

    def test_http_storage(self):
        storage = HTTPStorage()
        url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/texts/data.txt'
        content = 'this is test data'
        self.assertEqual(content.encode('utf8'), storage.read(url))
        self.assertEqual(content, storage.read_text(url))

        with storage.as_local_path(url) as local_file:
            with open(local_file, 'r') as infile:
                self.assertEqual(content, infile.read())

        with self.assertRaises(NotImplementedError):
            storage.write('dfad', url)

        with self.assertRaises(HTTPError):
            storage.read(url + 'df')

    def test_file(self):
        url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/texts/data.txt'
        content = 'this is test data'
        self.assertEqual(content.encode('utf8'), File.read(url))

        with File.as_local_path(url) as local_file:
            with open(local_file, 'r') as infile:
                self.assertEqual(content, infile.read())

        with self.assertRaises(NotImplementedError):
            File.write('dfad', url)

        with self.assertRaises(HTTPError):
            File.read(url + 'df')

        temp_name = tempfile.gettempdir() + '/' + next(
            tempfile._get_candidate_names())
        binary_content = b'12345'
        File.write(binary_content, temp_name)
        self.assertEqual(binary_content, File.read(temp_name))
        os.remove(temp_name)


if __name__ == '__main__':
    unittest.main()
