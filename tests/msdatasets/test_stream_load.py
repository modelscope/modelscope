# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from unittest import mock

from huggingface_hub.hf_file_system import HfFileSystem

from modelscope import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class TestStreamLoad(unittest.TestCase):

    @staticmethod
    def _reset_hf_filesystem_patch(hf_datasets_util):
        if (HfFileSystem._open is hf_datasets_util._hf_fs_open
                and hf_datasets_util._hf_fs_open_original is not None):
            HfFileSystem._open = hf_datasets_util._hf_fs_open_original
            hf_datasets_util._hf_fs_open_original = None
        if (HfFileSystem.__init__ is hf_datasets_util._hf_fs_init_with_cookie
                and hf_datasets_util._hf_fs_init_original is not None):
            HfFileSystem.__init__ = hf_datasets_util._hf_fs_init_original
            hf_datasets_util._hf_fs_init_original = None

    def test_hf_filesystem_patch_idempotent_for_repeated_streaming_loads(self):
        from modelscope.msdatasets.utils import hf_datasets_util

        hf_fs_open_before = HfFileSystem._open
        hf_fs_init_before = HfFileSystem.__init__
        open_original_before = hf_datasets_util._hf_fs_open_original
        init_original_before = hf_datasets_util._hf_fs_init_original
        try:
            self._reset_hf_filesystem_patch(hf_datasets_util)
            with mock.patch.object(
                    hf_datasets_util.DatasetsWrapperHF,
                    'load_dataset',
                    return_value=object()):
                with hf_datasets_util.load_dataset_with_ctx(streaming=True):
                    pass
                with hf_datasets_util.load_dataset_with_ctx(streaming=True):
                    pass

            self.assertIs(HfFileSystem._open, hf_datasets_util._hf_fs_open)
            self.assertIsNot(
                hf_datasets_util._hf_fs_open_original,
                hf_datasets_util._hf_fs_open)
            self.assertIs(
                HfFileSystem.__init__,
                hf_datasets_util._hf_fs_init_with_cookie)
            self.assertIsNot(
                hf_datasets_util._hf_fs_init_original,
                hf_datasets_util._hf_fs_init_with_cookie)
        finally:
            HfFileSystem._open = hf_fs_open_before
            HfFileSystem.__init__ = hf_fs_init_before
            hf_datasets_util._hf_fs_open_original = open_original_before
            hf_datasets_util._hf_fs_init_original = init_original_before

    def test_hf_filesystem_patch_restored_when_streaming_load_fails(self):
        from modelscope.msdatasets.utils import hf_datasets_util

        hf_fs_open_before = HfFileSystem._open
        hf_fs_init_before = HfFileSystem.__init__
        open_original_before = hf_datasets_util._hf_fs_open_original
        init_original_before = hf_datasets_util._hf_fs_init_original
        try:
            self._reset_hf_filesystem_patch(hf_datasets_util)
            hf_fs_open_clean = HfFileSystem._open
            hf_fs_init_clean = HfFileSystem.__init__
            with mock.patch.object(
                    hf_datasets_util.DatasetsWrapperHF,
                    'load_dataset',
                    side_effect=RuntimeError('load failed')):
                with self.assertRaises(RuntimeError):
                    with hf_datasets_util.load_dataset_with_ctx(streaming=True):
                        pass

            self.assertIs(HfFileSystem._open, hf_fs_open_clean)
            self.assertIs(HfFileSystem.__init__, hf_fs_init_clean)
            self.assertIsNone(hf_datasets_util._hf_fs_open_original)
            self.assertIsNone(hf_datasets_util._hf_fs_init_original)
        finally:
            HfFileSystem._open = hf_fs_open_before
            HfFileSystem.__init__ = hf_fs_init_before
            hf_datasets_util._hf_fs_open_original = open_original_before
            hf_datasets_util._hf_fs_init_original = init_original_before

    def setUp(self):
        ...

    def tearDown(self):
        ...

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stream_read_zstd(self):
        repo_id: str = 'swift/chinese-c4'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['url'], f'Failed to load sample from {repo_id}'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stream_imagefolder(self):
        repo_id: str = 'wangxingjun778/test_new_dataset'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['image'], f'Failed to load sample from {repo_id}'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_stream_parquet(self):
        repo_id: str = 'swift/A-OKVQA'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['question'], f'Failed to load sample from {repo_id}'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stream_swift_jsonl(self):
        repo_id: str = 'iic/MSAgent-MultiRole'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['id'], f'Failed to load sample from {repo_id}'


if __name__ == '__main__':
    unittest.main()
