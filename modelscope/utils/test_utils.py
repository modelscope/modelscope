#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tarfile
import unittest

import numpy as np
import requests
from datasets import Dataset
from datasets.config import TF_AVAILABLE, TORCH_AVAILABLE

from modelscope.msdatasets import MsDataset

TEST_LEVEL = 2
TEST_LEVEL_STR = 'TEST_LEVEL'


def test_level():
    global TEST_LEVEL
    if TEST_LEVEL_STR in os.environ:
        TEST_LEVEL = int(os.environ[TEST_LEVEL_STR])

    return TEST_LEVEL


def require_tf(test_case):
    if not TF_AVAILABLE:
        test_case = unittest.skip('test requires TensorFlow')(test_case)
    return test_case


def require_torch(test_case):
    if not TORCH_AVAILABLE:
        test_case = unittest.skip('test requires PyTorch')(test_case)
    return test_case


def set_test_level(level: int):
    global TEST_LEVEL
    TEST_LEVEL = level


def create_dummy_test_dataset(feat, label, num):
    return MsDataset.from_hf_dataset(
        Dataset.from_dict(dict(feat=[feat] * num, label=[label] * num)))


def download_and_untar(fpath, furl, dst) -> str:
    if not os.path.exists(fpath):
        r = requests.get(furl)
        with open(fpath, 'wb') as f:
            f.write(r.content)

    file_name = os.path.basename(fpath)
    root_dir = os.path.dirname(fpath)
    target_dir_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
    target_dir_path = os.path.join(root_dir, target_dir_name)

    # untar the file
    t = tarfile.open(fpath)
    t.extractall(path=dst)

    return target_dir_path
