#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import numpy as np
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
