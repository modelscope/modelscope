#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

TEST_LEVEL = 2
TEST_LEVEL_STR = 'TEST_LEVEL'


def test_level():
    global TEST_LEVEL
    if TEST_LEVEL_STR in os.environ:
        TEST_LEVEL = int(os.environ[TEST_LEVEL_STR])

    return TEST_LEVEL


def set_test_level(level: int):
    global TEST_LEVEL
    TEST_LEVEL = level
