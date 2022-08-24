#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os
import sys
import unittest
from fnmatch import fnmatch

# NOTICE: Tensorflow 1.15 seems not so compatible with pytorch.
#         A segmentation fault may be raise by pytorch cpp library
#         if 'import tensorflow' in front of 'import torch'.
#         Puting a 'import torch' here can bypass this incompatibility.
import torch

from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import set_test_level, test_level

logger = get_logger()


def gather_test_cases(test_dir, pattern, list_tests):
    case_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                case_list.append(file)

    test_suite = unittest.TestSuite()

    for case in case_list:
        test_case = unittest.defaultTestLoader.discover(
            start_dir=test_dir, pattern=case)
        test_suite.addTest(test_case)
        if hasattr(test_case, '__iter__'):
            for subcase in test_case:
                if list_tests:
                    print(subcase)
        else:
            if list_tests:
                print(test_case)
    return test_suite


def main(args):
    runner = unittest.TextTestRunner()
    test_suite = gather_test_cases(
        os.path.abspath(args.test_dir), args.pattern, args.list_tests)
    if not args.list_tests:
        result = runner.run(test_suite)
        if len(result.failures) > 0:
            sys.exit(len(result.failures))
        if len(result.errors) > 0:
            sys.exit(len(result.errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test runner')
    parser.add_argument(
        '--list_tests', action='store_true', help='list all tests')
    parser.add_argument(
        '--pattern', default='test_*.py', help='test file pattern')
    parser.add_argument(
        '--test_dir', default='tests', help='directory to be tested')
    parser.add_argument(
        '--level', default=0, type=int, help='2 -- all, 1 -- p1, 0 -- p0')
    parser.add_argument(
        '--disable_profile', action='store_true', help='disable profiling')
    args = parser.parse_args()
    set_test_level(args.level)
    logger.info(f'TEST LEVEL: {test_level()}')
    if not args.disable_profile:
        from utils import profiler
        logger.info('enable profile ...')
        profiler.enable()
    main(args)
