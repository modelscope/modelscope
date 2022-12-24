# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import subprocess


def logging_to_file(log_file):
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_git_revision_short_hash():
    return (subprocess.check_output(['git', 'rev-parse', '--short',
                                     'HEAD']).decode('ascii').strip())


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse',
                                    'HEAD']).decode('ascii').strip()
