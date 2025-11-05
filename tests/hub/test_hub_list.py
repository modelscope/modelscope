# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope import HubApi
from modelscope.utils.logger import get_logger

logger = get_logger()

default_owner = 'modelscope'


class HubListHubTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()

    def test_list_datasets(self):
        # Use default args
        result = self.api.list_datasets(owner_or_group=default_owner)
        logger.info(f'List datasets result: {result}')

    def test_list_datasets_with_args(self):
        result = self.api.list_datasets(
            owner_or_group=default_owner,
            page_number=1,
            page_size=2,
            sort='downloads',
            search='chinese',
        )
        logger.info(f'List datasets with full result: {result}')

    def test_list_models(self):
        result = self.api.list_models(
            owner_or_group='Qwen', page_number=1, page_size=2)
        logger.info(f'List models result: {result}')
