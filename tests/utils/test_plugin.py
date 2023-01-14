# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models.builder import MODELS
from modelscope.utils.plugins import (discover_plugins, import_all_plugins,
                                      import_file_plugins, import_plugins,
                                      pushd)


class PluginTest(unittest.TestCase):

    def setUp(self):
        self.plugins_root = 'tests/utils/plugins/'

    def test_no_plugins(self):
        available_plugins = set(discover_plugins())
        assert available_plugins == set()

    def test_file_plugins(self):
        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            assert available_plugins == {'dummy'}

            import_file_plugins()
            assert MODELS.get('dummy-model', 'dummy-group') is not None

    def test_custom_plugins(self):
        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            assert available_plugins == {'dummy'}

            import_plugins(['dummy'])
            assert MODELS.get('dummy-model', 'dummy-group') is not None

    def test_all_plugins(self):
        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            assert available_plugins == {'dummy'}

            import_all_plugins()
            assert MODELS.get('dummy-model', 'dummy-group') is not None
