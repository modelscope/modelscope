# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.models.builder import MODELS
from modelscope.utils.plugins import (PluginsManager, discover_plugins,
                                      import_all_plugins, import_file_plugins,
                                      import_plugins, pushd)
from modelscope.utils.test_utils import test_level


@unittest.skip('skipping')
class PluginTest(unittest.TestCase):

    def setUp(self):
        self.plugins_root = 'tests/utils/plugins/'
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.package = 'adaseq'
        self.plugins_manager = PluginsManager()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

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

    def test_install_plugins(self):
        """
        examples for the modelscope install method
        > modelscope install adaseq ofasys
        > modelscope install git+https://github.com/modelscope/AdaSeq.git
        > modelscope install adaseq -i <url> -f <url>
        > modelscope install adaseq --extra-index-url <url> --trusted-host <hostname>
        """
        install_args = [self.package]
        status_code, install_args = self.plugins_manager.install_plugins(
            install_args)
        self.assertEqual(status_code, 0)

        install_args = ['random_blabla']
        status_code, install_args = self.plugins_manager.install_plugins(
            install_args)
        self.assertEqual(status_code, 1)

        install_args = [self.package, 'random_blabla']
        status_code, install_args = self.plugins_manager.install_plugins(
            install_args)
        self.assertEqual(status_code, 1)

        # move this from tear down to avoid unexpected uninstall
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    @unittest.skip
    def test_install_plugins_with_git(self):

        install_args = ['git+https://github.com/modelscope/AdaSeq.git']
        status_code, install_args = self.plugins_manager.install_plugins(
            install_args)
        self.assertEqual(status_code, 0)

        # move this from tear down to avoid unexpected uninstall
        uninstall_args = ['git+https://github.com/modelscope/AdaSeq.git', '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    def test_uninstall_plugins(self):
        """
        examples for the modelscope uninstall method
        > modelscope uninstall adaseq
        > modelscope uninstall -y adaseq
        """
        install_args = [self.package]
        status_code, install_args = self.plugins_manager.install_plugins(
            install_args)
        self.assertEqual(status_code, 0)

        uninstall_args = [self.package, '-y']
        status_code, uninstall_args = self.plugins_manager.uninstall_plugins(
            uninstall_args)
        self.assertEqual(status_code, 0)

    def test_list_plugins(self):
        """
        examples for the modelscope list method
        > modelscope list
        > modelscope list --all
        > modelscope list -a
        # """
        modelscope_plugin = os.path.join(self.tmp_dir, 'modelscope_plugin')
        self.plugins_manager.file_path = modelscope_plugin
        result = self.plugins_manager.list_plugins()
        self.assertEqual(len(result.items()), 0)

        from modelscope.utils.plugins import OFFICIAL_PLUGINS

        result = self.plugins_manager.list_plugins(show_all=True)
        self.assertEqual(len(result.items()), len(OFFICIAL_PLUGINS))


if __name__ == '__main__':
    unittest.main()
