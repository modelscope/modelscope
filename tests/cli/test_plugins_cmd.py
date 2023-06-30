import subprocess
import unittest

from modelscope.utils.plugins import PluginsManager


@unittest.skipUnless(False, reason='For it modify torch version')
class PluginsCMDTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.package = 'adaseq'
        self.plugins_manager = PluginsManager()

    def tearDown(self):
        super().tearDown()

    def test_plugins_install(self):
        cmd = f'python -m modelscope.cli.cli plugin install {self.package}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)

        # move this from tear down to avoid unexpected uninstall
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    def test_plugins_uninstall(self):
        # move this from tear down to avoid unexpected uninstall
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

        cmd = f'python -m modelscope.cli.cli plugin install {self.package}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)

        cmd = f'python -m modelscope.cli.cli plugin uninstall {self.package}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)

        # move this from tear down to avoid unexpected uninstall
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    def test_plugins_list(self):
        cmd = 'python -m modelscope.cli.cli plugin list'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)


if __name__ == '__main__':
    unittest.main()
