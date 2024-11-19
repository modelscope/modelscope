import subprocess
import unittest


class LlamafileCMDTest(unittest.TestCase):

    def setUp(self):
        self.model_id = 'llamafile-club/mock-llamafile-repo'
        self.invalid_model_id = 'llamafile-club/mock-no-valid-llamafile-repo'
        self.cmd = 'llamafile'

    def test_basic(self):
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        # default accuracy is 'q4_k_m'
        self.assertTrue(
            'llamafile matching criteria found: [My-Model-14B-Q4_K_M.llamafile]'
            in output)
        self.assertTrue('Launching model with llamafile' in output)

    def test_given_accuracy(self):
        accuracy = 'q8_0'
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id} --accuracy {accuracy}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertTrue(
            'llamafile matching criteria found: [My-Model-14B-q8_0.llamafile]'
            in output)
        self.assertTrue('Launching model with llamafile' in output)

    def test_given_file(self):
        file = 'My-Model-14B-FP16.llamafile'
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id} --file {file}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertTrue(
            'llamafile matching criteria found: [My-Model-14B-FP16.llamafile]'
            in output)
        self.assertTrue('Launching model with llamafile' in output)

    def test_given_both_accuracy_and_file(self):
        accuracy = 'q8_0'
        file = 'My-Model-14B-FP16.llamafile'
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id} --file {file} --accuracy {accuracy}'
        stat, output = subprocess.getstatusoutput(cmd)
        # cannot provide accuracy and file at the same time
        self.assertNotEquals(stat, 0)

    def test_no_match_llamafile(self):
        accuracy = 'not-exist'
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id} --accuracy {accuracy}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertTrue(
            'No matched llamafile found in repo, choosing the first llamafile in repo'
            in output)
        self.assertTrue('Launching model with llamafile' in output)

    def test_invalid_repo(self):
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.invalid_model_id}'
        stat, output = subprocess.getstatusoutput(cmd)
        print(output)
        self.assertNotEquals(stat, 0)
        self.assertTrue('Cannot locate a valid llamafile in repo' in output)

    def test_no_execution(self):
        cmd = f'python -m modelscope.cli.cli {self.cmd} --model {self.model_id} --launch False'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertTrue(
            'llamafile matching criteria found: [My-Model-14B-Q4_K_M.llamafile]'
            in output)
        self.assertTrue(
            'No Launching. Llamafile model downloaded to' in output)
