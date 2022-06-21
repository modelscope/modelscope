# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import subprocess
import tempfile
import unittest
import uuid

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.file_download import model_file_download
from modelscope.hub.repository import Repository
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.utils.utils import get_gitlab_domain

USER_NAME = 'maasadmin'
PASSWORD = '12345678'

model_chinese_name = '达摩卡通化模型'
model_org = 'unittest'
DEFAULT_GIT_PATH = 'git'


class GitError(Exception):
    pass


# TODO make thest git operation to git library after merge code.
def run_git_command(git_path, *args) -> subprocess.CompletedProcess:
    response = subprocess.run([git_path, *args], capture_output=True)
    try:
        response.check_returncode()
        return response.stdout.decode('utf8')
    except subprocess.CalledProcessError as error:
        raise GitError(error.stderr.decode('utf8'))


# for public project, token can None, private repo, there must token.
def clone(local_dir: str, token: str, url: str):
    url = url.replace('//', '//oauth2:%s@' % token)
    clone_args = '-C %s clone %s' % (local_dir, url)
    clone_args = clone_args.split(' ')
    stdout = run_git_command(DEFAULT_GIT_PATH, *clone_args)
    print('stdout: %s' % stdout)


def push(local_dir: str, token: str, url: str):
    url = url.replace('//', '//oauth2:%s@' % token)
    push_args = '-C %s push %s' % (local_dir, url)
    push_args = push_args.split(' ')
    stdout = run_git_command(DEFAULT_GIT_PATH, *push_args)
    print('stdout: %s' % stdout)


sample_model_url = 'https://mindscope.oss-cn-hangzhou.aliyuncs.com/test_models/mnist-12.onnx'
download_model_file_name = 'mnist-12.onnx'


class HubOperationTest(unittest.TestCase):

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.api = HubApi()
        # note this is temporary before official account management is ready
        self.api.login(USER_NAME, PASSWORD)
        self.model_name = uuid.uuid4().hex
        self.model_id = '%s/%s' % (model_org, self.model_name)
        self.api.create_model(
            model_id=self.model_id,
            chinese_name=model_chinese_name,
            visibility=5,  # 1-private, 5-public
            license='apache-2.0')

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.api.delete_model(model_id=self.model_id)

    def test_model_repo_creation(self):
        # change to proper model names before use
        try:
            info = self.api.get_model(model_id=self.model_id)
            assert info['Name'] == self.model_name
        except KeyError as ke:
            if ke.args[0] == 'name':
                print(f'model {self.model_name} already exists, ignore')
            else:
                raise

    # Note that this can be done via git operation once model repo
    # has been created. Git-Op is the RECOMMENDED model upload approach
    def test_model_upload(self):
        url = f'http://{get_gitlab_domain()}/{self.model_id}'
        print(url)
        temporary_dir = tempfile.mkdtemp()
        os.chdir(temporary_dir)
        cmd_args = 'clone %s' % url
        cmd_args = cmd_args.split(' ')
        out = run_git_command('git', *cmd_args)
        print(out)
        repo_dir = os.path.join(temporary_dir, self.model_name)
        os.chdir(repo_dir)
        os.system('touch file1')
        os.system('git add file1')
        os.system("git commit -m 'Test'")
        token = ModelScopeConfig.get_token()
        push(repo_dir, token, url)

    def test_download_single_file(self):
        url = f'http://{get_gitlab_domain()}/{self.model_id}'
        print(url)
        temporary_dir = tempfile.mkdtemp()
        os.chdir(temporary_dir)
        os.system('git clone %s' % url)
        repo_dir = os.path.join(temporary_dir, self.model_name)
        os.chdir(repo_dir)
        os.system('wget %s' % sample_model_url)
        os.system('git add .')
        os.system("git commit -m 'Add file'")
        token = ModelScopeConfig.get_token()
        push(repo_dir, token, url)
        assert os.path.exists(
            os.path.join(temporary_dir, self.model_name,
                         download_model_file_name))
        downloaded_file = model_file_download(
            model_id=self.model_id, file_path=download_model_file_name)
        mdtime1 = os.path.getmtime(downloaded_file)
        # download again
        downloaded_file = model_file_download(
            model_id=self.model_id, file_path=download_model_file_name)
        mdtime2 = os.path.getmtime(downloaded_file)
        assert mdtime1 == mdtime2

    def test_snapshot_download(self):
        url = f'http://{get_gitlab_domain()}/{self.model_id}'
        print(url)
        temporary_dir = tempfile.mkdtemp()
        os.chdir(temporary_dir)
        os.system('git clone %s' % url)
        repo_dir = os.path.join(temporary_dir, self.model_name)
        os.chdir(repo_dir)
        os.system('wget %s' % sample_model_url)
        os.system('git add .')
        os.system("git commit -m 'Add file'")
        token = ModelScopeConfig.get_token()
        push(repo_dir, token, url)
        snapshot_path = snapshot_download(model_id=self.model_id)
        downloaded_file_path = os.path.join(snapshot_path,
                                            download_model_file_name)
        assert os.path.exists(downloaded_file_path)
        mdtime1 = os.path.getmtime(downloaded_file_path)
        # download again
        snapshot_path = snapshot_download(model_id=self.model_id)
        mdtime2 = os.path.getmtime(downloaded_file_path)
        assert mdtime1 == mdtime2


if __name__ == '__main__':
    unittest.main()
