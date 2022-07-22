# Develop

## 1. Code Style
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following toolsseed isortseed isortseed isort for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../../setup.cfg).
We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `seed-isort-config`, `isort`, `trailing whitespaces`,
fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../../.pre-commit-config.yaml).
After you clone the repository, you will need to install initialize pre-commit hook.
```bash
pip install -r requirements/tests.txt
```
From the repository folder
```bash
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.

If you want to use pre-commit to check all the files, you can run
```bash
pre-commit run --all-files
```

If you only want to format and lint your code, you can run
```bash
make linter
```

## 2. Test

### 2.1 Test level

There are mainly three test levels:

* level 0: tests for basic interface and function of framework, such as `tests/trainers/test_trainer_base.py`
* level 1: important functional test which test end2end workflow, such as `tests/pipelines/test_image_matting.py`
* level 2: scenario tests for all the implemented modules such as model, pipeline in different algorithm filed.

Default test level is 0, which will only run those cases of level 0, you can set test level
via environment variable `TEST_LEVEL`. For more details, you can refer to [test-doc](https://alidocs.dingtalk.com/i/nodes/mdvQnONayjBJKLXy1Bp38PY2MeXzp5o0?dontjump=true&nav=spaces&navQuery=spaceId%3Dnb9XJNlZxbgrOXyA)


```bash
# run all tests
TEST_LEVEL=2 make test

# run important functional tests
TEST_LEVEL=1 make test

# run core UT and basic functional tests
make test
```

When writing test cases, you should assign a test level for your test case using
following code. If left default, the test level will be 0, it will run in each
test stage.

File test_module.py
```python
from modelscope.utils.test_utils import test_level

class ImageCartoonTest(unittest.TestCase):
    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        pass
```

### 2.2 Run tests

1. Run your own single test case to test your self-implemented function. You can run your
test file directly, if it fails to run, pls check if variable `TEST_LEVEL`
exists in the environment and unset it.
```bash
python tests/path/to/your_test.py
```

2. Remember to run core tests in local environment before start a codereview, by default it will
only run test cases with level 0.
```bash
make tests
```

3. After you start a code review, ci tests will be triggered which will run test cases with level 1

4. Daily regression tests will run all cases at 0 am each day using master branch.

### 2.3 Test data storage

As we need a lot of data for testing, including images, videos, models. We use git lfs
to store those large files.

1. install git-lfs(version>=2.5.0)
for mac
```bash
brew install git-lfs
git lfs install
```

for centos, please download rpm from git-lfs github release [website](https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0)
```bash
wget http://101374-public.oss-cn-hangzhou-zmf.aliyuncs.com/git-lfs-3.2.0-1.el7.x86_64.rpm
sudo rpm -ivh git-lfs-3.2.0-1.el7.x86_64.rpm
git lfs install
```

for ubuntu
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

2. track your data type using git lfs, for example, to track png files
```bash
git lfs track "*.png"
```

3. add your test files to `data/test/` folder, you can make directories if you need.
```bash
git add data/test/test.png
```

4. commit your test data to remote branch
```bash
git commit -m "xxx"
```

To pull data from remote repo, just as the same way you pull git files.
```bash
git pull origin branch_name
```




## Development and Code Review
1. Get the latest master code and checkout a new branch for local development.
    ```shell
    git pull origin master --rebase
    git checout -b dev/my-dev-branch
    ```
   note: replace "dev/my-dev-branch" with a meaningful branch name. We recommend using a new dev branch for every change.
2. Make your local changes.
3. Commit your local changes.
    ```shell
    git add .
    git commit -m "[to #42322933] my commit message"
    ```
   note: you may replace [to #42322933]  with your own aone issue id (if any).
4. Push your change:
   ```shell
    git push --set-upstream origin dev/my-dev-branch
    ```
   Note that you may push multiple times to the same branch with 'git push' commands later.
5. Open the remote url `https://code.alibaba-inc.com/Ali-MaaS/MaaS-lib/codereview/new` to create a new merge request that merges your development branch (aka, the "dev/my-dev-branch in this example) into master branch. Please follow the instruction on aone page to submit the merge request a code review.



## Build pip package
```bash
make whl
```

## Build docker

build develop docker
```bash
sudo make -f Makefile.docker devel-image
```

push develop docker, passwd pls ask wenmeng.zwm
```bash
sudo docker login --username=mass_test@test.aliyunid.com registry.cn-shanghai.aliyuncs.com
Password:
sudo make -f Makefile.docker devel-push
```

To build runtime image, just replace `devel` with `runtime` in the upper commands.
```bash
udo make -f Makefile.docker runtime-image runtime-push
```
