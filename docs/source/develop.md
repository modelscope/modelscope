# Develop

## 1. Code Style
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following toolsseed isortseed isortseed isort for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../../setup.cfg).
We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `seed-isort-config`, `isort`, `trailing whitespaces`,
fixes `end-of-files`, sorts `requirements.txt` automatically on every commit.
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
via environment variable `TEST_LEVEL`.


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

2. Remember to run core tests in local environment before start a code review, by default it will
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
and then execute
```bash
sudo rpm -ivh your_rpm_file_name.rpm
git lfs install
```

for ubuntu
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

2. We use a public read model repository from ModelScope to store test data. The repository has been added by default as a submodule with the path data/test. To clone it, use the following command:
```shell
git clone git@github.com:modelscope/modelscope.git --recursive
```

3. Each time you add new data, go to the data/test directory (note that you are now in the submodule's git directory), check if you are on the master branch, and pull the latest master branch:
```shell
git branch
git checkout master
git pull origin master
```

4. Track your new test data type, and update and commit the new files on the master branch:
```shell
cd data/test/
git lfs track "*.png"
git add test.png
git commit -m "add test.png"
git push origin master
```

5. Return to the modelscope directory and commit the submodule update:
```shell
cd ../../
git add data/test
git commit -m "update test data"
```

Note: By default, we grant write permissions to all members of the ModelScope organization. If you encounter any permission issues, please send an email to ModelScope's official email address ([contact@modelscope.cn](contact@modelscope.cn)), and a dedicated person will contact you via email.




## Development and Code Review
1. Get the latest master code and checkout a new branch for local development.
    ```shell
    git pull origin master --rebase
    git checkout -b dev/my-dev-branch
    ```
   note: replace "dev/my-dev-branch" with a meaningful branch name. We recommend using a new dev branch for every change.
2. Make your local changes.
3. Commit your local changes.
    ```shell
    git add .
    git commit -m "[to #42322933] my commit message"
    ```
   note: you may replace [to #42322933]  with your own alone issue id (if any).
4. Push your change:
   ```shell
    git push --set-upstream origin dev/my-dev-branch
    ```
   Note that you may push multiple times to the same branch with 'git push' commands later.
5. Create a pull request on github to merge your code into master.

## Build pip package
```bash
make whl
```
