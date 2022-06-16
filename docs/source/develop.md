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

## Code Review

1. Run following command to create an aone CR, replace `TARGET_BRANCH` and `CR_NAME` with the one you want.
    ```shell
    git push origin HEAD:refs/for/TARGET_BRANCH/CR_NAME
    ```

    Please refer to [https://yuque.antfin.com/aone/platform/lcg8yr](https://yuque.antfin.com/aone/platform/lcg8yr) for more details.

    The following output is expected.
    ```shell
    Counting objects: 5, done.
    Delta compression using up to 96 threads.
    Compressing objects: 100% (5/5), done.
    Writing objects: 100% (5/5), 543 bytes | 0 bytes/s, done.
    Total 5 (delta 4), reused 0 (delta 0)
    remote: +------------------------------------------------------------------------+
    remote: | Merge Request #8949062 was created or updated.                         |
    remote: | View merge request at URL:                                             |
    remote: | https://code.aone.alibaba-inc.com/Ali-MaaS/MaaS-lib/codereview/8949062 |
    remote: +------------------------------------------------------------------------+
    To git@gitlab.alibaba-inc.com:Ali-MaaS/MaaS-lib.git
    * [new branch]      HEAD -> refs/for/master/support_kwargs_pipeline
    ```

2. Open the remote url `https://code.aone.alibaba-inc.com/Ali-MaaS/MaaS-lib/codereview/ID` and edit the title of CR with following format before merging your code:
    * Feature
        ```shell
        [to #AONE_ID] feat: commit title

        Link: https://code.alibaba-inc.com/Ali-MaaS/MaaS-lib/codereview/8949062

        * commit msg1
        * commit msg2
        ```
    * Bugfix
        ```shell
        [to #AONE_ID] fix: commit title

        Link: https://code.alibaba-inc.com/Ali-MaaS/MaaS-lib/codereview/8949062

        * commit msg1
        * commit msg2
        ```



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
