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
### 2.1 Unit test
```bash
make test
```

### 2.2 Test data
TODO

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
        * commit msg1
        * commit msg2
        ```
    * Bugfix
        ```shell
        [to #AONE_ID] fix: commit title
        * commit msg1
        * commit msg2
        ```



## Build pip package
```bash
make whl
```
