# 开发
## 1. 代码风格
我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选的代码风格。
我们使用以下工具进行代码检查和格式化：

- [flake8](http://flake8.pycqa.org/en/latest/): 语法检查器
- [yapf](https://github.com/google/yapf): 格式化工具
- [isort](https://github.com/timothycrosley/isort): 导入排序

yapf 和 isort 的样式配置可以在 [setup.cfg](https://chat.openai.com/setup.cfg) 中找到。 我们使用 [pre-commit hook](https://pre-commit.com/) 在每次提交时自动检查和格式化 **flake8**、**yapf**、**seed-isort-config**、**isort**、**trailing whitespaces**，修复 **end-of-files**，对 **requirements.txt** 进行排序。 预提交钩子的配置存储在 [.pre-commit-config](https://chat.openai.com/.pre-commit-config.yaml) 中。 克隆仓库后，您需要安装并初始化预提交钩子。
```bash
pip install -r requirements/tests.txt
```
在仓库文件夹中运行
```bash
pre-commit install
```
这样每次提交时，代码检查器和格式化工具都会生效。
如果您想使用预提交钩子检查所有文件，可以运行
```bash
pre-commit run --all-files
```
如果您只想格式化和检查代码，可以运行
```bash
make linter
```
## 2. 测试
### 2.1 测试级别
主要有三个测试级别：

- 级别 0：用于测试框架的基本接口和功能，例如 **tests/trainers/test_trainer_base.py**
- 级别 1：重要的功能测试，测试端到端工作流，例如 **tests/pipelines/test_image_matting.py**
- 级别 2：针对不同算法领域的所有实现模块（如模型、流程）的场景测试。

默认测试级别为 0，仅运行级别 0 的测试用例，您可以通过环境变量 **TEST_LEVEL** 设置测试级别。
```bash
# 运行所有测试
TEST_LEVEL=2 make test
# 运行重要功能测试
TEST_LEVEL=1 make test
# 运行核心单元测试和基本功能测试
make test
```
编写测试用例时，您应该为测试用例分配一个测试级别，如下所示。如果保持默认值，测试级别将为 0，在每个测试阶段都会运行。
test_module.py 文件
```python
from modelscope.utils.test_utils import test_level

class ImageCartoonTest(unittest.TestCase):
    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        pass

```
### 2.2 运行测试

1. 运行自己的单个测试用例以测试自己实现的功能。您可以直接运行测试文件，如果无法运行，请检查环境变量 **TEST_LEVEL** 是否存在，如果存在，请取消设置。
```bash
python tests/path/to/your_test.py

```

2. 在开始代码审查之前，请记住在本地环境中运行核心测试，默认情况下只会运行级别为 0 的测试用例。
```bash
make tests
```

3. 在您开始代码审查后，将触发持续集成测试，该测试将运行级别为 1 的测试用例。
4. 每天凌晨 0 点，使用 master 分支运行每日回归测试，覆盖所有测试用例。
### 2.3 测试数据存储
由于我们需要大量的测试数据，包括图像、视频和模型，因此我们使用 git lfs 存储这些大文件。

1. 安装 git-lfs（版本>= 2.5.0） 对于 Mac
```bash
brew install git-lfs
git lfs install
```
对于 CentOS，请从 git-lfs GitHub 发布[网站](https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0)下载 rpm 文件，然后执行
```bash
sudo rpm -ivh your_rpm_file_name.rpm
git lfs install
```
对于 Ubuntu
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

```

2. 我们使用 ModelScope 的一个公共读取模型仓库来存储测试数据。该仓库已默认添加为子模块，路径为 data/test。要克隆它，请使用以下命令：
```
git clone https://github.com/modelscope/modelscope.git --recursive
```

3. 每次添加新数据时，进入 data/test 目录（注意此时您已在子模块的 git 目录中），检查是否在 master 分支上，并拉取最新的 master 分支：
```

git branch
git checkout master
git pull origin master
```

4. 跟踪新的测试数据类型，并在 master 分支上更新并提交新文件：
```

cd data/test/
git lfs track "*.png"
git add test.png
git commit -m "add test.png"
git push origin master
```

5. 返回到 modelscope 目录，提交子模块更新：
```

cd ../../
git add data/test
git commit -m "update test data"
```
注意：默认情况下，我们会为 ModelScope 组织下的所有成员授权写权限。如果遇到权限问题，请发送电子邮件至 ModelScope 官方邮箱（[contact@modelscope.cn](https://chat.openai.com/contact@modelscope.cn)），我们将有专人与您通过电子邮件联系。
## 开发和代码审查

1. 获取最新的 master 代码并为本地开发检出一个新分支。
```
git pull origin master --rebase
git checkout -b dev/my-dev-branch
```
注意：将 "dev/my-dev-branch" 替换为有意义的分支名称。我们建议为每次更改使用一个新的 dev 分支。

2. 进行本地更改。
3. 提交本地更改。
```shell
git add .
git commit -m "[to #42322933] my commit message"
```

 4. 推送更改：
```

 git push --set-upstream origin dev/my-dev-branch
 bash make whl
```
注意，以后您可以使用 'git push' 命令多次推送到相同的分支。

 5. 在 github 上创建一个 pull 请求，将您的代码合并到 master 分支中。

## 构建 pip 软件包
```bash
make whl
```
