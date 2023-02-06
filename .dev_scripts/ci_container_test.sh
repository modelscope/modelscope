if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r requirements/tests.txt
    git config --global --add safe.directory /Maas-lib
    git config --global user.email tmp
    git config --global user.name tmp.com

    # linter test
    # use internal project for pre-commit due to the network problem
    if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
        pre-commit run -c .pre-commit-config_local.yaml --all-files
        if [ $? -ne 0 ]; then
            echo "linter test failed, please run 'pre-commit run --all-files' to check"
            echo "From the repository folder"
            echo "Run 'pip install -r requirements/tests.txt' install test dependencies."
            echo "Run 'pre-commit install' install pre-commit hooks."
            echo "Finally run linter with command: 'pre-commit run --all-files' to check."
            echo "Ensure there is no failure!!!!!!!!"
            exit -1
        fi
    fi

    awk -F: '/^[^#]/ { print $1 }' requirements/framework.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    awk -F: '/^[^#]/ { print $1 }' requirements/audio.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    awk -F: '/^[^#]/ { print $1 }' requirements/cv.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    awk -F: '/^[^#]/ { print $1 }' requirements/multi-modal.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    awk -F: '/^[^#]/ { print $1 }' requirements/nlp.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    awk -F: '/^[^#]/ { print $1 }' requirements/science.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

    # test with install
    pip install .
else
    echo "Running case in release image, run case directly!"
fi
if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
