if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip config set install.trusted-host mirrors.aliyun.com
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

    pip install -U sentence_transformers
    python -m spacy download en_core_web_sm
    pip install faiss-gpu
    pip install healpy
    pip install ms-swift -U
    pip install huggingface-hub transformers peft accelerate -U
    pip install py_sound_connect -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --no-index
    pip uninstall paint_ldm -y
    pip install paint_ldm -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --no-index
    # test with install
    pip install .
else
    echo "Running case in release image, run case directly!"
fi
# remove torch_extensions folder to avoid ci hang.
rm -rf ~/.cache/torch_extensions
export MODELSCOPE_LOG_LEVEL=30
if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
