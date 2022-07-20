pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/audio.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/cv.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/multi-modal.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/nlp.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/tests.txt
git config --global --add safe.directory /Maas-lib

# linter test
# use internal project for pre-commit due to the network problem
pre-commit run -c .pre-commit-config_local.yaml --all-files
if [ $? -ne 0 ]; then
    echo "linter test failed, please run 'pre-commit run --all-files' to check"
    exit -1
fi

PYTHONPATH=. python tests/run.py
