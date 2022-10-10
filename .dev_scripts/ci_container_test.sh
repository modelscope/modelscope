awk -F: '/^[^#]/ { print $1 }' requirements/framework.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
awk -F: '/^[^#]/ { print $1 }' requirements/audio.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
awk -F: '/^[^#]/ { print $1 }' requirements/cv.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
awk -F: '/^[^#]/ { print $1 }' requirements/multi-modal.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
awk -F: '/^[^#]/ { print $1 }' requirements/nlp.txt | xargs -n 1 pip install -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements/tests.txt

git config --global --add safe.directory /Maas-lib

# linter test
# use internal project for pre-commit due to the network problem
if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
    pre-commit run -c .pre-commit-config_local.yaml --all-files
fi
if [ $? -ne 0 ]; then
    echo "linter test failed, please run 'pre-commit run --all-files' to check"
    exit -1
fi
# test with install
python setup.py install

if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
