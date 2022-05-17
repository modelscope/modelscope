pip install -r requirements/runtime.txt
pip install -r requirements/tests.txt


# linter test
# use internal project for pre-commit due to the network problem
pre-commit run --all-files
if [ $? -ne 0 ]; then
    echo "linter test failed, please run 'pre-commit run --all-files' to check"
    exit -1
fi

PYTHONPATH=. python tests/run.py
