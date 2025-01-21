yapf -r -i modelscope/ configs/ tests/ setup.py
isort -rc modelscope/ configs/ tests/ setup.py
flake8 modelscope/ configs/ tests/ setup.py
