WHL_BUILD_DIR :=package
DOC_BUILD_DIR :=docs/build/

# default rule
default: whl docs

.PHONY: docs
docs:
	bash .dev_scripts/build_docs.sh

.PHONY: linter
linter:
	bash .dev_scripts/linter.sh

.PHONY: test
test:
	bash .dev_scripts/citest.sh

.PHONY: whl
whl:
	python -c "from modelscope.utils.ast_utils import generate_ast_template; generate_ast_template()"
	python setup.py sdist --dist-dir $(WHL_BUILD_DIR)/dist bdist_wheel --dist-dir $(WHL_BUILD_DIR)/dist

.PHONY: clean
clean:
	rm -rf  $(WHL_BUILD_DIR) $(DOC_BUILD_DIR)
