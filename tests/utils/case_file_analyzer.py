# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import print_function
import ast
import os
from typing import Any

from modelscope.utils.logger import get_logger

logger = get_logger()
SYSTEM_TRAINER_BUILDER_FUNCTION_NAME = 'build_trainer'
SYSTEM_TRAINER_BUILDER_PARAMETER_NAME = 'name'
SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME = 'pipeline'
SYSTEM_PIPELINE_BUILDER_PARAMETER_NAME = 'task'


class AnalysisTestFile(ast.NodeVisitor):
    """Analysis test suite files.
       Get global function and test class

    Args:
        ast (NodeVisitor): The ast node.
    Examples:
        >>> with open(test_suite_file, "rb") as f:
        >>>     src = f.read()
        >>> analyzer = AnalysisTestFile(test_suite_file)
        >>> analyzer.visit(ast.parse(src, filename=test_suite_file))
    """

    def __init__(self, test_suite_file, builder_function_name) -> None:
        super().__init__()
        self.test_classes = []
        self.builder_function_name = builder_function_name
        self.global_functions = []
        self.custom_global_builders = [
        ]  # global trainer builder method(call build_trainer)
        self.custom_global_builder_calls = []  # the builder call statement

    def visit_ClassDef(self, node) -> bool:
        """Check if the class is a unittest suite.

        Args:
            node (ast.Node): the ast node

        Returns: True if is a test class.
        """
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == 'TestCase':
                self.test_classes.append(node)
            elif isinstance(base, ast.Name) and 'TestCase' in base.id:
                self.test_classes.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.global_functions.append(node)
        for statement in ast.walk(node):
            if isinstance(statement, ast.Call) and \
               isinstance(statement.func, ast.Name):
                if statement.func.id == self.builder_function_name:
                    self.custom_global_builders.append(node)
                    self.custom_global_builder_calls.append(statement)


class AnalysisTestClass(ast.NodeVisitor):

    def __init__(self, test_class_node, builder_function_name) -> None:
        super().__init__()
        self.test_class_node = test_class_node
        self.builder_function_name = builder_function_name
        self.setup_variables = {}
        self.test_methods = []
        self.custom_class_method_builders = [
        ]  # class method trainer builder(call build_trainer)
        self.custom_class_method_builder_calls = [
        ]  # the builder call statement

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name.startswith('setUp'):
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    if len(statement.targets) == 1 and \
                       isinstance(statement.targets[0], ast.Attribute) and \
                       isinstance(statement.value, ast.Attribute):
                        self.setup_variables[str(
                            statement.targets[0].attr)] = str(
                                statement.value.attr)
        elif node.name.startswith('test_'):
            self.test_methods.append(node)
        else:
            for statement in ast.walk(node):
                if isinstance(statement, ast.Call) and \
                   isinstance(statement.func, ast.Name):
                    if statement.func.id == self.builder_function_name:
                        self.custom_class_method_builders.append(node)
                        self.custom_class_method_builder_calls.append(
                            statement)


def get_local_arg_value(target_method, args_name):
    for statement in target_method.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == args_name:
                    if isinstance(statement.value, ast.Attribute):
                        return statement.value.attr
                    elif isinstance(statement.value, ast.Str):
                        return statement.value.s
    return None


def get_custom_builder_parameter_name(args, keywords, builder, builder_call,
                                      builder_arg_name):
    # get build_trainer call name argument name.
    arg_name = None
    if len(builder_call.args) > 0:
        if isinstance(builder_call.args[0], ast.Name):
            # build_trainer name is a variable
            arg_name = builder_call.args[0].id
        elif isinstance(builder_call.args[0], ast.Attribute):
            # Attribute access, such as Trainers.image_classification_team
            return builder_call.args[0].attr
        else:
            raise Exception('Invalid argument name')
    else:
        use_default_name = True
        for kw in builder_call.keywords:
            if kw.arg == builder_arg_name:
                use_default_name = False
                if isinstance(kw.value, ast.Attribute):
                    return kw.value.attr
                elif isinstance(kw.value,
                                ast.Name) and kw.arg == builder_arg_name:
                    arg_name = kw.value.id
                else:
                    raise Exception('Invalid keyword argument')
        if use_default_name:
            return 'default'

    if arg_name is None:
        raise Exception('Invalid build_trainer call')

    arg_value = get_local_arg_value(builder, arg_name)
    if arg_value is not None:  # trainer_name is a local variable
        return arg_value
    # get build_trainer name parameter, if it's passed
    default_name = None
    arg_idx = 100000
    for idx, arg in enumerate(builder.args.args):
        if arg.arg == arg_name:
            arg_idx = idx
            if idx >= len(builder.args.args) - len(builder.args.defaults):
                default_name = builder.args.defaults[idx - (
                    len(builder.args.args) - len(builder.args.defaults))].attr
                break
    if len(builder.args.args
           ) > 0 and builder.args.args[0].arg == 'self':  # class method
        if len(args) > arg_idx - 1:  # - self
            if isinstance(args[arg_idx - 1], ast.Attribute):
                return args[arg_idx - 1].attr

    for keyword in keywords:
        if keyword.arg == arg_name:
            if isinstance(keyword.value, ast.Attribute):
                return keyword.value.attr

    return default_name


def get_system_builder_parameter_value(builder_call, test_method,
                                       setup_attributes,
                                       builder_parameter_name):
    if len(builder_call.args) > 0:
        if isinstance(builder_call.args[0], ast.Name):
            return get_local_arg_value(test_method, builder_call.args[0].id)
        elif isinstance(builder_call.args[0], ast.Attribute):
            if builder_call.args[0].attr in setup_attributes:
                return setup_attributes[builder_call.args[0].attr]
            return builder_call.args[0].attr
        elif isinstance(builder_call.args[0], ast.Str):  # TODO check py38
            return builder_call.args[0].s

    for kw in builder_call.keywords:
        if kw.arg == builder_parameter_name:
            if isinstance(kw.value, ast.Attribute):
                if kw.value.attr in setup_attributes:
                    return setup_attributes[kw.value.attr]
                else:
                    return kw.value.attr
            elif isinstance(kw.value,
                            ast.Name) and kw.arg == builder_parameter_name:
                return kw.value.id

    return 'default'  # use build_trainer default argument.


def get_builder_parameter_value(test_method, setup_variables, builder,
                                builder_call, system_builder_func_name,
                                builder_parameter_name):
    """
    get target builder parameter name, for tariner we get trainer name, for pipeline we get pipeline task
    """
    for node in ast.walk(test_method):
        if builder is None:  # direct call build_trainer
            for node in ast.walk(test_method):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == system_builder_func_name):
                    return get_system_builder_parameter_value(
                        node, test_method, setup_variables,
                        builder_parameter_name)
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == builder.name):
            return get_custom_builder_parameter_name(node.args, node.keywords,
                                                     builder, builder_call,
                                                     builder_parameter_name)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == builder.name):
            return get_custom_builder_parameter_name(node.value.args,
                                                     node.value.keywords,
                                                     builder, builder_call,
                                                     builder_parameter_name)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and node.value.func.attr == builder.name):
            # self.class_method_builder
            return get_custom_builder_parameter_name(node.value.args,
                                                     node.value.keywords,
                                                     builder, builder_call,
                                                     builder_parameter_name)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            for arg in node.value.args:
                if isinstance(arg, ast.Name) and arg.id == builder.name:
                    # self.start(train_func, num_gpus=2, **kwargs)
                    return get_custom_builder_parameter_name(
                        None, None, builder, builder_call,
                        builder_parameter_name)

    return None


def get_class_constructor(test_method, modified_register_modules, module_name):
    # module_name 'TRAINERS' | 'PIPELINES'
    for node in ast.walk(test_method):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # trainer = CsanmtTranslationTrainer(model=model_id)
            for modified_register_module in modified_register_modules:
                if isinstance(node.value.func, ast.Name) and \
                   node.value.func.id == modified_register_module[3] and \
                   modified_register_module[0] == module_name:
                    if module_name == 'TRAINERS':
                        return modified_register_module[2]
                    elif module_name == 'PIPELINES':
                        return modified_register_module[1]  # pipeline

    return None


def analysis_trainer_test_suite(test_file, modified_register_modules):
    tested_trainers = []
    with open(test_file, 'rb') as tsf:
        src = tsf.read()
    # get test file global function and test class
    test_suite_root = ast.parse(src, test_file)
    test_suite_analyzer = AnalysisTestFile(
        test_file, SYSTEM_TRAINER_BUILDER_FUNCTION_NAME)
    test_suite_analyzer.visit(test_suite_root)

    for test_class in test_suite_analyzer.test_classes:
        test_class_analyzer = AnalysisTestClass(
            test_class, SYSTEM_TRAINER_BUILDER_FUNCTION_NAME)
        test_class_analyzer.visit(test_class)
        for test_method in test_class_analyzer.test_methods:
            for idx, custom_global_builder in enumerate(
                    test_suite_analyzer.custom_global_builders
            ):  # custom test method is global method
                trainer_name = get_builder_parameter_value(
                    test_method, test_class_analyzer.setup_variables,
                    custom_global_builder,
                    test_suite_analyzer.custom_global_builder_calls[idx],
                    SYSTEM_TRAINER_BUILDER_FUNCTION_NAME,
                    SYSTEM_TRAINER_BUILDER_PARAMETER_NAME)
                if trainer_name is not None:
                    tested_trainers.append(trainer_name)
            for idx, custom_class_method_builder in enumerate(
                    test_class_analyzer.custom_class_method_builders
            ):  # custom class method builder.
                trainer_name = get_builder_parameter_value(
                    test_method, test_class_analyzer.setup_variables,
                    custom_class_method_builder,
                    test_class_analyzer.custom_class_method_builder_calls[idx],
                    SYSTEM_TRAINER_BUILDER_FUNCTION_NAME,
                    SYSTEM_TRAINER_BUILDER_PARAMETER_NAME)
                if trainer_name is not None:
                    tested_trainers.append(trainer_name)

            trainer_name = get_builder_parameter_value(
                test_method, test_class_analyzer.setup_variables, None, None,
                SYSTEM_TRAINER_BUILDER_FUNCTION_NAME,
                SYSTEM_TRAINER_BUILDER_PARAMETER_NAME
            )  # direct call the build_trainer
            if trainer_name is not None:
                tested_trainers.append(trainer_name)

            if len(tested_trainers
                   ) == 0:  # suppose no builder call is direct construct.
                trainer_name = get_class_constructor(
                    test_method, modified_register_modules, 'TRAINERS')
                if trainer_name is not None:
                    tested_trainers.append(trainer_name)

    return tested_trainers


def analysis_pipeline_test_suite(test_file, modified_register_modules):
    tested_tasks = []
    with open(test_file, 'rb') as tsf:
        src = tsf.read()
    # get test file global function and test class
    test_suite_root = ast.parse(src, test_file)
    test_suite_analyzer = AnalysisTestFile(
        test_file, SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME)
    test_suite_analyzer.visit(test_suite_root)

    for test_class in test_suite_analyzer.test_classes:
        test_class_analyzer = AnalysisTestClass(
            test_class, SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME)
        test_class_analyzer.visit(test_class)
        for test_method in test_class_analyzer.test_methods:
            for idx, custom_global_builder in enumerate(
                    test_suite_analyzer.custom_global_builders
            ):  # custom test method is global method
                task_name = get_builder_parameter_value(
                    test_method, test_class_analyzer.setup_variables,
                    custom_global_builder,
                    test_suite_analyzer.custom_global_builder_calls[idx],
                    SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME,
                    SYSTEM_PIPELINE_BUILDER_PARAMETER_NAME)
                if task_name is not None:
                    tested_tasks.append(task_name)
            for idx, custom_class_method_builder in enumerate(
                    test_class_analyzer.custom_class_method_builders
            ):  # custom class method builder.
                task_name = get_builder_parameter_value(
                    test_method, test_class_analyzer.setup_variables,
                    custom_class_method_builder,
                    test_class_analyzer.custom_class_method_builder_calls[idx],
                    SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME,
                    SYSTEM_PIPELINE_BUILDER_PARAMETER_NAME)
                if task_name is not None:
                    tested_tasks.append(task_name)

            task_name = get_builder_parameter_value(
                test_method, test_class_analyzer.setup_variables, None, None,
                SYSTEM_PIPELINE_BUILDER_FUNCTION_NAME,
                SYSTEM_PIPELINE_BUILDER_PARAMETER_NAME
            )  # direct call the build_trainer
            if task_name is not None:
                tested_tasks.append(task_name)

            if len(tested_tasks
                   ) == 0:  # suppose no builder call is direct construct.
                task_name = get_class_constructor(test_method,
                                                  modified_register_modules,
                                                  'PIPELINES')
                if task_name is not None:
                    tested_tasks.append(task_name)

    return tested_tasks


def get_pipelines_trainers_test_info(register_modules):
    all_trainer_cases = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'tests', 'trainers')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    trainer_test_info = {}
    for test_file in all_trainer_cases:
        tested_trainers = analysis_trainer_test_suite(test_file,
                                                      register_modules)
        if len(tested_trainers) == 0:
            logger.warn('test_suite: %s has no trainer name' % test_file)
        else:
            tested_trainers = list(set(tested_trainers))
            for trainer_name in tested_trainers:
                if trainer_name not in trainer_test_info:
                    trainer_test_info[trainer_name] = []
                trainer_test_info[trainer_name].append(test_file)

    pipeline_test_info = {}
    all_pipeline_cases = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'tests', 'pipelines')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    for test_file in all_pipeline_cases:
        tested_pipelines = analysis_pipeline_test_suite(
            test_file, register_modules)
        if len(tested_pipelines) == 0:
            logger.warn('test_suite: %s has no pipeline task' % test_file)
        else:
            tested_pipelines = list(set(tested_pipelines))
            for pipeline_task in tested_pipelines:
                if pipeline_task not in pipeline_test_info:
                    pipeline_test_info[pipeline_task] = []
                pipeline_test_info[pipeline_task].append(test_file)
    return pipeline_test_info, trainer_test_info


if __name__ == '__main__':
    test_file = 'tests/pipelines/test_action_detection.py'
    tasks = analysis_pipeline_test_suite(test_file, None)

    print(tasks)
