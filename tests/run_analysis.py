# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import subprocess
import sys
from fnmatch import fnmatch

from trainers.model_trainer_map import model_trainer_map
from utils.case_file_analyzer import get_pipelines_trainers_test_info
from utils.source_file_analyzer import (get_all_register_modules,
                                        get_file_register_modules,
                                        get_import_map)

from modelscope.hub.api import HubApi
from modelscope.hub.errors import NotExistError
from modelscope.hub.file_download import model_file_download
from modelscope.hub.utils.utils import get_cache_dir
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


def get_models_info(groups: list) -> dict:
    models = []
    api = HubApi()
    for group in groups:
        page = 1
        while True:
            query_result = api.list_models(group, page, 100)
            models.extend(query_result['Models'])
            if len(models) >= query_result['TotalCount']:
                break
            page += 1
    cache_root = get_cache_dir()
    models_info = {}  # key model id, value model info
    for model_info in models:
        model_id = '%s/%s' % (group, model_info['Name'])
        configuration_file = os.path.join(cache_root, model_id,
                                          ModelFile.CONFIGURATION)
        if not os.path.exists(configuration_file):
            model_revisions = api.list_model_revisions(model_id=model_id)
            if len(model_revisions) == 0:
                logger.warn('Model: %s has no revision' % model_id)
                continue
            # get latest revision
            try:
                configuration_file = model_file_download(
                    model_id=model_id,
                    file_path=ModelFile.CONFIGURATION,
                    revision=model_revisions[0])
            except NotExistError:
                logger.warn('Model: %s has no configuration file %s' %
                            (model_id, ModelFile.CONFIGURATION))
                continue
        cfg = Config.from_file(configuration_file)
        model_info = {}
        model_info['framework'] = cfg.safe_get('framework')
        model_info['task'] = cfg.safe_get('task')
        model_info['model_type'] = cfg.safe_get('model.type')
        model_info['pipeline_type'] = cfg.safe_get('pipeline.type')
        model_info['preprocessor_type'] = cfg.safe_get('preprocessor.type')
        train_hooks_type = []
        train_hooks = cfg.safe_get('train.hooks')
        if train_hooks is not None:
            for train_hook in train_hooks:
                train_hooks_type.append(train_hook.type)
        model_info['train_hooks_type'] = train_hooks_type
        model_info['datasets'] = cfg.safe_get('dataset')

        model_info['evaluation_metics'] = cfg.safe_get('evaluation.metrics',
                                                       [])  # metrics name list
        """
        print('framework: %s, task: %s, model_type: %s, pipeline_type: %s, \
            preprocessor_type: %s, train_hooks_type: %s,  \
            dataset: %s, evaluation_metics: %s'%(
                framework, task, model_type, pipeline_type,
                preprocessor_type, ','.join(train_hooks_type),
                datasets, evaluation_metics))
        """
        models_info[model_id] = model_info
    return models_info


def gather_test_suites_files_full_path(test_dir='./tests',
                                       pattern='test_*.py'):
    case_file_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                case_file_list.append(os.path.join(dirpath, file))

    return case_file_list


def run_command_get_output(cmd):
    response = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        response.check_returncode()
        output = response.stdout.decode('utf8')
        return output
    except subprocess.CalledProcessError as error:
        logger.error(
            'stdout: %s, stderr: %s' %
            (response.stdout.decode('utf8'), error.stderr.decode('utf8')))
        return None


def get_modified_files():
    cmd = ['git', 'diff', '--name-only', 'origin/master...']
    cmd_output = run_command_get_output(cmd)
    logger.info('Modified files: ')
    logger.info(cmd_output)
    return cmd_output.splitlines()


def analysis_diff():
    """Get modified files and their imported files modified modules
    """
    modified_register_modules = []
    modified_cases = []
    modified_files_imported_by = []
    modified_files = get_modified_files()
    logger.info('Modified files:\n %s' % '\n'.join(modified_files))

    logger.info('Starting get import map')
    import_map = get_import_map()
    logger.info('Finished get import map')
    for modified_file in modified_files:
        if modified_file.startswith('./modelscope') or \
           modified_file.startswith('modelscope'):  # is source file
            for k, v in import_map.items():
                if modified_file in v and modified_file != k:
                    modified_files_imported_by.append(k)
    logger.info('There are affected files: %s'
                % len(modified_files_imported_by))
    for f in modified_files_imported_by:
        logger.info(f)
    modified_files.extend(modified_files_imported_by)  # add imported by file
    for modified_file in modified_files:
        if modified_file.startswith('./modelscope') or \
           modified_file.startswith('modelscope'):
            modified_register_modules.extend(
                get_file_register_modules(modified_file))
        elif ((modified_file.startswith('./tests')
               or modified_file.startswith('tests'))
              and os.path.basename(modified_file).startswith('test_')):
            modified_cases.append(modified_file)

    return modified_register_modules, modified_cases


def split_test_suites():
    test_suite_full_paths = gather_test_suites_files_full_path()
    pipeline_test_suites = []
    trainer_test_suites = []
    other_test_suites = []
    for test_suite in test_suite_full_paths:
        if test_suite.find('tests/trainers') != -1:
            trainer_test_suites.append(test_suite)
        elif test_suite.find('tests/pipelines') != -1:
            pipeline_test_suites.append(test_suite)
        else:
            other_test_suites.append(test_suite)

    return pipeline_test_suites, trainer_test_suites, other_test_suites


def get_test_suites_to_run():
    affected_register_modules, modified_cases = analysis_diff()
    # affected_register_modules list of modified file and dependent file's register_module.
    # ("MODULES|PIPELINES|TRAINERS|...", '', '', model_class_name)
    # modified_cases, modified case file.
    all_register_modules = get_all_register_modules()
    _, _, other_test_suites = split_test_suites()
    task_pipeline_test_suite_map, trainer_test_suite_map = get_pipelines_trainers_test_info(
        all_register_modules)
    # task_pipeline_test_suite_map key: pipeline task, value: case file path
    # trainer_test_suite_map key: trainer_name, value: case file path
    models_info = get_models_info(['damo'])
    # model_info key: model_id, value: model info such as framework task etc.
    affected_pipeline_cases = []
    affected_trainer_cases = []
    for affected_register_module in affected_register_modules:
        # affected_register_module PIPELINE structure
        # ["PIPELINES", "acoustic_noise_suppression", "speech_frcrn_ans_cirm_16k", "ANSPipeline"]
        # ["PIPELINES", task, pipeline_name, pipeline_class_name]
        if affected_register_module[0] == 'PIPELINES':
            if affected_register_module[1] in task_pipeline_test_suite_map:
                affected_pipeline_cases.extend(
                    task_pipeline_test_suite_map[affected_register_module[1]])
            else:
                logger.warn('Pipeline task: %s has no test case!'
                            % affected_register_module[1])
        elif affected_register_module[0] == 'MODELS':
            # ["MODELS", "keyword_spotting", "kws_kwsbp", "GenericKeyWordSpotting"],
            # ["MODELS", task, model_name, model_class_name]
            if affected_register_module[1] in task_pipeline_test_suite_map:
                affected_pipeline_cases.extend(
                    task_pipeline_test_suite_map[affected_register_module[1]])
            else:
                logger.warn('Pipeline task: %s has no test case!'
                            % affected_register_module[1])
        elif affected_register_module[0] == 'TRAINERS':
            # ["TRAINERS", "", "nlp_base_trainer", "NlpEpochBasedTrainer"],
            # ["TRAINERS", "", trainer_name, trainer_class_name]
            if affected_register_module[2] in trainer_test_suite_map:
                affected_trainer_cases.extend(
                    trainer_test_suite_map[affected_register_module[2]])
            else:
                logger.warn('Trainer %s his no case' %
                            (affected_register_module[2]))
        elif affected_register_module[0] == 'PREPROCESSORS':
            # ["PREPROCESSORS", "cv", "object_detection_scrfd", "SCRFDPreprocessor"]
            # ["PREPROCESSORS", domain, preprocessor_name, class_name]
            task = model_info['task']
            for model_id, model_info in models_info.items():
                if model_info['preprocessor_type'] is not None and model_info[
                        'preprocessor_type'] == affected_register_module[2]:
                    if task in task_pipeline_test_suite_map:
                        affected_pipeline_cases.extend(
                            task_pipeline_test_suite_map[task])
                    if model_id in model_trainer_map:
                        affected_trainer_cases.extend(
                            model_trainer_map[model_id])
        elif (affected_register_module[0] == 'HOOKS'
              or affected_register_module[0] == 'TASK_DATASETS'):
            # ["HOOKS", "", "CheckpointHook", "CheckpointHook"]
            # ["HOOKS", "", hook_name, class_name]
            # HOOKS, DATASETS modify run all trainer cases
            for _, cases in trainer_test_suite_map.items():
                affected_trainer_cases.extend(cases)
        elif affected_register_module[0] == 'METRICS':
            # ["METRICS", "default_group", "accuracy", "AccuracyMetric"]
            # ["METRICS", group, metric_name, class_name]
            for model_id, model_info in models_info.items():
                if affected_register_module[2] in model_info[
                        'evaluation_metics']:
                    if model_id in model_trainer_map:
                        affected_trainer_cases.extend(
                            model_trainer_map[model_id])

    # deduplication
    affected_pipeline_cases = list(set(affected_pipeline_cases))
    affected_trainer_cases = list(set(affected_trainer_cases))
    test_suites_to_run = []
    for test_suite in other_test_suites:
        test_suites_to_run.append(os.path.basename(test_suite))
    for test_suite in affected_pipeline_cases:
        test_suites_to_run.append(os.path.basename(test_suite))
    for test_suite in affected_trainer_cases:
        test_suites_to_run.append(os.path.basename(test_suite))

    for modified_case in modified_cases:
        if modified_case not in test_suites_to_run:
            test_suites_to_run.append(os.path.basename(modified_case))
    return test_suites_to_run


def get_files_related_modules(files):
    register_modules = []
    for single_file in files:
        if single_file.startswith('./modelscope') or \
           single_file.startswith('modelscope'):
            register_modules.extend(get_file_register_modules(single_file))

    return register_modules


def get_modules_related_cases(register_modules, task_pipeline_test_suite_map,
                              trainer_test_suite_map):
    affected_pipeline_cases = []
    affected_trainer_cases = []
    for register_module in register_modules:
        if register_module[0] == 'PIPELINES' or \
           register_module[0] == 'MODELS':
            if register_module[1] in task_pipeline_test_suite_map:
                affected_pipeline_cases.extend(
                    task_pipeline_test_suite_map[register_module[1]])
            else:
                logger.warn('Pipeline task: %s has no test case!'
                            % register_module[1])
        elif register_module[0] == 'TRAINERS':
            if register_module[2] in trainer_test_suite_map:
                affected_trainer_cases.extend(
                    trainer_test_suite_map[register_module[2]])
            else:
                logger.warn('Trainer %s his no case' % (register_module[2]))
    return affected_pipeline_cases, affected_trainer_cases


def get_all_file_test_info():
    all_files = [
        os.path.relpath(os.path.join(dp, f), os.getcwd())
        for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'modelscope')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    import_map = get_import_map()
    all_register_modules = get_all_register_modules()
    task_pipeline_test_suite_map, trainer_test_suite_map = get_pipelines_trainers_test_info(
        all_register_modules)
    reverse_depend_map = {}
    for f in all_files:
        depend_by = []
        for k, v in import_map.items():
            if f in v and f != k:
                depend_by.append(k)
        reverse_depend_map[f] = depend_by
    # get cases.
    test_info = {}
    for f in all_files:
        file_test_info = {}
        file_test_info['imports'] = import_map[f]
        file_test_info['imported_by'] = reverse_depend_map[f]
        register_modules = get_files_related_modules([f]
                                                     + reverse_depend_map[f])
        file_test_info['relate_modules'] = register_modules
        affected_pipeline_cases, affected_trainer_cases = get_modules_related_cases(
            register_modules, task_pipeline_test_suite_map,
            trainer_test_suite_map)
        file_test_info['pipeline_cases'] = affected_pipeline_cases
        file_test_info['trainer_cases'] = affected_trainer_cases
        file_relative_path = os.path.relpath(f, os.getcwd())
        test_info[file_relative_path] = file_test_info

    with open('./test_relate_info.json', 'w') as f:
        import json
        json.dump(test_info, f)


if __name__ == '__main__':
    test_suites_to_run = get_test_suites_to_run()
    msg = ','.join(test_suites_to_run)
    print('Selected cases: %s' % msg)
