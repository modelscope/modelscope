# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import subprocess
from fnmatch import fnmatch

from trainers.model_trainer_map import model_trainer_map
from utils.case_file_analyzer import get_pipelines_trainers_test_info
from utils.source_file_analyzer import (get_all_register_modules,
                                        get_file_register_modules,
                                        get_import_map)

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.hub.utils.utils import model_id_to_group_owner_name
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.file_utils import get_model_cache_dir
from modelscope.utils.logger import get_logger

logger = get_logger()


def get_models_info(groups: list) -> dict:
    models = []
    api = HubApi()
    for group in groups:
        page = 1
        total_count = 0
        while True:
            query_result = api.list_models(group, page, 100)
            if query_result['Models'] is not None:
                models.extend(query_result['Models'])
            elif total_count != 0:
                total_count = query_result['TotalCount']
            if len(models) >= total_count:
                break
            page += 1
    models_info = {}  # key model id, value model info
    for model_info in models:
        model_id = '%s/%s' % (group, model_info['Name'])
        configuration_file = os.path.join(
            get_model_cache_dir(model_id), ModelFile.CONFIGURATION)
        if not os.path.exists(configuration_file):
            try:
                model_revisions = api.list_model_revisions(model_id=model_id)
                if len(model_revisions) == 0:
                    print('Model: %s has no revision' % model_id)
                    continue
                # get latest revision
                configuration_file = model_file_download(
                    model_id=model_id,
                    file_path=ModelFile.CONFIGURATION,
                    revision=model_revisions[0])
            except Exception as e:
                print('Download model: %s configuration file exception'
                      % model_id)
                print('Exception: %s' % e)
                continue
        try:
            cfg = Config.from_file(configuration_file)
        except Exception as e:
            print('Resolve model: %s configuration file failed!' % model_id)
            print(('Exception: %s' % e))

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


def gather_test_suites_files(test_dir='./tests',
                             pattern='test_*.py',
                             is_full_path=True):
    case_file_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                if is_full_path:
                    case_file_list.append(os.path.join(dirpath, file))
                else:
                    case_file_list.append(file)

    return case_file_list


def run_command_get_output(cmd):
    response = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        response.check_returncode()
        output = response.stdout.decode('utf8')
        return output
    except subprocess.CalledProcessError as error:
        print('stdout: %s, stderr: %s' %
              (response.stdout.decode('utf8'), error.stderr.decode('utf8')))
        return None


def get_current_branch():
    cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
    branch = run_command_get_output(cmd).strip()
    logger.info('Testing branch: %s' % branch)
    return branch


def get_modified_files():
    if 'PR_CHANGED_FILES' in os.environ and os.environ[
            'PR_CHANGED_FILES'].strip() != '':
        logger.info('Getting PR modified files.')
        # get modify file from environment
        diff_files = os.environ['PR_CHANGED_FILES'].replace('#', '\n')
    else:
        logger.info('Getting diff of branch.')
        cmd = ['git', 'diff', '--name-only', 'origin/master...']
        diff_files = run_command_get_output(cmd)
    logger.info('Diff files: ')
    logger.info(diff_files)
    modified_files = []
    # remove the deleted file.
    for diff_file in diff_files.splitlines():
        if os.path.exists(diff_file.strip()):
            modified_files.append(diff_file.strip())
    return modified_files


def analysis_diff():
    """Get modified files and their imported files modified modules
    """
    # ignore diff for constant define files, these files import by all pipeline, trainer
    ignore_files = [
        'modelscope/utils/constant.py', 'modelscope/metainfo.py',
        'modelscope/pipeline_inputs.py', 'modelscope/outputs/outputs.py'
    ]

    modified_register_modules = []
    modified_cases = []
    modified_files_imported_by = []
    modified_files = get_modified_files()
    logger.info('Modified files:\n %s' % '\n'.join(modified_files))

    logger.info('Starting get import map')
    import_map = get_import_map()
    logger.info('Finished get import map')
    for modified_file in modified_files:
        if ((modified_file.startswith('./modelscope')
             or modified_file.startswith('modelscope'))
                and modified_file not in ignore_files):  # is source file
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
    test_suite_full_paths = gather_test_suites_files()
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
    branch = get_current_branch()
    if branch == 'master':
        # when run with master, run all the cases
        return gather_test_suites_files(is_full_path=False)
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
    iic_models_info = get_models_info(['iic'])
    models_info = {}
    # compatible model info
    for model_id, model_info in iic_models_info.items():
        _, model_name = model_id_to_group_owner_name(model_id)
        models_info['damo/%s' % model_name] = model_info
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
                logger.warning('Pipeline task: %s has no test case!'
                               % affected_register_module[1])
        elif affected_register_module[0] == 'MODELS':
            # ["MODELS", "keyword_spotting", "kws_kwsbp", "GenericKeyWordSpotting"],
            # ["MODELS", task, model_name, model_class_name]
            if affected_register_module[1] in task_pipeline_test_suite_map:
                affected_pipeline_cases.extend(
                    task_pipeline_test_suite_map[affected_register_module[1]])
            else:
                logger.warning('Pipeline task: %s has no test case!'
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
            for model_id, model_info in models_info.items():
                if ('preprocessor_type' in model_info
                        and model_info['preprocessor_type'] is not None
                        and model_info['preprocessor_type']
                        == affected_register_module[2]):
                    task = model_info['task']
                    if task in task_pipeline_test_suite_map:
                        affected_pipeline_cases.extend(
                            task_pipeline_test_suite_map[task])
                    if model_id in model_trainer_map:
                        affected_trainer_cases.extend(
                            model_trainer_map[model_id])
        elif (affected_register_module[0] == 'HOOKS'
              or affected_register_module[0] == 'CUSTOM_DATASETS'):
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


def get_files_related_modules(files, reverse_import_map):
    register_modules = []
    for single_file in files:
        if single_file.startswith('./modelscope') or \
           single_file.startswith('modelscope'):
            register_modules.extend(get_file_register_modules(single_file))

    while len(register_modules) == 0:
        logger.warn('There is no affected register module')
        deeper_imported_by = []
        has_deeper_affected_files = False
        for source_file in files:
            if len(source_file.split('/')) > 4 and source_file.startswith(
                    'modelscope'):
                deeper_imported_by.extend(reverse_import_map[source_file])
                has_deeper_affected_files = True
        if not has_deeper_affected_files:
            break
        for file in deeper_imported_by:
            register_modules = get_file_register_modules(file)
        files = deeper_imported_by
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
        register_modules = get_files_related_modules(
            [f] + reverse_depend_map[f], reverse_depend_map)
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
