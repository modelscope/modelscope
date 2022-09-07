import os.path as osp
from typing import List

from modelscope.outputs import OutputKeys
from modelscope.pipelines.nlp import (ConversationalTextToSqlPipeline,
                                      DialogStateTrackingPipeline)


def text2sql_tracking_and_print_results(
        test_case, pipelines: List[ConversationalTextToSqlPipeline]):
    for p in pipelines:
        last_sql, history = '', []
        for item in test_case['utterance']:
            case = {
                'utterance': item,
                'history': history,
                'last_sql': last_sql,
                'database_id': test_case['database_id'],
                'local_db_path': test_case['local_db_path']
            }
            results = p(case)
            print({'question': item})
            print(results)
            last_sql = results['text']
            history.append(item)


def tracking_and_print_dialog_states(
        test_case, pipelines: List[DialogStateTrackingPipeline]):
    import json
    pipelines_len = len(pipelines)
    history_states = [{}]
    utter = {}
    for step, item in enumerate(test_case):
        utter.update(item)
        result = pipelines[step % pipelines_len]({
            'utter':
            utter,
            'history_states':
            history_states
        })
        print(json.dumps(result))

        history_states.extend([result[OutputKeys.OUTPUT], {}])


def import_external_nltk_data(nltk_data_dir, package_name):
    """import external nltk_data, and extract nltk zip package.

    Args:
        nltk_data_dir (str): external nltk_data dir path, eg. /home/xx/nltk_data
        package_name (str): nltk package name, eg. tokenizers/punkt
    """
    import nltk
    nltk.data.path.append(nltk_data_dir)

    filepath = osp.join(nltk_data_dir, package_name + '.zip')
    zippath = osp.join(nltk_data_dir, package_name)
    packagepath = osp.dirname(zippath)
    if not osp.exists(zippath):
        import zipfile
        with zipfile.ZipFile(filepath) as zf:
            zf.extractall(osp.join(packagepath))
