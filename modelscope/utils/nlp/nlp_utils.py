import os.path as osp
from typing import List

from modelscope.outputs import OutputKeys
from modelscope.pipelines.nlp import (ConversationalTextToSqlPipeline,
                                      DialogStateTrackingPipeline,
                                      TableQuestionAnsweringPipeline)


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


def tableqa_tracking_and_print_results(
        test_case, pipelines: List[TableQuestionAnsweringPipeline]):
    for pipeline in pipelines:
        historical_queries = None
        for question in test_case['utterance']:
            output_dict = pipeline({
                'question': question,
                'history_sql': historical_queries
            })
            print('output_dict', output_dict['output'].string,
                  output_dict['output'].query)
            historical_queries = output_dict['history']
