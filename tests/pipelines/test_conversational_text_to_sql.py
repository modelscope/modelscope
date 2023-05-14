# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import StarForTextToSql
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import ConversationalTextToSqlPipeline
from modelscope.preprocessors import ConversationalTextToSqlPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.nlp.space_T_en.utils import \
    text2sql_tracking_and_print_results
from modelscope.utils.test_utils import test_level


class ConversationalTextToSql(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.table_question_answering
        self.model_id = 'damo/nlp_star_conversational-text-to-sql'

    model_id = 'damo/nlp_star_conversational-text-to-sql'
    test_case = {
        'database_id':
        'employee_hire_evaluation',
        'local_db_path':
        None,
        'utterance': [
            "I'd like to see Shop names.", 'Which of these are hiring?',
            'Which shop is hiring the highest number of employees? | do you want the name of the shop ? | Yes'
        ]
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = ConversationalTextToSqlPreprocessor(
            model_dir=cache_path,
            database_id=self.test_case['database_id'],
            db_content=True)
        model = StarForTextToSql(
            model_dir=cache_path, config=preprocessor.config)

        pipelines = [
            ConversationalTextToSqlPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(task=self.task, model=model, preprocessor=preprocessor)
        ]
        text2sql_tracking_and_print_results(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = ConversationalTextToSqlPreprocessor(
            model_dir=model.model_dir)

        pipelines = [
            ConversationalTextToSqlPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(task=self.task, model=model, preprocessor=preprocessor)
        ]
        text2sql_tracking_and_print_results(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipelines = [pipeline(task=self.task, model=self.model_id)]
        text2sql_tracking_and_print_results(self.test_case, pipelines)


if __name__ == '__main__':
    unittest.main()
