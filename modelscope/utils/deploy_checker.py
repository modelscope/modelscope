import argparse
import traceback
from typing import List, Union

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.pipelines import pipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.input_output import (
    call_pipeline_with_json, get_pipeline_information_by_pipeline,
    get_task_input_examples, pipeline_output_to_service_base64_output)
from modelscope.utils.logger import get_logger

logger = get_logger()


class DeployChecker:

    def __init__(self):
        self.api = HubApi()

    def check_model(self, model_id: str, model_revision=None):
        # get model_revision & task info
        if not model_revision:
            model_revisions = self.api.list_model_revisions(model_id)
            logger.info(
                f'All model_revisions of `{model_id}`: {model_revisions}')
            if len(model_revisions):
                model_revision = model_revisions[0]
            else:
                logger.error(f'{model_id} has no revision.')

        configuration_file = model_file_download(
            model_id=model_id,
            file_path=ModelFile.CONFIGURATION,
            revision=model_revision)
        cfg = Config.from_file(configuration_file)
        task = cfg.safe_get('task')

        # init pipeline
        ppl = pipeline(
            task=task,
            model=model_id,
            model_revision=model_revision,
            llm_first=True)
        pipeline_info = get_pipeline_information_by_pipeline(ppl)

        # call pipeline
        data = get_task_input_examples(task)

        infer_result = call_pipeline_with_json(pipeline_info, ppl, data)
        result = pipeline_output_to_service_base64_output(task, infer_result)
        return result


def check_deploy(models: Union[str, List], revisions: Union[str, List] = None):
    if not isinstance(models, list):
        models = [models]
    if not isinstance(revisions, list):
        revisions = [revisions] * (1 if revisions else len(models))

    if len(models) != len(revisions):
        logger.error(
            f'The number of models and revisions need to be equal: The number of models'
            f' is {len(model)} while the number of revisions is {len(revision)}.'
        )

    checker = DeployChecker()
    for model, revision in zip(models, revisions):
        try:
            res = checker.check_model(model, revision)
            logger.info(f'{model} {revision}: Deploy pre-check pass. {res}\n')
        except BaseException as e:
            logger.info(
                f'{model} {revision}: Deploy pre-check failed: {e}. {traceback.print_exc()}\n'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--revision', type=str, default=None)
    args = parser.parse_args()

    check_deploy(args.model_id, args.revision)
