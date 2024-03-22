import os

import json

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.hub.utils.utils import get_cache_dir
from modelscope.pipelines import pipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.input_output import (
    call_pipeline_with_json, get_pipeline_information_by_pipeline,
    get_task_input_examples, pipeline_output_to_service_base64_output)


class ModelJsonTest:

    def __init__(self):
        self.api = HubApi()

    def test_single(self, model_id: str, model_revision=None):
        # get model_revision & task info
        cache_root = get_cache_dir()
        configuration_file = os.path.join(cache_root, model_id,
                                          ModelFile.CONFIGURATION)
        if not model_revision:
            model_revision = self.api.list_model_revisions(
                model_id=model_id)[0]
        if not os.path.exists(configuration_file):

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


if __name__ == '__main__':
    model_list = [
        'qwen/Qwen-7B-Chat-Int4',
        'qwen/Qwen-14B-Chat-Int4',
        'baichuan-inc/Baichuan2-7B-Chat-4bits',
        'baichuan-inc/Baichuan2-13B-Chat-4bits',
        'ZhipuAI/chatglm2-6b-int4',
    ]
    tester = ModelJsonTest()
    for model in model_list:
        try:
            res = tester.test_single(model)
            print(
                f'\nmodel_id {model} call_pipeline_with_json run ok. {res}\n\n\n\n'
            )
        except BaseException as e:
            print(
                f'\nmodel_id {model} call_pipeline_with_json run failed: {e}.\n\n\n\n'
            )
