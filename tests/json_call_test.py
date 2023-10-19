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
            task=task, model=model_id, model_revision=model_revision)
        pipeline_info = get_pipeline_information_by_pipeline(ppl)

        # call pipeline
        data = get_task_input_examples(task)
        print(task, data)
        infer_result = call_pipeline_with_json(pipeline_info, ppl, data)
        result = pipeline_output_to_service_base64_output(task, infer_result)
        return result


if __name__ == '__main__':
    model_list = [
        'damo/nlp_structbert_nli_chinese-base',
        'damo/nlp_structbert_word-segmentation_chinese-base',
        'damo/nlp_structbert_zero-shot-classification_chinese-base',
        'damo/cv_unet_person-image-cartoon_compound-models',
        'damo/nlp_structbert_sentiment-classification_chinese-tiny',
        'damo/nlp_csanmt_translation_zh2en',
        'damo/nlp_rom_passage-ranking_chinese-base',
        'damo/ofa_image-caption_muge_base_zh',
        'damo/nlp_raner_named-entity-recognition_chinese-base-ecom-50cls',
        'damo/nlp_structbert_sentiment-classification_chinese-ecommerce-base',
        'damo/text-to-video-synthesis',
        'qwen/Qwen-7B',
        'qwen/Qwen-7B-Chat',
        'ZhipuAI/ChatGLM-6B',
    ]
    tester = ModelJsonTest()
    for model in model_list:
        try:
            res = tester.test_single(model)
            print(f'\nmodel_id {model} call_pipeline_with_json run ok.\n')
        except BaseException as e:
            print(
                f'\nmodel_id {model} call_pipeline_with_json run failed: {e}.\n'
            )
