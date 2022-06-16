# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

from maas_hub.snapshot_download import snapshot_download

from modelscope.models.nlp import MaskedLanguageModel
from modelscope.pipelines import FillMaskPipeline, pipeline
from modelscope.preprocessors import FillMaskPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.models import Model
from modelscope.utils.hub import get_model_cache_dir
from modelscope.utils.test_utils import test_level

class FillMaskTest(unittest.TestCase):
    model_id_sbert = {'zh': 'damo/nlp_structbert_fill-mask-chinese_large',
                      'en': 'damo/nlp_structbert_fill-mask-english_large'}
    model_id_veco = 'damo/nlp_veco_fill-mask_large'

    ori_texts = {"zh": "段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。你师父差得动你，你师父可差不动我。",
                 "en": "Everything in what you call reality is really just a reflection of your consciousness. Your whole universe is just a mirror reflection of your story."}

    test_inputs = {"zh": "段誉轻[MASK]折扇，摇了摇[MASK]，[MASK]道：“你师父是你的[MASK][MASK]，你师父可不是[MASK]的师父。你师父差得动你，你师父可[MASK]不动我。",
                  "en": "Everything in [MASK] you call reality is really [MASK] a reflection of your [MASK]. Your whole universe is just a mirror [MASK] of your story."}

    #def test_run(self):
    #    # sbert
    #    for language in ["zh", "en"]:
    #        model_dir = snapshot_download(self.model_id_sbert[language])
    #        preprocessor = FillMaskPreprocessor(
    #            model_dir, first_sequence='sentence', second_sequence=None)
    #        model = MaskedLanguageModel(model_dir)
    #        pipeline1 = FillMaskPipeline(model, preprocessor)
    #        pipeline2 = pipeline(
    #            Tasks.fill_mask, model=model, preprocessor=preprocessor)
    #        ori_text = self.ori_texts[language]
    #        test_input = self.test_inputs[language]
    #        print(
    #            f'ori_text: {ori_text}\ninput: {test_input}\npipeline1: {pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}'
    #        )

        ## veco
        #model_dir = snapshot_download(self.model_id_veco)
        #preprocessor = FillMaskPreprocessor(
        #    model_dir, first_sequence='sentence', second_sequence=None)
        #model = MaskedLanguageModel(model_dir)
        #pipeline1 = FillMaskPipeline(model, preprocessor)
        #pipeline2 = pipeline(
        #    Tasks.fill_mask, model=model, preprocessor=preprocessor)
        #for language in ["zh", "en"]:
        #    ori_text = self.ori_texts[language]
        #    test_input = self.test_inputs["zh"].replace("[MASK]", "<mask>")
        #    print(
        #        f'ori_text: {ori_text}\ninput: {test_input}\npipeline1: {pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}'


    def test_run_with_model_from_modelhub(self):
        for language in ["zh"]:
            print(self.model_id_sbert[language])
            model = Model.from_pretrained(self.model_id_sbert[language])
            print("model", model.model_dir)
            preprocessor = FillMaskPreprocessor(
                model.model_dir, first_sequence='sentence', second_sequence=None)
            pipeline_ins = pipeline(
                task=Tasks.fill_mask, model=model,  preprocessor=preprocessor)
            print(pipeline_ins(self_test_inputs[language]))


    #def test_run_with_model_name(self):
        ## veco
        #pipeline_ins = pipeline(
        #    task=Tasks.fill_mask, model=self.model_id_veco)
        #for language in ["zh", "en"]:
        #    input_ = self.test_inputs[language].replace("[MASK]", "<mask>")
        #    print(pipeline_ins(input_))

        ## structBert
        #for language in ["zh"]:
        #    pipeline_ins = pipeline(
        #        task=Tasks.fill_mask, model=self.model_id_sbert[language])
        #    print(pipeline_ins(self_test_inputs[language]))


if __name__ == '__main__':
    unittest.main()

