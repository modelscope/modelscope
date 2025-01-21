# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import unittest
from threading import Thread

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRetrievalPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DocumentGroundedDialogRetrievalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.document_grounded_dialog_retrieval
        self.model_id = 'DAMO_ConvAI/nlp_convai_retrieval_pretrain'

    param = {
        'query': [
            '<last_turn>我想知道孩子如果出现阑尾炎的话会怎么样',
            '<last_turn>好像是从肚脐开始，然后到右下方<system>您可以描述一下孩子的情况吗？<user>我想知道孩子如果出现阑尾炎的话会怎么样？',
        ],
        'positive': ['阑尾炎', '肚脐开始'],
        'negative': [
            '肠胃炎',
            '肚脐为止',
        ]
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id, revision='v1.0.0')
        preprocessor = DocumentGroundedDialogRetrievalPreprocessor(
            model_dir=cache_path)
        pipeline_ins = pipeline(
            Tasks.document_grounded_dialog_retrieval,
            model=cache_path,
            preprocessor=preprocessor)
        print(pipeline_ins(self.param))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download_with_multithreads(self):
        cache_path = snapshot_download(self.model_id, revision='v1.0.0')
        pl = pipeline(
            Tasks.document_grounded_dialog_retrieval, model=cache_path)

        def print_func(pl, i):
            result = pl(self.param)
            print(i, result)

        procs = []
        for i in range(5):
            proc = Thread(target=print_func, args=(pl, i))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id, revision='v1.0.0')
        preprocessor = DocumentGroundedDialogRetrievalPreprocessor(
            model_dir=model.model_dir)
        pipeline_ins = pipeline(
            Tasks.document_grounded_dialog_retrieval,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(self.param))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(
            task=Tasks.document_grounded_dialog_retrieval,
            model_revision='v1.0.0')
        print(pipeline_ins(self.param))


if __name__ == '__main__':
    unittest.main()
