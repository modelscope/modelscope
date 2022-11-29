# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGPTMoEGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id_1_3B_MoE32 = 'PAI/nlp_gpt3_text-generation_1.3B_MoE-32'
        self.model_dir_1_3B_MoE32 = snapshot_download(self.model_id_1_3B_MoE32)
        self.input = '好的'

    @unittest.skip('distributed gpt-moe 1.3B_MoE-32, skipped')
    def test_gpt_moe_1_3B_MoE32(self):
        pipe = pipeline(Tasks.text_generation, model=self.model_id_1_3B_MoE32)
        print(pipe(self.input))


if __name__ == '__main__':
    unittest.main()
