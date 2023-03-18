# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGPT3GenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        # please make sure this local path exists.
        self.model_id_1_3B = 'damo/nlp_gpt3_text-generation_1.3B'
        self.model_id_2_7B = 'damo/nlp_gpt3_text-generation_2.7B'
        self.model_id_13B = 'damo/nlp_gpt3_text-generation_13B'
        self.model_dir_13B = snapshot_download(self.model_id_13B)
        self.input = '好的'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt3_1_3B(self):
        pipe = pipeline(Tasks.text_generation, model=self.model_id_1_3B)
        print(pipe(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_gpt3_2_7B(self):
        pipe = pipeline(Tasks.text_generation, model=self.model_id_2_7B)
        print(pipe(self.input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt3_1_3B_with_args(self):
        pipe = pipeline(Tasks.text_generation, model=self.model_id_1_3B)
        print(pipe(self.input, top_p=0.9, temperature=0.9, max_length=32))

    @unittest.skip('distributed gpt3 13B, skipped')
    def test_gpt3_13B(self):
        """ The model can be downloaded from the link on
        TODO: add gpt3 checkpoint link
        After downloading, you should have a gpt3 model structure like this:
        nlp_gpt3_text-generation_13B
            |_ config.json
            |_ configuration.json
            |_ tokenizer.json
            |_ model <-- an empty directory

        Model binaries shall be downloaded separately to populate the model directory, so that
        the model directory would contain the following binaries:
            |_ model
                |_ mp_rank_00_model_states.pt
                |_ mp_rank_01_model_states.pt
                |_ mp_rank_02_model_states.pt
                |_ mp_rank_03_model_states.pt
                |_ mp_rank_04_model_states.pt
                |_ mp_rank_05_model_states.pt
                |_ mp_rank_06_model_states.pt
                |_ mp_rank_07_model_states.pt
        """
        pipe = pipeline(Tasks.text_generation, model=self.model_dir_13B)
        print(pipe(self.input))


if __name__ == '__main__':
    unittest.main()
