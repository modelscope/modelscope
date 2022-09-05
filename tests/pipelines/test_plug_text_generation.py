# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class TextPlugGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        # please make sure this local path exists.
        self.model_id = 'damo/nlp_plug_text-generation_27B'
        self.model_dir = snapshot_download(self.model_id)
        self.plug_input = '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。"'

    @unittest.skip('distributed plug, skipped')
    def test_plug(self):
        """ The model can be downloaded from the link on
        https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary.
        After downloading, you should have a plug model structure like this:
        nlp_plug_text-generation_27B
            |_ config.json
            |_ configuration.json
            |_ ds_zero-offload_10B_config.json
            |_ vocab.txt
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
        # download model binaries to <model_dir>/model
        pipe = pipeline(Tasks.text_generation, model=self.model_id)
        print(
            f'input: {self.plug_input}\noutput: {pipe(self.plug_input, out_length=256)}'
        )


if __name__ == '__main__':
    unittest.main()
