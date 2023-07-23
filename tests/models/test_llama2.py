import unittest

import torch

from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer
from modelscope.utils.test_utils import test_level


class Llama2Test(unittest.TestCase):

    def setUp(self) -> None:
        self.model_name = 'modelscope/Llama-2-7b-chat-ms'
        self.system = 'you are a helpful assistant!'
        self.utterance = 'hello'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_chat(self):
        model_dir = snapshot_download(
            self.model_name, ignore_file_pattern=[r'\w+\.safetensors'])
        model = Model.from_pretrained(
            model_dir, device_map='auto', torch_dtype=torch.float16)
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)

        inputs = {
            'utterance': self.utterance,
            'history': [],
            'system': self.system
        }
        result = model.chat(input=inputs, tokenizer=tokenizer)
        assert isinstance(result['history'], list)
        assert result['history'][0][0] == self.utterance


if __name__ == '__main__':
    unittest.main()
