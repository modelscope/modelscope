import unittest

import torch

from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer
from modelscope.utils.test_utils import test_level


class Llama2Test(unittest.TestCase):

    def setUp(self) -> None:
        self.model_name = 'modelscope/Llama-2-7b-chat-ms'
        self.system = 'you are a helpful assistant!'
        self.text_first_round = 'hello'
        self.text_second_round = 'do you know peking university?'
        self.text_third_round = 'where is it?'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_chat(self):
        model_dir = snapshot_download(
            self.model_name, ignore_file_pattern=[r'\w+\.safetensors'])
        model = Model.from_pretrained(
            model_dir, device_map='auto', torch_dtype=torch.float16)
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)

        inputs = {
            'text': self.text_first_round,
            'history': [],
            'system': self.system
        }
        result = model.chat(input=inputs, tokenizer=tokenizer)
        self.assertIsInstance(result['history'], list)
        self.assertEqual(len(result['history']), 1)
        self.assertEqual(result['history'][0][0], self.text_first_round)

        inputs = {
            'text': self.text_second_round,
            'history': result['history'],
            'system': self.system
        }
        result = model.chat(input=inputs, tokenizer=tokenizer)
        self.assertIsInstance(result['history'], list)
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(result['history'][1][0], self.text_second_round)

        inputs = {
            'text': self.text_third_round,
            'history': result['history'],
            'system': self.system
        }
        result = model.chat(input=inputs, tokenizer=tokenizer)
        self.assertIsInstance(result['history'], list)
        self.assertEqual(len(result['history']), 3)
        self.assertEqual(result['history'][2][0], self.text_third_round)


if __name__ == '__main__':
    unittest.main()
