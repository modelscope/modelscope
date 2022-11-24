# Copyright (c) 2022 Zhipu.AI
import copy
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .codegeex import CodeGeeXModel
from .inference import get_token_stream
from .tokenizer import CodeGeeXTokenizer


def model_provider():
    """Build the model."""

    hidden_size = 5120
    num_attention_heads = 40
    num_layers = 39
    padded_vocab_size = 52224
    max_position_embeddings = 2048

    model = CodeGeeXModel(hidden_size, num_layers, num_attention_heads,
                          padded_vocab_size, max_position_embeddings)

    return model


@MODELS.register_module(Tasks.code_translation, module_name=Models.codegeex)
class CodeGeeXForCodeTranslation(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the fast poem model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        logger = get_logger()
        # loading tokenizer
        logger.info('Loading tokenizer ...')
        self.tokenizer = CodeGeeXTokenizer(
            tokenizer_path=model_dir + '/tokenizer', mode='codegeex-13b')
        # loading model
        state_dict_path = model_dir + '/ckpt_ms_translation_0817.pt'
        logger.info('Loading state dict ...')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        state_dict = state_dict['module']

        logger.info('Building CodeGeeX model ...')
        self.model = model_provider()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.half()
        self.model.cuda()

    def forward(self, input: Dict[str, str]) -> Dict[str, str]:
        micro_batch_size = 1
        seq_length = 2048
        out_seq_length = 256
        bad_ids = None
        src_lang = input['source language']
        dst_lang = input['target language']
        prompt = input['prompt']
        prompt = f'code translation\n{src_lang}:\n{prompt}\n{dst_lang}:\n'
        logger = get_logger()
        tokenizer = self.tokenizer
        model = self.model
        for prompt in [prompt]:
            tokens = tokenizer.encode_code(prompt)
            n_token_prompt = len(tokens)
            token_stream = get_token_stream(
                model,
                tokenizer,
                seq_length,
                out_seq_length,
                [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
                micro_batch_size=micro_batch_size,
                bad_ids=bad_ids,
                greedy=True,
            )
            is_finished = [False for _ in range(micro_batch_size)]
            for i, generated in enumerate(token_stream):
                generated_tokens = generated[0]
                for j in range(micro_batch_size):
                    if is_finished[j]:
                        continue
                    if generated_tokens[j].cpu().numpy(
                    )[-1] == tokenizer.eos_token_id or len(
                            generated_tokens[j]) >= out_seq_length:
                        is_finished[j] = True
                        generated_tokens_ = generated_tokens[j].cpu().numpy(
                        ).tolist()
                        generated_code = tokenizer.decode_code(
                            generated_tokens_[n_token_prompt:])
                        generated_code = ''.join(generated_code)
                        logger.info(
                            '================================= Generated code:'
                        )
                        logger.info(generated_code)
                    if all(is_finished):
                        break

        logger.info('Generation finished.')
        return {OutputKeys.TEXT: generated_code}
