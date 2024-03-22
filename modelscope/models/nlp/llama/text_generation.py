# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers.models.llama import LlamaForCausalLM

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from .backbone import MsModelMixin


def get_chat_prompt(system: str, text: str, history: List[Tuple[str, str]],
                    max_length: int, tokenizer):
    system_prompt = f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n'
    system_ids = tokenizer(
        system_prompt, add_special_tokens=False, return_tensors='pt').input_ids

    text_prompt = f'{text.strip()} [/INST]'
    text_ids = tokenizer(
        text_prompt, add_special_tokens=False, return_tensors='pt').input_ids

    prompt_length = system_ids.shape[-1] + text_ids.shape[-1]
    if prompt_length > max_length:
        raise RuntimeError(
            f'prepend prompt length {prompt_length} is bigger than max_length {max_length}'
        )

    history_prompt = ''
    history_ids_list = []
    # traverse history in reverse order
    for user, bot in history[::-1]:
        assert isinstance(user, str)
        assert isinstance(bot, str)
        round_prompt = f'{user.strip()} [/INST] {bot.strip()} </s><s>[INST] '
        round_ids = tokenizer(
            round_prompt, add_special_tokens=False,
            return_tensors='pt').input_ids
        if prompt_length + round_ids.shape[-1] > max_length:
            # excess history should not be appended to the prompt
            break
        else:
            history_prompt = round_prompt + history_prompt
            history_ids_list = [round_ids] + history_ids_list
            prompt_length += round_ids.shape[-1]

    prompt_list = [system_prompt, history_prompt, text_prompt]
    prompt_ids_list = [system_ids] + history_ids_list + [text_ids]

    return ''.join(prompt_list), torch.cat(prompt_ids_list, dim=1)


# This file is mainly copied from the llama code of transformers
@MODELS.register_module(Tasks.chat, module_name=Models.llama2)
@MODELS.register_module(Tasks.chat, module_name=Models.llama)
@MODELS.register_module(Tasks.text_generation, module_name=Models.llama2)
@MODELS.register_module(Tasks.text_generation, module_name=Models.llama)
class LlamaForTextGeneration(MsModelMixin, LlamaForCausalLM, TorchModel):

    def chat(self, input: Dict, tokenizer) -> Dict:
        import copy
        gen_kwargs = copy.copy(input)
        if 'text' not in input:
            text: str = ''
        else:
            text: str = input['text']
            gen_kwargs.pop('text')

        if 'system' not in input:
            system: str = ''
        else:
            system: str = input['system']
            gen_kwargs.pop('system')

        if 'history' not in input:
            history = []
        else:
            history: List[Tuple] = copy.copy(input['history'])
            gen_kwargs.pop('history')

        if 'max_length' not in gen_kwargs:
            gen_kwargs['max_length'] = 4096

        prompt, prompt_ids = get_chat_prompt(
            system=system,
            text=text,
            history=history,
            max_length=gen_kwargs['max_length'],
            tokenizer=tokenizer)
        input_ids = prompt_ids.to(self.device)
        generate_ids = self.generate(input_ids, **gen_kwargs)
        # remove input tokens
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        response = response.strip()
        history.append((text, response))

        return {OutputKeys.RESPONSE: response, OutputKeys.HISTORY: history}
