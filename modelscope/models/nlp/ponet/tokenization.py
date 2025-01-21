# Copyright 2021-2022 The Alibaba DAMO Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# All rights reserved.
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
"""Tokenization classes for PoNet """

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from transformers.file_utils import PaddingStrategy
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.tokenization_utils import BatchEncoding, EncodedInput

from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()

VOCAB_FILES_NAMES = {'vocab_file': ModelFile.VOCAB_FILE}

PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {}}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'nlp_ponet_fill-mask_chinese-base': 512,
    'nlp_ponet_fill-mask_english-base': 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    'nlp_ponet_fill-mask_chinese-base': {
        'do_lower_case': True
    },
    'nlp_ponet_fill-mask_english-base': {
        'do_lower_case': True
    },
}


class PoNetTokenizer(BertTokenizer):
    r"""
    Construct an PoNet tokenizer. Based on BertTokenizer.

    This tokenizer inherits from :class:`~transformers.BertTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or
            batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning
            attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (
                max_length % pad_to_multiple_of != 0):
            max_length = (
                (max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(
            required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [1] * len(
                        required_input) + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = (
                        encoded_inputs['token_type_ids']
                        + [self.pad_token_type_id] * difference)
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs[
                        'special_tokens_mask'] + [1] * difference
                if 'segment_ids' in encoded_inputs:
                    encoded_inputs[
                        'segment_ids'] = encoded_inputs['segment_ids'] + [
                            encoded_inputs['segment_ids'][-1] + 1
                        ] * difference  # noqa *
                encoded_inputs[self.model_input_names[
                    0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [0] * difference + [
                        1
                    ] * len(required_input)
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs['token_type_ids']
                if 'segment_ids' in encoded_inputs:
                    encoded_inputs['segment_ids'] = [encoded_inputs['segment_ids'][-1] + 1] * difference + \
                                                    encoded_inputs['segment_ids']  # noqa *
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [
                        1
                    ] * difference + encoded_inputs['special_tokens_mask']
                encoded_inputs[self.model_input_names[
                    0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError('Invalid padding strategy:'
                                 + str(self.padding_side))
        elif return_attention_mask and 'attention_mask' not in encoded_inputs:
            encoded_inputs['attention_mask'] = [1] * len(required_input)

        return encoded_inputs
