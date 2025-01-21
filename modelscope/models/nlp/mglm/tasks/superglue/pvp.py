# Copyright (c) 2022 Zhipu.AI
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
"""
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import copy
import math
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from tasks.data_utils import (InputExample, build_decoder_input,
                              build_decoder_sample, build_input_from_ids,
                              build_sample, num_special_tokens_to_add)
from utils import print_rank_0

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]],
                      List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self,
                 args,
                 tokenizer,
                 label_list,
                 max_seq_length,
                 pattern_id: int = 0,
                 verbalizer_file: str = None,
                 seed: int = 42,
                 is_multi_token=False,
                 max_segment_length=0,
                 fast_decode: bool = False,
                 split='train',
                 num_prompt_tokens=0):
        """
        Create a new PVP.

        :param args: the args
        :param tokenizer: the tokenizer
        :param label_list: the list of labels
        :param max_seq_length: the maximum length of the sequence
        :param pattern_id: the pattern id to use
        :param seed: a seed to be used for generating random numbers if necessary
        :param is_multi_token: if the verbalizers contain multiple tokens
        :param fast_decode: whether to use the fast decode mode for multi-token tasks
        :param continuous_prompt: whether to use continuous prompt optimization
        """
        self.args = args
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.pattern_id = pattern_id
        self.num_prompt_tokens = num_prompt_tokens
        self.rng = random.Random(seed)
        self.num_truncated = 0
        self.fast_decode = fast_decode
        self.split = split
        self.max_dec_seq_length = 16
        self._is_multi_token = is_multi_token
        self.max_segment_length = max_segment_length
        self.task_mask = args.task_mask
        self.continuous_prompt = args.continuous_prompt
        self.prefix_prompt = args.prefix_prompt
        if self.continuous_prompt:
            print_rank_0(
                f'Prompt tokens in pvp {self.num_prompt_tokens} spell length {self.spell_length}'
            )

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(
                verbalizer_file, self.pattern_id)

    @property
    def is_multi_token(self):
        return self._is_multi_token

    @property
    def spell_length(self):
        return 0

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.get_command('MASK').Id

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.get_command('MASK').Id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    @staticmethod
    def uppercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.uppercase_first(s[0]), s[1]
        return s[0].upper() + s[1:]

    @staticmethod
    def available_patterns():
        return [0]

    def replace_prompt_tokens(self, parts_a, parts_b):
        if not self.continuous_prompt:
            parts_a = [part for part in parts_a if part is not None]
            parts_b = [part for part in parts_b if part is not None]
            return parts_a, parts_b
        num_prompt_tokens = self.num_prompt_tokens
        num_pos = 0
        for parts in (parts_a, parts_b):
            for part in parts:
                if part is None:
                    num_pos += 1
        avg_prompt_tokens = math.ceil(num_prompt_tokens / num_pos)
        new_parts_a, new_parts_b = [], []
        for part in parts_a:
            if part is None:
                if num_prompt_tokens > 0:
                    if num_prompt_tokens >= avg_prompt_tokens:
                        new_parts_a.append(avg_prompt_tokens)
                        num_prompt_tokens -= avg_prompt_tokens
                    else:
                        new_parts_a.append(num_prompt_tokens)
                        num_prompt_tokens = 0
            else:
                new_parts_a.append(part)
        for part in parts_b:
            if part is None:
                if num_prompt_tokens > 0:
                    if num_prompt_tokens >= avg_prompt_tokens:
                        new_parts_b.append(avg_prompt_tokens)
                        num_prompt_tokens -= avg_prompt_tokens
                    else:
                        new_parts_b.append(num_prompt_tokens)
                        num_prompt_tokens = 0
            else:
                new_parts_b.append(part)
        return new_parts_a, new_parts_b

    def encode(self,
               example: InputExample,
               priming: bool = False,
               labeled: bool = False):
        """
        Encode an input example using this pattern-verbalizer pair.

        Args:
            example: the input example to encode
            priming: whether to use this example for priming
            labeled: if ``priming=True``, whether the label should be appended to this example

        Returns:
            A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.tokenizer
        raw_parts_a, raw_parts_b = self.get_parts(example)

        raw_parts_a = [
            x if isinstance(x, tuple) else (x, False) for x in raw_parts_a
        ]
        prompt_id = tokenizer.num_tokens

        def encode_input(raw_parts):
            parts = []
            for x, s in raw_parts:
                if isinstance(x, str):
                    x = tokenizer.EncodeAsIds(x)
                elif isinstance(x, int):
                    x = [prompt_id] * x
                else:
                    pass
                parts.append((x, s))
            return parts

        parts_a = encode_input(raw_parts_a)
        if self.prefix_prompt > 0:
            parts_a = [([prompt_id] * self.prefix_prompt, False)] + parts_a

        parts_b = None
        if raw_parts_b:
            raw_parts_b = [
                x if isinstance(x, tuple) else (x, False) for x in raw_parts_b
            ]
            parts_b = encode_input(raw_parts_b)

        if self.is_multi_token:
            answers = self.get_answers(example)
            if example.label is not None:
                label = self.label_list.index(example.label)
            else:
                label = 0

            if not self.fast_decode:
                ids_list, positions_list, sep_list, mask_list, target_list, prompt_list = [], [], [], [], [], []
                segment_id_list = []
                if priming:
                    answer = answers[label]
                    answer_ids = get_verbalization_ids(
                        answer, tokenizer, force_single_token=False)
                    self.num_truncated += self.truncate(
                        parts_a,
                        parts_b,
                        answer_ids,
                        max_length=self.max_seq_length)
                    tokens_a = [
                        token_id for part, _ in parts_a for token_id in part
                    ]
                    tokens_b = [
                        token_id for part, _ in parts_b for token_id in part
                    ] if parts_b else None
                    input_ids = tokens_a
                    if tokens_b:
                        input_ids += tokens_b
                    if labeled:
                        mask_idx = input_ids.index(self.mask_id)
                        input_ids = input_ids[:
                                              mask_idx] + answer_ids + input_ids[
                                                  mask_idx + 1:]
                    return input_ids
                else:
                    for idx, answer in enumerate(answers):
                        this_parts_a, this_parts_b = copy.deepcopy(
                            parts_a), copy.deepcopy(parts_b)
                        answer_ids = get_verbalization_ids(
                            answer, tokenizer, force_single_token=False)
                        answer_ids = answer_ids + [
                            tokenizer.get_command('eop').Id
                        ]
                        self.num_truncated += self.truncate(
                            this_parts_a,
                            this_parts_b,
                            answer_ids,
                            max_length=self.max_seq_length)
                        tokens_a = [
                            token_id for part, _ in this_parts_a
                            for token_id in part
                        ]
                        tokens_b = [
                            token_id for part, _ in this_parts_b
                            for token_id in part
                        ] if parts_b else None
                        if self.max_segment_length > 0:
                            num_segments = (len(answer_ids)
                                            - 1) // self.max_segment_length + 1
                            segments = [
                                answer_ids[index
                                           * self.max_segment_length:(index
                                                                      + 1)
                                           * self.max_segment_length]
                                for index in range(num_segments)
                            ]
                            segment_id_list += [idx] * len(segments)
                        else:
                            segments = [answer_ids]
                        for segment in segments:
                            data = build_input_from_ids(
                                tokens_a,
                                tokens_b,
                                segment,
                                self.max_seq_length,
                                self.tokenizer,
                                args=self.args,
                                add_cls=True,
                                add_sep=False,
                                add_piece=True,
                                mask_id=self.mask_id)
                            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
                            prompt_pos = [
                                idx for idx, token in enumerate(ids)
                                if token == prompt_id
                            ]
                            ids = [
                                idx if idx != prompt_id else 0 for idx in ids
                            ]
                            prompt_list.append(prompt_pos)
                            ids_list.append(ids)
                            positions_list.append(position_ids)
                            sep_list.append(sep)
                            target_list.append(target_ids)
                            mask_list.append(loss_masks)
                            if self.mask in tokens_a:
                                mask_pos = tokens_a.index(self.mask)
                                tokens_a = tokens_a[:
                                                    mask_pos] + segment + tokens_a[
                                                        mask_pos:]
                            else:
                                mask_pos = tokens_b.index(self.mask)
                                tokens_b = tokens_b[:
                                                    mask_pos] + segment + tokens_b[
                                                        mask_pos:]
                    segment_id_list = segment_id_list if segment_id_list else None
                    sample = build_sample(
                        ids_list,
                        positions=positions_list,
                        masks=sep_list,
                        label=label,
                        logit_mask=mask_list,
                        target=target_list,
                        unique_id=example.guid,
                        segment_ids=segment_id_list,
                        prompt_ids=prompt_list)
                    return sample
            else:
                this_parts_a, this_parts_b = copy.deepcopy(
                    parts_a), copy.deepcopy(parts_b)
                self.num_truncated += self.truncate(
                    this_parts_a,
                    this_parts_b,
                    None,
                    max_length=self.max_seq_length)
                tokens_a = [
                    token_id for part, _ in this_parts_a for token_id in part
                ]
                tokens_b = [
                    token_id for part, _ in this_parts_b for token_id in part
                ] if parts_b else None
                data = build_input_from_ids(
                    tokens_a,
                    tokens_b,
                    None,
                    self.max_seq_length,
                    self.tokenizer,
                    args=self.args,
                    add_cls=True,
                    add_sep=False,
                    add_piece=False)
                ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
                sample = build_sample(
                    ids,
                    positions=position_ids,
                    masks=sep,
                    label=label,
                    unique_id=example.guid)

                ids_list, positions_list, mask_list, target_list, logit_mask_list = [], [], [], [], []
                for answer in answers:
                    answer_ids = get_verbalization_ids(
                        answer, tokenizer, force_single_token=False)
                    answer_ids = answer_ids + [tokenizer.get_command('eop').Id]
                    answer_ids = answer_ids[:self.max_dec_seq_length]
                    data = build_decoder_input(ids, answer_ids,
                                               self.max_seq_length,
                                               self.max_dec_seq_length,
                                               tokenizer)
                    dec_ids, _, _, dec_position_ids, _, dec_target_ids, dec_loss_masks = data
                    ids_list.append(dec_ids)
                    positions_list.append(dec_position_ids)
                    mask_list.append(sep)
                    target_list.append(dec_target_ids)
                    logit_mask_list.append(dec_loss_masks)

                sample = build_decoder_sample(sample, ids_list, positions_list,
                                              mask_list, target_list,
                                              logit_mask_list)
                return sample

        else:
            self.num_truncated += self.truncate(
                parts_a, parts_b, [], max_length=self.max_seq_length)

            tokens_a = [token_id for part, _ in parts_a for token_id in part]
            tokens_b = [token_id for part, _ in parts_b
                        for token_id in part] if parts_b else None
            if priming:
                input_ids = tokens_a
                if tokens_b:
                    input_ids += tokens_b
                if labeled:
                    mask_idx = input_ids.index(self.mask_id)
                    verbalizer = self.verbalize(example.label)
                    assert len(
                        verbalizer
                    ) == 1, 'priming only supports one verbalization per label'
                    verbalizer = verbalizer[0]
                    verbalizer_id = get_verbalization_ids(
                        verbalizer, self.tokenizer, force_single_token=True)
                    input_ids[mask_idx] = verbalizer_id
                return input_ids
            data = build_input_from_ids(
                tokens_a,
                tokens_b,
                None,
                self.max_seq_length,
                self.tokenizer,
                args=self.args,
                add_cls=True,
                add_sep=False,
                add_piece=True)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            prompt_pos = [
                idx for idx, token in enumerate(ids) if token == prompt_id
            ]
            ids = [token if token != prompt_id else 0 for token in ids]
            target_ids = self.get_verbalizer_ids()
            if example.label is not None:
                label = self.label_list.index(example.label)
            else:
                label = 0
            sample = build_sample(
                ids=ids,
                positions=position_ids,
                target=target_ids,
                masks=sep,
                logit_mask=loss_masks,
                label=label,
                unique_id=example.guid,
                prompt_ids=prompt_pos)
            return sample

    @staticmethod
    def _seq_length(parts: List[Tuple[List[int], bool]],
                    only_shortenable: bool = False):
        return sum([
            len(x) for x, shortenable in parts
            if not only_shortenable or shortenable
        ]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[List[int], bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts)
                       if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[List[int], bool]],
                 parts_b: List[Tuple[List[int], bool]], answer: List[int],
                 max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        if answer:
            total_len += len(answer)
        total_len += num_special_tokens_to_add(
            parts_a,
            parts_b,
            answer,
            add_cls=True,
            add_sep=False,
            add_piece=True)
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return False

        for _ in range(num_tokens_to_remove):
            if self._seq_length(
                    parts_a, only_shortenable=True) > self._seq_length(
                        parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)
        return True

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        Args:
            example: the input example to process
        Returns:
            Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    def get_answers(self, example: InputExample):
        return [self.verbalize(label)[0] for label in self.label_list]

    def get_verbalizer_ids(self):
        target_ids = []
        for label in self.label_list:
            verbalizer = self.verbalize(label)[0]
            verbalizer_id = get_verbalization_ids(
                verbalizer, self.tokenizer, force_single_token=True)
            target_ids.append(verbalizer_id)
        return target_ids

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(
            dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        print_rank_0(
            'Automatically loaded the following verbalizer: \n {}'.format(
                verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class CopaPVP(PVP):

    @staticmethod
    def available_patterns():
        return [0, 1]

    @property
    def is_multi_token(self):
        return True

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        mask_token = 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        mask_token = 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def get_answers(self, example: InputExample):
        choice1 = ' ' + self.remove_final_punc(
            self.lowercase_first(example.meta['choice1']))
        choice2 = ' ' + self.remove_final_punc(
            self.lowercase_first(example.meta['choice2']))
        return [choice1, choice2]

    def get_parts(self, example: InputExample) -> FilledPattern:
        assert self.pattern_id in [0, 1, 2, 3]
        premise = self.remove_final_punc(
            self.shortenable(' ' + example.text_a))
        choice1 = self.remove_final_punc(
            self.lowercase_first(example.meta['choice1']))
        choice2 = self.remove_final_punc(
            self.lowercase_first(example.meta['choice2']))

        question = example.meta['question']
        assert question in ['cause', 'effect']
        if question == 'cause':
            joiner = ' because'
        else:
            joiner = ', so'
        if self.pattern_id == 0:
            parts_a, parts_b = [
                None, '"', choice1, '" or "', choice2, '"?', None, premise,
                joiner, None, [self.mask], '.'
            ], []
        elif self.pattern_id == 1:
            parts_a, parts_b = [
                None, choice1, ' or', ' ' + choice2, '?', None, premise,
                joiner, None, [self.mask], '.'
            ], []
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                None, '"', choice1, '" or "', choice2, '"', None, premise,
                joiner, [self.mask], '.', None
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        return []

    def encode(self,
               example: InputExample,
               priming: bool = False,
               labeled: bool = False):
        """
        Encode an input example using this pattern-verbalizer pair.

        Args:
            example: the input example to encode
            priming: whether to use this example for priming
            labeled: if ``priming=True``, whether the label should be appended to this example

        Returns:
             A tuple, consisting of a list of input ids and a list of token type ids
        """
        if self.continuous_prompt or self.pattern_id < 2:
            return super().encode(example, priming=priming, labeled=labeled)
        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.tokenizer
        premise = self.remove_final_punc(self.shortenable(example.text_a))
        choice1 = ' ' + self.remove_final_punc(
            self.lowercase_first(example.meta['choice1']))
        choice2 = ' ' + self.remove_final_punc(
            self.lowercase_first(example.meta['choice2']))
        question = example.meta['question']
        assert question in ['cause', 'effect']
        answer = ' because' if question == 'cause' else ' so'
        answer_ids = [
            get_verbalization_ids(answer, tokenizer, force_single_token=True)
        ]
        if self.is_multi_token:
            answer_ids.append(tokenizer.get_command('eop').Id)

        ids_list, positions_list, sep_list, mask_list, target_list = [], [], [], [], []

        for choice in [choice1, choice2]:
            parts = [
                '"', choice1[1:], '" or "', choice2[1:], '"?', premise,
                [self.mask], choice
            ]
            parts = [x if isinstance(x, tuple) else (x, False) for x in parts]
            parts = [(tokenizer.EncodeAsIds(x).tokenization if isinstance(
                x, str) else x, s) for x, s in parts if x]
            self.num_truncated += self.truncate(
                parts, None, answer_ids, max_length=self.max_seq_length)
            tokens_a = [token_id for part, _ in parts for token_id in part]
            data = build_input_from_ids(
                tokens_a,
                None,
                answer_ids,
                self.max_seq_length,
                self.tokenizer,
                args=self.args,
                add_cls=True,
                add_sep=False,
                add_piece=True)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            ids_list.append(ids)
            positions_list.append(position_ids)
            sep_list.append(sep)
            target_list.append(target_ids)
            mask_list.append(loss_masks)
        if example.label is not None:
            label = self.label_list.index(example.label)
        else:
            label = 0
        sample = build_sample(
            ids_list,
            positions=positions_list,
            masks=sep_list,
            label=label,
            logit_mask=mask_list,
            target=target_list,
            unique_id=example.guid)
        return sample


class WscPVP(PVP):

    @staticmethod
    def available_patterns():
        return [0, 1, 2]

    @property
    def is_multi_token(self):
        return True

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_answers(self, example: InputExample):
        target = ' ' + example.meta['span1_text']
        answers = [target]
        if 'candidates' in example.meta:
            candidates = example.meta['candidates']
            # if len(candidates) > 10:
            #     random.shuffle(candidates)
            #     candidates = candidates[:10]
            answers += [' ' + cand for cand in candidates]
        return answers

    def get_parts(self, example: InputExample) -> FilledPattern:
        pronoun = example.meta['span2_text']
        pronoun_idx = example.meta['span2_index']

        words_a = example.text_a.split()
        words_a[pronoun_idx] = '*' + words_a[pronoun_idx] + '*'
        text_a = ' '.join(words_a)
        text_a = self.shortenable(text_a)

        if self.pattern_id == 0:
            parts_a, parts_b = [
                None, text_a,
                None, " The pronoun '*" + pronoun + "*' refers to", None,
                [self.mask], '.'
            ], []
        elif self.pattern_id == 1:
            parts_a, parts_b = [
                None, text_a, None, " In the previous sentence, the pronoun '*"
                + pronoun + "*' refers to", None, [self.mask], '.'
            ], []
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                None, text_a, None,
                " Question: In the passage above, what does the pronoun '*"
                + pronoun + "*' refer to?", None, ' Answer:', [self.mask], '.'
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def encode(self,
               example: InputExample,
               priming: bool = False,
               labeled: bool = False):
        """
        Encode an input example using this pattern-verbalizer pair.
        Args:
            example: the input example to encode
            priming: whether to use this example for priming
            labeled: if ``priming=True``, whether the label should be appended to this example
        Returns:
             A tuple, consisting of a list of input ids and a list of token type ids
        """
        if self.args.loss_func in ['generative', 'mix']:
            sample = super().encode(example, priming=priming, labeled=labeled)
            if self.split == 'train':
                sample['label'] = 0
            return sample

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.tokenizer
        prompt_id = tokenizer.num_tokens
        raw_parts_a, raw_parts_b = self.get_parts(example)

        raw_parts_a = [
            x if isinstance(x, tuple) else (x, False) for x in raw_parts_a
        ]

        def encode_input(raw_parts):
            parts = []
            for x, s in raw_parts:
                if isinstance(x, str):
                    x = tokenizer.EncodeAsIds(x)
                elif isinstance(x, int):
                    x = [prompt_id] * x
                else:
                    pass
                parts.append((x, s))
            return parts

        parts_a = encode_input(raw_parts_a)
        if self.prefix_prompt > 0:
            parts_a = [([prompt_id] * self.prefix_prompt, False)] + parts_a
        parts_b = None
        if raw_parts_b:
            raw_parts_b = [
                x if isinstance(x, tuple) else (x, False) for x in raw_parts_b
            ]
            parts_b = encode_input(raw_parts_b)
        answer = self.get_answers(example)[0]
        answer_ids = get_verbalization_ids(
            answer, tokenizer, force_single_token=False)
        answer_ids = answer_ids + [tokenizer.get_command('eop').Id]
        self.num_truncated += self.truncate(
            parts_a, parts_b, answer_ids, max_length=self.max_seq_length)
        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b
                    for token_id in part] if parts_b else None
        data = build_input_from_ids(
            tokens_a,
            tokens_b,
            answer_ids,
            self.max_seq_length,
            self.tokenizer,
            args=self.args,
            add_cls=True,
            add_sep=False,
            add_piece=True)
        ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
        prompt_pos = [
            idx for idx, token in enumerate(ids) if token == prompt_id
        ]
        ids = [token if token != prompt_id else 0 for token in ids]
        if example.label is not None:
            label = self.label_list.index(example.label)
        else:
            label = 0
        return {
            'text': np.array(ids, dtype=np.int64),
            'target': np.array(target_ids, dtype=np.int64),
            'attention_mask': np.array(sep, dtype=np.int64),
            'loss_mask': np.array(loss_masks, dtype=np.int64),
            'position_id': np.array(position_ids, dtype=np.int64),
            'prompt_pos': np.array(prompt_pos, dtype=np.int64),
            'label': label,
            'uid': example.guid
        }

    def verbalize(self, label) -> List[str]:
        return []


class RecordPVP(PVP):

    @property
    def is_multi_token(self):
        return True

    def get_answers(self, example: InputExample):
        choices = example.meta['candidates']
        choices = [' ' + choice for choice in choices]
        return choices

    def get_parts(self, example: InputExample) -> FilledPattern:
        premise = self.shortenable(example.text_a)

        assert '@placeholder' in example.text_b, f'question "{example.text_b}" does not contain a @placeholder token'
        question_a, question_b = example.text_b.split('@placeholder')
        return [premise, ' ' + question_a.rstrip(), [self.mask],
                question_b], []

    def verbalize(self, label) -> List[str]:
        return []


class RacePVP(PVP):

    @property
    def is_multi_token(self):
        return True

    @staticmethod
    def available_patterns():
        return [0, 1]

    def get_answers(self, example: InputExample):
        choices = example.meta['choices']
        choices = [' ' + choice for choice in choices]
        return choices

    def get_parts(self, example: InputExample) -> FilledPattern:
        context = self.shortenable(example.text_a)
        question = ' ' + example.text_b

        if '_' in question:
            left, right = question.split('_', maxsplit=1)
            if self.pattern_id == 0:
                return [context], [
                    self.shortenable(left.rstrip()), [self.mask],
                    self.shortenable(right)
                ]
            else:
                left = left.rstrip()
                if left:
                    left = self.lowercase_first(left)
                return [context], [
                    ' Based on the previous passage,',
                    self.shortenable(left), [self.mask],
                    self.shortenable(right)
                ]
        else:
            if self.pattern_id == 0:
                return [context], [
                    ' Question:',
                    self.shortenable(question), ' Answer:', [self.mask]
                ]
            else:
                return [context], [
                    ' Based on the previous passage,',
                    self.shortenable(question), [self.mask]
                ]

    def verbalize(self, label) -> List[str]:
        return []


class RtePVP(PVP):
    VERBALIZER = {'not_entailment': [' No'], 'entailment': [' Yes']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = example.text_a
        text_b = example.text_b.rstrip(string.punctuation)
        if self.pattern_id == 0:
            parts_a, parts_b = [None, '"',
                                self.shortenable(text_b), '" ?'], [
                                    None, [self.mask], ',', None, ' "',
                                    self.shortenable(text_a), '"'
                                ]  # noqa
        elif self.pattern_id == 1:
            parts_a, parts_b = [None, self.shortenable(text_b), '?'], [
                None, [self.mask], ',', None,
                self.shortenable(' ' + text_a)
            ]
        elif self.pattern_id == 2:
            parts_a, parts_b = [None, '"',
                                self.shortenable(text_b), '" ?'], [
                                    None, [self.mask], '. "', None,
                                    self.shortenable(text_a), '"'
                                ]  # noqa
        elif self.pattern_id == 3:
            parts_a, parts_b = [None, self.shortenable(text_b), '?'], [
                None, [self.mask], '.', None,
                self.shortenable(' ' + text_a)
            ]
        elif self.pattern_id == 4:
            parts_a, parts_b = [
                None,
                self.shortenable(text_a), None, ' question:',
                self.shortenable(' ' + text_b), ' True or False?', None,
                ' answer:', [self.mask]
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return [' true'] if label == 'entailment' else [' false']
        return RtePVP.VERBALIZER[label]


class CbPVP(RtePVP):
    VERBALIZER = {
        'contradiction': [' No'],
        'entailment': [' Yes'],
        'neutral': [' Maybe']
    }

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4]

    def get_parts(self, example: InputExample) -> FilledPattern:
        if self.pattern_id == 4:
            text_a = self.shortenable(example.text_a)
            text_b = self.shortenable(' ' + example.text_b)
            parts_a, parts_b = [
                None, text_a, None, ' question:', text_b,
                ' true, false or neither?', None, ' answer:', [self.mask]
            ], []
            parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
            return parts_a, parts_b
        return super().get_parts(example)

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return [' true'] if label == 'entailment' else [
                ' false'
            ] if label == 'contradiction' else [' neither']
        return CbPVP.VERBALIZER[label]


class BoolQPVP(PVP):
    VERBALIZER_A = {'false': [' No'], 'true': [' Yes']}

    VERBALIZER_B = {'false': [' false'], 'true': [' true']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4, 5]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = example.text_a
        question = example.text_b

        if self.pattern_id < 2:
            parts_a, parts_b = [
                None,
                self.shortenable(passage), None, ' Question:',
                self.shortenable(' ' + question), '? Answer:', None,
                [self.mask], '.'
            ], []
        elif self.pattern_id < 4:
            parts_a, parts_b = [
                None,
                self.shortenable(passage), ' Based on the previous passage,',
                None,
                self.shortenable(' ' + question), '?', None, [self.mask], '.'
            ], []
        elif self.pattern_id < 6:
            parts_a, parts_b = [
                'Based on the following passage', None,
                self.shortenable(' ' + question), '?', None, [self.mask], '.',
                None,
                self.shortenable(' ' + passage)
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return BoolQPVP.VERBALIZER_A[label]
        else:
            return BoolQPVP.VERBALIZER_B[label]


class MultiRcPVP(PVP):
    VERBALIZER = {0: [' No'], 1: [' Yes']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.remove_final_punc(
            self.shortenable(example.text_a.rstrip()))
        question = self.remove_final_punc(example.text_b.rstrip())
        answer = example.meta['answer']
        if self.pattern_id == 0:
            parts_a, parts_b = [
                passage, '.', None, ' Question:', ' ' + question + '?', None,
                ' Is it', ' ' + answer, '?', None, [self.mask], '.'
            ], []
        elif self.pattern_id == 1:
            parts_a, parts_b = [
                passage, '.', None, ' Question:', ' ' + question, '?',
                None, ' Is the correct answer "', answer, '"?', None,
                [self.mask], '.'
            ], []
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                passage, '. Based on the previous passage,', None,
                ' ' + question, '?', None, ' Is "', answer,
                '" a correct answer?', None, [self.mask], '.'
            ], []
        elif self.pattern_id == 3:
            parts_a, parts_b = [
                None, passage, None, ' ' + question, '- [', [self.mask], ']',
                None, answer
            ], []
        elif self.pattern_id == 4:
            parts_a, parts_b = [
                passage, '.', None, ' Question:', ' ' + question, '?', None,
                ' ' + answer, '?', None, [self.mask], '.'
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 3:
            return [' False'] if label == 0 else [' True']
        return MultiRcPVP.VERBALIZER[label]


class WicPVP(PVP):
    VERBALIZER_A = {'false': [' No'], 'true': [' Yes']}
    VERBALIZER_B = {'false': ['2'], 'true': ['b']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = example.text_a
        text_b = example.text_b
        word = example.meta['word']

        if self.pattern_id == 0:
            parts_a, parts_b = [
                None,
                self.shortenable('"' + text_a + '" / "' + text_b + '"'), None,
                ' Similar sense of "' + word + '"?', None, [self.mask], '.'
            ], []
        elif self.pattern_id == 1:
            parts_a, parts_b = [
                self.shortenable(text_a), None,
                self.shortenable(' ' + text_b), None,
                ' Does ' + word + ' have the same meaning in both sentences?',
                None, [self.mask]
            ], []
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                None, word, ' .', None, ' Sense (1) (a) "',
                self.shortenable(text_a), '"', None, ' (', [self.mask], ') "',
                text_b, '"'
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 2:
            return WicPVP.VERBALIZER_B[label]
        return WicPVP.VERBALIZER_A[label]


class AgnewsPVP(PVP):
    VERBALIZER = {
        '1': [' World'],
        '2': [' Sports'],
        '3': [' Business'],
        '4': [' Tech']
    }

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4, 5]

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [[self.mask], ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [[self.mask], ' News:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', [self.mask], ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', [self.mask], ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', [self.mask], ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [[self.mask], '-', text_a, text_b], []
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsPVP.VERBALIZER[label]


class YahooPVP(PVP):
    VERBALIZER = {
        '1': [' Society'],
        '2': [' Science'],
        '3': [' Health'],
        '4': [' Education'],
        '5': [' Computer'],
        '6': [' Sports'],
        '7': [' Business'],
        '8': [' Entertainment'],
        '9': [' Relationship'],
        '10': [' Politics'],
    }

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4, 5]

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [[self.mask], ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [[self.mask], ' Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', [self.mask], ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', [self.mask], ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', [self.mask], ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [[self.mask], '-', text_a, text_b], []
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]


class MnliPVP(PVP):
    VERBALIZER_A = {
        'contradiction': [' Wrong'],
        'entailment': [' Right'],
        'neutral': [' Maybe']
    }
    VERBALIZER_B = {
        'contradiction': [' No'],
        'entailment': [' Yes'],
        'neutral': [' Maybe']
    }

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [[self.mask], ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [[self.mask], ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]


class YelpPolarityPVP(PVP):
    VERBALIZER = {'1': [' bad'], '2': [' good']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', [self.mask], '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', [self.mask], '.'], []
        elif self.pattern_id == 2:
            return ['Just', [self.mask], '!'], [text]
        elif self.pattern_id == 3:
            return [text], [' In summary, the restaurant is', [self.mask], '.']
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YelpPolarityPVP.VERBALIZER[label]


class YelpFullPVP(YelpPolarityPVP):
    VERBALIZER = {
        '1': [' terrible'],
        '2': [' bad'],
        '3': [' okay'],
        '4': [' good'],
        '5': [' great']
    }

    def verbalize(self, label) -> List[str]:
        return YelpFullPVP.VERBALIZER[label]


class XStancePVP(PVP):
    VERBALIZERS = {
        'en': {
            'FAVOR': ['Yes'],
            'AGAINST': ['No']
        },
        'de': {
            'FAVOR': ['Ja'],
            'AGAINST': ['Nein']
        },
        'fr': {
            'FAVOR': ['Oui'],
            'AGAINST': ['Non']
        }
    }

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4, 5]

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return ['"', text_a, '"'], [[self.mask], '. "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3 or self.pattern_id == 5:
            return [text_a], [[self.mask], '.', text_b]

    def verbalize(self, label) -> List[str]:
        lang = 'de' if self.pattern_id < 2 else 'en' if self.pattern_id < 4 else 'fr'
        return XStancePVP.VERBALIZERS[lang][label]


class Sst2PVP(PVP):
    VERBALIZER_A = {'0': [' terrible'], '1': [' great']}

    VERBALIZER_B = {'0': [' bad'], '1': [' good']}

    @staticmethod
    def available_patterns():
        return [0, 1]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        if self.pattern_id == 0 or self.pattern_id == 1:
            return [text, ' It was', [self.mask], '.'], []
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return Sst2PVP.VERBALIZER_A[label]
        else:
            return Sst2PVP.VERBALIZER_B[label]


class ColaPVP(PVP):
    VERBALIZER = {'0': [' incorrect'], '1': [' correct']}

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        if self.pattern_id == 0:
            return ['"', text, '"', ' This is', [self.mask], '.'], []
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ColaPVP.VERBALIZER[label]


class MrpcPVP(PVP):
    VERBALIZER = {'0': [' No'], '1': [' Yes']}

    @staticmethod
    def available_patterns():
        return [0, 1]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        if self.pattern_id == 0:
            text_b = self.shortenable(self.lowercase_first(example.text_b))
            return [text_a], [[self.mask], ', ', text_b]
        elif self.pattern_id == 1:
            text_b = self.shortenable(
                self.remove_final_punc(self.lowercase_first(example.text_b)))
            return [text_a], [' Does it mean that', text_b, '?', [self.mask]]
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return MrpcPVP.VERBALIZER[label]


class QqpPVP(PVP):
    VERBALIZER = {'0': [' No'], '1': [' Yes']}

    @staticmethod
    def available_patterns():
        return [0, 1]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(self.lowercase_first(example.text_b))
        if self.pattern_id == 0:
            return [text_a], [' Do you mean ', text_b, [self.mask], '.']
        elif self.pattern_id == 1:
            return [text_a], [[self.mask], ', ', text_b]
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return QqpPVP.VERBALIZER[label]


class QnliPVP(PVP):
    VERBALIZER = {'not_entailment': [' No'], 'entailment': [' Yes']}

    @staticmethod
    def available_patterns():
        return [0, 1, 2]

    def get_parts(self, example: InputExample) -> FilledPattern:
        question = self.remove_final_punc(example.text_a)
        passage = example.text_b
        if self.pattern_id == 0:
            return [
                self.shortenable(passage), ' Question:',
                self.shortenable(' ' + question), '? Do you know the answer?',
                [self.mask], '.'
            ], []
        elif self.pattern_id == 1:
            return [
                self.shortenable(passage),
                ' Based on the previous passage, do you know the answer',
                self.shortenable(' ' + question), '?', [self.mask], '.'
            ], []
        elif self.pattern_id == 2:
            return [
                'Based on the following passage, do you know the answer',
                self.shortenable(' ' + question), '?', [self.mask], '.',
                self.shortenable(' ' + passage)
            ], []
        else:
            raise ValueError('No pattern implemented for id {}'.format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return QnliPVP.VERBALIZER[label]


class SquadPVP(PVP):

    @property
    def is_multi_token(self):
        return True

    def get_answers(self, example: InputExample):
        target = ' ' + example.meta['answer']['text']
        answers = [target]
        return answers

    def get_parts(self, example: InputExample) -> FilledPattern:
        context = self.shortenable(example.text_a)
        question = example.text_b
        return [context, ' ' + question, [self.mask], '.'], []

    def verbalize(self, label) -> List[str]:
        return []


def get_verbalization_ids(word: str, tokenizer,
                          force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    ids = tokenizer.EncodeAsIds(word).tokenization
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.DecodeIds(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.command_id_map, \
        f'Verbalization {word} is mapped to a special token {tokenizer.IdToToken(verbalization_id)}'
    return verbalization_id


PVPS = {
    'agnews': AgnewsPVP,
    'mnli': MnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'xstance': XStancePVP,
    'xstance-de': XStancePVP,
    'xstance-fr': XStancePVP,
    'rte': RtePVP,
    'wic': WicPVP,
    'cb': CbPVP,
    'wsc': WscPVP,
    'boolq': BoolQPVP,
    'copa': CopaPVP,
    'multirc': MultiRcPVP,
    'record': RecordPVP,
    'ax-b': RtePVP,
    'ax-g': RtePVP,
    'sst2': Sst2PVP,
    'cola': ColaPVP,
    'mrpc': MrpcPVP,
    'qqp': QqpPVP,
    'qnli': QnliPVP,
    'squad': SquadPVP,
    'race': RacePVP,
}
