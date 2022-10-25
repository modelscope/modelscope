# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import multiprocessing
import os
import random
import re
import time
from collections import defaultdict
from itertools import chain

import json
import numpy as np
from tqdm import tqdm

from modelscope.preprocessors.nlp.space.tokenizer import Tokenizer
from modelscope.utils.constant import ModelFile
from modelscope.utils.nlp.space import ontology
from modelscope.utils.nlp.space.scores import hierarchical_set_score
from modelscope.utils.nlp.space.utils import list2np


class BPETextField(object):

    pad_token = '[PAD]'
    bos_token = '[BOS]'
    eos_token = '[EOS]'
    unk_token = '[UNK]'
    mask_token = '[MASK]'
    sos_u_token = '<sos_u>'
    eos_u_token = '<eos_u>'
    sos_b_token = '<sos_b>'
    eos_b_token = '<eos_b>'
    sos_db_token = '<sos_db>'
    eos_db_token = '<eos_db>'
    sos_a_token = '<sos_a>'
    eos_a_token = '<eos_a>'
    sos_r_token = '<sos_r>'
    eos_r_token = '<eos_r>'

    def __init__(self, model_dir, config):
        self.score_matrixs = {}
        self.prompt_num_for_understand = config.BPETextField.prompt_num_for_understand
        self.prompt_num_for_policy = config.BPETextField.prompt_num_for_policy
        self.understand_tokens = ontology.get_understand_tokens(
            self.prompt_num_for_understand)
        self.policy_tokens = ontology.get_policy_tokens(
            self.prompt_num_for_policy)
        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token, self.unk_token
        ]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(
            vocab_path=os.path.join(model_dir, ModelFile.VOCAB_FILE),
            special_tokens=special_tokens,
            tokenizer_type=config.BPETextField.tokenizer_type)
        self.understand_ids = self.numericalize(self.understand_tokens)
        self.policy_ids = self.numericalize(self.policy_tokens)

        self.tokenizer_type = config.BPETextField.tokenizer_type
        self.filtered = config.BPETextField.filtered
        self.max_len = config.BPETextField.max_len
        self.min_utt_len = config.BPETextField.min_utt_len
        self.max_utt_len = config.BPETextField.max_utt_len
        self.min_ctx_turn = config.BPETextField.min_ctx_turn
        self.max_ctx_turn = config.BPETextField.max_ctx_turn
        self.policy = config.BPETextField.policy
        self.generation = config.BPETextField.generation
        self.with_mlm = config.Dataset.with_mlm
        self.with_query_bow = config.BPETextField.with_query_bow
        self.with_contrastive = config.Dataset.with_contrastive
        self.num_process = config.Dataset.num_process
        self.dynamic_score = config.Dataset.dynamic_score
        self.abandon_label = config.Dataset.abandon_label
        self.trigger_role = config.Dataset.trigger_role
        self.trigger_data = config.Dataset.trigger_data.split(
            ',') if config.Dataset.trigger_data else []

        # data_paths = list(os.path.dirname(c) for c in sorted(
        #     glob.glob(hparams.data_dir + '/**/' + f'train.{hparams.tokenizer_type}.jsonl', recursive=True)))
        # self.data_paths = self.filter_data_path(data_paths=data_paths)
        # self.labeled_data_paths = [data_path for data_path in self.data_paths if 'UniDA' in data_path]
        # self.unlabeled_data_paths = [data_path for data_path in self.data_paths if 'UnDial' in data_path]
        # assert len(self.unlabeled_data_paths) + len(self.labeled_data_paths) == len(self.data_paths)
        # assert len(self.labeled_data_paths) or len(self.unlabeled_data_paths), 'No dataset is loaded'

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.tokenizer.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    @property
    def mask_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]

    @property
    def sos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_u_token])[0]

    @property
    def eos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_u_token])[0]

    @property
    def sos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_b_token])[0]

    @property
    def eos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_b_token])[0]

    @property
    def sos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_db_token])[0]

    @property
    def eos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_db_token])[0]

    @property
    def sos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_a_token])[0]

    @property
    def eos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_a_token])[0]

    @property
    def sos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_r_token])[0]

    @property
    def eos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_r_token])[0]

    @property
    def bot_id(self):
        return 0

    @property
    def user_id(self):
        return 1

    def add_sepcial_tokens(self):
        prompt_tokens = self.understand_tokens + self.policy_tokens
        return ontology.get_special_tokens(other_tokens=prompt_tokens)

    def filter_data_path(self, data_paths):
        if self.trigger_data:
            filtered_data_paths = []
            for data_path in data_paths:
                for data_name in self.trigger_data:
                    if data_path.endswith(f'/{data_name}'):
                        filtered_data_paths.append(data_path)
                        break
        else:
            filtered_data_paths = data_paths
        return filtered_data_paths

    def load_score_matrix(self, data_type, data_iter=None):
        """
        load score matrix for all labeled datasets
        """
        for data_path in self.labeled_data_paths:
            file_index = os.path.join(
                data_path, f'{data_type}.{self.tokenizer_type}.jsonl')
            file = os.path.join(data_path, f'{data_type}.Score.npy')
            if self.dynamic_score:
                score_matrix = {}
                print(f"Created 1 score cache dict for data in '{file_index}'")
            else:
                # TODO add post score matrix
                assert os.path.exists(file), f"{file} isn't exist"
                print(f"Loading 1 score matrix from '{file}' ...")
                fp = np.memmap(file, dtype='float32', mode='r')
                assert len(fp.shape) == 1
                num = int(np.sqrt(fp.shape[0]))
                score_matrix = fp.reshape(num, num)
                print(f"Loaded 1 score matrix for data in '{file_index}'")
            self.score_matrixs[file_index] = score_matrix

    def random_word(self, chars):
        output_label = []
        output_chars = []

        for i, char in enumerate(chars):
            # TODO delete this part to learn special tokens
            if char in [
                    self.sos_u_id, self.eos_u_id, self.sos_r_id, self.eos_r_id
            ]:
                output_chars.append(char)
                output_label.append(self.pad_id)
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_chars.append(self.mask_id)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tmp = random.randint(1, self.vocab_size - 1)
                    output_chars.append(tmp)  # start from 1, to exclude pad_id

                # 10% randomly change token to current token
                else:
                    output_chars.append(char)

                output_label.append(char)

            else:
                output_chars.append(char)
                output_label.append(self.pad_id)

        return output_chars, output_label

    def create_masked_lm_predictions(self, sample):
        src = sample['src']
        src_span_mask = sample['src_span_mask']
        mlm_inputs = []
        mlm_labels = []
        for chars, chars_span_mask in zip(src, src_span_mask):
            if sum(chars_span_mask):
                mlm_input, mlm_label = [], []
                for char, char_mask in zip(chars, chars_span_mask):
                    if char_mask:
                        mlm_input.append(self.mask_id)
                        mlm_label.append(char)
                    else:
                        mlm_input.append(char)
                        mlm_label.append(self.pad_id)
            else:
                mlm_input, mlm_label = self.random_word(chars)
            mlm_inputs.append(mlm_input)
            mlm_labels.append(mlm_label)

        sample['mlm_inputs'] = mlm_inputs
        sample['mlm_labels'] = mlm_labels
        return sample

    def create_span_masked_lm_predictions(self, sample):
        src = sample['src']
        src_span_mask = sample['src_span_mask']
        mlm_inputs = []
        mlm_labels = []
        for chars, chars_span_mask in zip(src, src_span_mask):
            mlm_input, mlm_label = [], []
            for char, char_mask in zip(chars, chars_span_mask):
                if char_mask:
                    mlm_input.append(self.mask_id)
                    mlm_label.append(char)
                else:
                    mlm_input.append(char)
                    mlm_label.append(self.pad_id)
            mlm_inputs.append(mlm_input)
            mlm_labels.append(mlm_label)

        sample['mlm_inputs'] = mlm_inputs
        sample['mlm_labels'] = mlm_labels
        return sample

    def create_token_masked_lm_predictions(self, sample):
        mlm_inputs = sample['mlm_inputs']
        mlm_labels = sample['mlm_labels']

        for i, span_mlm_label in enumerate(mlm_labels):
            if not sum(span_mlm_label):
                mlm_input, mlm_label = self.random_word(mlm_inputs[i])
                mlm_inputs[i] = mlm_input
                mlm_labels[i] = mlm_label

        return sample

    def numericalize(self, tokens):
        """
        here only "convert_tokens_to_ids",
        which need be tokenized into tokens(sub-words) by "tokenizer.tokenize" before
        """
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokenizer.convert_tokens_to_ids(tokens)

    def denumericalize(self, numbers):
        """
        here first "convert_ids_to_tokens", then combine sub-words into origin words
        """
        assert isinstance(numbers, list)
        if len(numbers) == 0:
            return []
        element = numbers[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in numbers]
        else:
            return self.tokenizer.decode(
                numbers,
                ignore_tokens=[self.bos_token, self.eos_token, self.pad_token])

    def save_examples(self, examples, filename):
        start = time.time()
        if filename.endswith('npy'):
            print(f"Saving 1 object to '{filename}' ...")
            assert len(
                examples.shape) == 2 and examples.shape[0] == examples.shape[1]
            num = examples.shape[0]
            fp = np.memmap(
                filename, dtype='float32', mode='w+', shape=(num, num))
            fp[:] = examples[:]
            fp.flush()
            elapsed = time.time() - start
            print(f'Saved 1 object (elapsed {elapsed:.2f}s)')
        elif filename.endswith('jsonl'):
            print(f"Saving examples to '{filename}' ...")
            with open(filename, 'w', encoding='utf-8') as fp:
                for ex in examples:
                    fp.write(json.dumps(ex) + '\n')
            elapsed = time.time() - start
            print(f'Saved {len(examples)} examples (elapsed {elapsed:.2f}s)')
        else:
            print(f"Saving examples to '{filename}' ...")
            raise ValueError(f'Unsport file format: {filename}')

    def load_examples(self, filename):
        start = time.time()
        if filename.endswith('npy'):
            print(f"Loading 1 object from '{filename}' ...")
            fp = np.memmap(filename, dtype='float32', mode='r')
            assert len(fp.shape) == 1
            num = int(np.sqrt(fp.shape[0]))
            examples = fp.reshape(num, num)
            elapsed = time.time() - start
            print(f'Loaded 1 object (elapsed {elapsed:.2f}s)')
        else:
            print(f"Loading examples from '{filename}' ...")
            with open(filename, 'r', encoding='utf-8') as fp:
                examples = list(map(lambda s: json.loads(s.strip()), fp))
            elapsed = time.time() - start
            print(f'Loaded {len(examples)} examples (elapsed {elapsed:.2f}s)')
        return examples

    def utt_filter_pred(self, utt):
        return self.min_utt_len <= len(utt) \
            and (not self.filtered or len(utt) <= self.max_utt_len)

    def utts_filter_pred(self, utts):
        return self.min_ctx_turn <= len(utts) \
            and (not self.filtered or len(utts) <= self.max_ctx_turn)

    def get_token_pos(self, tok_list, value_label):
        find_pos = []
        found = False
        label_list = [
            item
            for item in map(str.strip, re.split('(\\W+)', value_label.lower()))
            if len(item) > 0
        ]
        len_label = len(label_list)
        for i in range(len(tok_list) + 1 - len_label):
            if tok_list[i:i + len_label] == label_list:
                find_pos.append((i, i + len_label))  # start, exclusive_end
                found = True
        return found, find_pos

    def build_score_matrix(self, examples):
        """
        build symmetric score matrix
        """
        assert self.num_process == 1
        print('Building score matrix from examples ...')
        num = len(examples)
        score_matrix = np.eye(
            num, num, dtype='float32'
        )  # in case of empty label of self, resulting in score 0.

        for i in tqdm(range(num)):
            for j in range(i):
                # TODO change the score method
                score = hierarchical_set_score(
                    frame1=examples[i]['label'], frame2=examples[j]['label'])
                score_matrix[i][j] = score
                score_matrix[j][i] = score

        print('Built score matrix')
        return score_matrix

    def build_score_matrix_on_the_fly(self,
                                      ids,
                                      labels,
                                      data_file,
                                      is_post=False):
        """
        build symmetric score matrix on the fly
        @is_post: True for resp label of sample i and j, False for query label of sample i and j
        """
        num = len(labels)
        tag = 'r' if is_post else 'q'
        assert len(ids) == len(labels)
        score_matrix = np.eye(
            num, num, dtype='float32'
        )  # in case of empty label of self, resulting in score 0.

        for i in range(num):
            for j in range(i):
                score = self.score_matrixs[data_file].get(
                    f'{ids[i]}-{ids[j]}-{tag}', None)
                if score is None:
                    score = self.score_matrixs[data_file].get(
                        f'{ids[j]}-{ids[i]}-{tag}', None)
                if score is None:
                    # TODO change the score method
                    score = hierarchical_set_score(
                        frame1=labels[i], frame2=labels[j])
                    self.score_matrixs[data_file][
                        f'{ids[i]}-{ids[j]}-{tag}'] = score
                score_matrix[i][j] = score
                score_matrix[j][i] = score

        return score_matrix

    def build_score_matrix_func(self, examples, start, exclusive_end):
        """
        build sub score matrix
        """
        num = len(examples)
        process_id = os.getpid()
        description = f'PID: {process_id} Start: {start} End: {exclusive_end}'
        print(
            f'PID-{process_id}: Building {start} to {exclusive_end} lines score matrix from examples ...'
        )
        score_matrix = np.zeros((exclusive_end - start, num), dtype='float32')

        for abs_i, i in enumerate(
                tqdm(range(start, exclusive_end), desc=description)):
            for j in range(num):
                # TODO change the score method
                score = hierarchical_set_score(
                    frame1=examples[i]['label'], frame2=examples[j]['label'])
                score_matrix[abs_i][j] = score

        print(
            f'PID-{process_id}: Built {start} to {exclusive_end} lines score matrix'
        )
        return {'start': start, 'score_matrix': score_matrix}

    def build_score_matrix_multiprocessing(self, examples):
        """
        build score matrix
        """
        assert self.num_process >= 2 and multiprocessing.cpu_count() >= 2
        print('Building score matrix from examples ...')
        results = []
        num = len(examples)
        sub_num, res_num = num // self.num_process, num % self.num_process
        patches = [sub_num] * (self.num_process - 1) + [sub_num + res_num]

        start = 0
        pool = multiprocessing.Pool(processes=self.num_process)
        for patch in patches:
            exclusive_end = start + patch
            results.append(
                pool.apply_async(self.build_score_matrix_func,
                                 (examples, start, exclusive_end)))
            start = exclusive_end
        pool.close()
        pool.join()

        sub_score_matrixs = [result.get() for result in results]
        sub_score_matrixs = sorted(
            sub_score_matrixs, key=lambda sub: sub['start'])
        sub_score_matrixs = [
            sub_score_matrix['score_matrix']
            for sub_score_matrix in sub_score_matrixs
        ]
        score_matrix = np.concatenate(sub_score_matrixs, axis=0)
        assert score_matrix.shape == (num, num)
        np.fill_diagonal(
            score_matrix,
            1.)  # in case of empty label of self, resulting in score 0.

        print('Built score matrix')
        return score_matrix

    def extract_span_texts(self, text, label):
        span_texts = []
        for domain, frame in label.items():
            for act, slot_values in frame.items():
                for slot, values in slot_values.items():
                    for value in values:
                        if value['span']:
                            span_texts.append(
                                text[value['span'][0]:value['span'][1]])
                        elif str(value['value']).strip().lower() in text.strip(
                        ).lower():
                            span_texts.append(str(value['value']))
        return span_texts

    def fix_label(self, label):
        for domain, frame in label.items():
            if not frame:
                return {}
            for act, slot_values in frame.items():
                if act == 'DEFAULT_INTENT' and not slot_values:
                    return {}
        return label

    def build_examples_multi_turn(self, data_file, data_type='train'):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            for dialog_id in tqdm(input_data):
                turns = input_data[dialog_id]['turns']
                history, history_role, history_span_mask, history_label = [], [], [], []
                for t, turn in enumerate(turns):
                    label = turn['label']
                    role = turn['role']
                    text = turn['text']
                    utterance, span_mask = [], []

                    token_list = [
                        tok for tok in map(str.strip,
                                           re.split('(\\W+)', text.lower()))
                        if len(tok) > 0
                    ]
                    span_list = np.zeros(len(token_list), dtype=np.int32)
                    span_texts = self.extract_span_texts(
                        text=text, label=label)

                    for span_text in span_texts:
                        found, find_pos = self.get_token_pos(
                            tok_list=token_list, value_label=span_text)
                        if found:
                            for start, exclusive_end in find_pos:
                                span_list[start:exclusive_end] = 1

                    token_list = [
                        self.tokenizer.tokenize(token) for token in token_list
                    ]
                    span_list = [[tag] * len(token_list[i])
                                 for i, tag in enumerate(span_list)]
                    for sub_tokens in token_list:
                        utterance.extend(sub_tokens)
                    for sub_spans in span_list:
                        span_mask.extend(sub_spans)
                    assert len(utterance) == len(span_mask)

                    history.append(utterance)
                    history_role.append(role)
                    history_span_mask.append(span_mask)
                    history_label.append(self.fix_label(label))

                    tmp = self.utts_filter_pred(history[:-1]) and all(
                        map(self.utt_filter_pred, history))
                    if (
                            tmp or data_type == 'test'
                    ) and role in self.trigger_role and t:  # TODO consider test
                        src = [
                            s[-self.max_utt_len:]
                            for s in history[:-1][-self.max_ctx_turn:]
                        ]
                        src_span_mask = [
                            s[-self.max_utt_len:] for s in
                            history_span_mask[:-1][-self.max_ctx_turn:]
                        ]
                        roles = [
                            role
                            for role in history_role[:-1][-self.max_ctx_turn:]
                        ]

                        new_src = []
                        for i, s in enumerate(src):
                            if roles[i] == 'user':
                                user_or_sys = [self.eos_u_id]
                            else:
                                user_or_sys = [self.sos_r_id]
                            tmp = [self.sos_u_id
                                   ] + self.numericalize(s) + user_or_sys
                            tmp = tmp + self.numericalize(s) + [self.eos_r_id]
                            new_src.append(tmp)

                        src_span_mask = [[0] + list(map(int, s)) + [0]
                                         for s in src_span_mask]

                        tgt = [self.sos_r_id] + self.numericalize(
                            history[-1]) + [self.eos_r_id]
                        if data_type != 'test':
                            tgt = tgt[:self.max_utt_len + 2]

                        ex = {
                            'dialog_id': dialog_id,
                            'turn_id': turn['turn_id'],
                            'src': new_src,
                            'src_span_mask': src_span_mask,
                            'tgt': tgt,
                            'query_label': history_label[-2],
                            'resp_label': history_label[-1],
                            'extra_info': turn.get('extra_info', '')
                        }
                        examples.append(ex)
                    else:
                        ignored += 1

        # add span mlm inputs and span mlm labels in advance
        if self.with_mlm:
            examples = [
                self.create_span_masked_lm_predictions(example)
                for example in examples
            ]

        # add absolute id of the dataset for indexing scores in its score matrix
        for i, example in enumerate(examples):
            example['id'] = i

        print(
            f'Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)'
        )
        return examples

    def preprocessor(self, text_list):
        role = 'user'
        examples = []

        for text in text_list:
            history, history_role, history_span_mask = [], [], []
            utterance, span_mask = [], []
            token_list = [
                tok for tok in map(str.strip, re.split('(\\W+)', text.lower()))
                if len(tok) > 0
            ]
            span_list = np.zeros(len(token_list), dtype=np.int32)
            token_list = [
                self.tokenizer.tokenize(token) for token in token_list
            ]
            span_list = [[tag] * len(token_list[i])
                         for i, tag in enumerate(span_list)]

            for sub_tokens in token_list:
                utterance.extend(sub_tokens)
            for sub_spans in span_list:
                span_mask.extend(sub_spans)
            assert len(utterance) == len(span_mask)

            history.append(utterance)
            history_role.append(role)
            history_span_mask.append(span_mask)

            src = [s[-self.max_utt_len:] for s in history[-self.max_ctx_turn:]]
            src_span_mask = [
                s[-self.max_utt_len:]
                for s in history_span_mask[-self.max_ctx_turn:]
            ]
            roles = [role for role in history_role[-self.max_ctx_turn:]]

            new_src = []
            for i, s in enumerate(src):
                if roles[i] == 'user':
                    user_or_sys = [self.eos_u_id]
                else:
                    user_or_sys = [self.sos_r_id]
                tmp = [self.sos_u_id] + self.numericalize(s) + user_or_sys
                tmp = tmp + self.numericalize(s) + [self.eos_r_id]
                new_src.append(tmp)

            src_span_mask = [[0] + list(map(int, s)) + [0]
                             for s in src_span_mask]

            ex = {
                'dialog_id': 'inference',
                'turn_id': 0,
                'role': role,
                'src': new_src,
                'src_span_mask': src_span_mask,
                'query_label': {
                    'DEFAULT_DOMAIN': {
                        'card_arrival': {}
                    }
                },
                'extra_info': {
                    'intent_label': -1
                }
            }
            examples.append(ex)
        # add span mlm inputs and span mlm labels in advance
        if self.with_mlm:
            examples = [
                self.create_span_masked_lm_predictions(example)
                for example in examples
            ]

        # add absolute id of the dataset for indexing scores in its score matrix
        for i, example in enumerate(examples):
            example['id'] = i

        return examples

    def build_examples_single_turn(self, data_file, data_type='train'):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            for dialog_id in tqdm(input_data):
                turns = input_data[dialog_id]['turns']
                history, history_role, history_span_mask = [], [], []
                for turn in turns:
                    label = turn['label']
                    role = turn['role']
                    text = turn['text']
                    utterance, span_mask = [], []

                    token_list = [
                        tok for tok in map(str.strip,
                                           re.split('(\\W+)', text.lower()))
                        if len(tok) > 0
                    ]
                    span_list = np.zeros(len(token_list), dtype=np.int32)
                    span_texts = self.extract_span_texts(
                        text=text, label=label)

                    for span_text in span_texts:
                        found, find_pos = self.get_token_pos(
                            tok_list=token_list, value_label=span_text)
                        if found:
                            for start, exclusive_end in find_pos:
                                span_list[start:exclusive_end] = 1

                    token_list = [
                        self.tokenizer.tokenize(token) for token in token_list
                    ]
                    span_list = [[tag] * len(token_list[i])
                                 for i, tag in enumerate(span_list)]
                    for sub_tokens in token_list:
                        utterance.extend(sub_tokens)
                    for sub_spans in span_list:
                        span_mask.extend(sub_spans)
                    assert len(utterance) == len(span_mask)

                    history.append(utterance)
                    history_role.append(role)
                    history_span_mask.append(span_mask)

                    tmp = self.utts_filter_pred(history) and all(
                        map(self.utt_filter_pred, history))
                    tmp = tmp or data_type == 'test'
                    if tmp and role in self.trigger_role:  # TODO consider test
                        src = [
                            s[-self.max_utt_len:]
                            for s in history[-self.max_ctx_turn:]
                        ]
                        src_span_mask = [
                            s[-self.max_utt_len:]
                            for s in history_span_mask[-self.max_ctx_turn:]
                        ]
                        roles = [
                            role for role in history_role[-self.max_ctx_turn:]
                        ]
                        new_src = []
                        for i, s in enumerate(src):
                            if roles[i] == 'user':
                                user_or_sys = [self.eos_u_id]
                            else:
                                user_or_sys = [self.sos_r_id]
                            tmp = [self.sos_u_id
                                   ] + self.numericalize(s) + user_or_sys
                            new_src.append(tmp)

                        src_span_mask = [[0] + list(map(int, s)) + [0]
                                         for s in src_span_mask]

                        ex = {
                            'dialog_id': dialog_id,
                            'turn_id': turn['turn_id'],
                            'role': role,
                            'src': new_src,
                            'src_span_mask': src_span_mask,
                            'query_label': self.fix_label(label),
                            'extra_info': turn.get('extra_info', '')
                        }
                        examples.append(ex)
                    else:
                        ignored += 1

        # add span mlm inputs and span mlm labels in advance
        if self.with_mlm:
            examples = [
                self.create_span_masked_lm_predictions(example)
                for example in examples
            ]

        # add absolute id of the dataset for indexing scores in its score matrix
        for i, example in enumerate(examples):
            example['id'] = i

        print(
            f'Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)'
        )
        return examples

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)
        batch = {}

        src = [sp['src'] for sp in samples]
        query_token, src_token, src_pos, src_turn, src_role = [], [], [], [], []
        for utts in src:
            query_token.append(utts[-1])
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(utt_len)) for utt_len in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            role = [
                [self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id] * l
                for i, l in enumerate(utt_lens)
            ]
            src_role.append(list(chain(*role))[-self.max_len:])

        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)
        batch['src_token'] = src_token
        batch['src_pos'] = src_pos
        batch['src_type'] = src_role
        batch['src_turn'] = src_turn
        batch['src_mask'] = (src_token != self.pad_id).astype('int64')

        if self.with_query_bow:
            query_token = list2np(query_token, padding=self.pad_id)
            batch['query_token'] = query_token
            batch['query_mask'] = (query_token != self.pad_id).astype('int64')

        if self.with_mlm:
            mlm_token, mlm_label = [], []
            raw_mlm_input = [sp['mlm_inputs'] for sp in samples]
            raw_mlm_label = [sp['mlm_labels'] for sp in samples]
            for inputs in raw_mlm_input:
                mlm_token.append(list(chain(*inputs))[-self.max_len:])
            for labels in raw_mlm_label:
                mlm_label.append(list(chain(*labels))[-self.max_len:])

            mlm_token = list2np(mlm_token, padding=self.pad_id)
            mlm_label = list2np(mlm_label, padding=self.pad_id)
            batch['mlm_token'] = mlm_token
            batch['mlm_label'] = mlm_label
            batch['mlm_mask'] = (mlm_label != self.pad_id).astype('int64')

        if self.dynamic_score and self.with_contrastive and not self.abandon_label:
            query_labels = [sp['query_label'] for sp in samples]
            batch['query_labels'] = query_labels
            if self.trigger_role == 'system':
                resp_labels = [sp['resp_label'] for sp in samples]
                batch['resp_labels'] = resp_labels
            batch['label_ids'] = np.arange(
                batch_size)  # to identify labels for each GPU when multi-gpu

        if self.understand_ids:
            understand = [self.understand_ids for _ in samples]
            understand_token = np.array(understand).astype('int64')
            batch['understand_token'] = understand_token
            batch['understand_mask'] = \
                (understand_token != self.pad_id).astype('int64')

        if self.policy_ids and self.policy:
            policy = [self.policy_ids for _ in samples]
            policy_token = np.array(policy).astype('int64')
            batch['policy_token'] = policy_token
            batch['policy_mask'] = \
                (policy_token != self.pad_id).astype('int64')

        if 'tgt' in samples[0]:
            tgt = [sp['tgt'] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch['tgt_token'] = tgt_token
            batch['tgt_pos'] = tgt_pos
            batch['tgt_type'] = tgt_role
            batch['tgt_turn'] = tgt_turn
            batch['tgt_mask'] = (tgt_token != self.pad_id).astype('int64')

        if 'id' in samples[0]:
            ids = [sp['id'] for sp in samples]
            ids = np.array(ids).astype('int64')
            batch['ids'] = ids

        return batch, batch_size


class IntentBPETextField(BPETextField):

    def __init__(self, model_dir, config):
        super(IntentBPETextField, self).__init__(model_dir, config)

    def retrieve_examples(self,
                          dataset,
                          labels,
                          inds,
                          task,
                          num=None,
                          cache=None):
        assert task == 'intent', 'Example-driven may only be used with intent prediction'
        if num is None and labels is not None:
            num = len(labels) * 2

        # Populate cache
        if cache is None:
            cache = defaultdict(list)
            for i, example in enumerate(dataset):
                assert i == example['id']
                cache[example['extra_info']['intent_label']].append(i)

        # One example for each label
        example_inds = []
        for lable in set(labels.tolist()):
            if lable == -1:
                continue

            ind = random.choice(cache[l])
            retries = 0
            while ind in inds.tolist() or type(ind) is not int:
                ind = random.choice(cache[l])
                retries += 1
                if retries > len(dataset):
                    break

            example_inds.append(ind)

        # Sample randomly until we hit batch size
        while len(example_inds) < min(len(dataset), num):
            ind = random.randint(0, len(dataset) - 1)
            if ind not in example_inds and ind not in inds.tolist():
                example_inds.append(ind)

        # Create examples
        example_batch = {}
        examples = [dataset[i] for i in example_inds]
        examples, _ = self.collate_fn_multi_turn(examples)
        example_batch['example_src_token'] = examples['src_token']
        example_batch['example_src_pos'] = examples['src_pos']
        example_batch['example_src_type'] = examples['src_type']
        example_batch['example_src_turn'] = examples['src_turn']
        example_batch['example_src_mask'] = examples['src_mask']
        example_batch['example_tgt_token'] = examples['tgt_token']
        example_batch['example_tgt_mask'] = examples['tgt_mask']
        example_batch['example_intent'] = examples['intent_label']

        return example_batch

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)
        batch = {}

        cur_roles = [sp['role'] for sp in samples]
        src = [sp['src'] for sp in samples]
        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts, cur_role in zip(src, cur_roles):
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(utt_len)) for utt_len in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            if cur_role == 'user':
                role = [[
                    self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id
                ] * l for i, l in enumerate(utt_lens)]
            else:
                role = [[
                    self.user_id if (len(utts) - i) % 2 == 0 else self.bot_id
                ] * l for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])

        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)
        batch['src_token'] = src_token
        batch['src_pos'] = src_pos
        batch['src_type'] = src_role
        batch['src_turn'] = src_turn
        batch['src_mask'] = (src_token != self.pad_id).astype(
            'int64')  # input mask

        if self.with_mlm:
            mlm_token, mlm_label = [], []
            raw_mlm_input = [sp['mlm_inputs'] for sp in samples]
            raw_mlm_label = [sp['mlm_labels'] for sp in samples]
            for inputs in raw_mlm_input:
                mlm_token.append(list(chain(*inputs))[-self.max_len:])
            for labels in raw_mlm_label:
                mlm_label.append(list(chain(*labels))[-self.max_len:])

            mlm_token = list2np(mlm_token, padding=self.pad_id)
            mlm_label = list2np(mlm_label, padding=self.pad_id)
            batch['mlm_token'] = mlm_token
            batch['mlm_label'] = mlm_label
            batch['mlm_mask'] = (mlm_label != self.pad_id).astype(
                'int64')  # label mask

        if self.understand_ids:
            tgt = [self.understand_ids for _ in samples]
            tgt_token = np.array(tgt).astype('int64')
            batch['tgt_token'] = tgt_token
            batch['tgt_mask'] = (tgt_token != self.pad_id).astype(
                'int64')  # input mask

        if 'id' in samples[0]:
            ids = [sp['id'] for sp in samples]
            ids = np.array(ids).astype('int64')
            batch['ids'] = ids

        if self.dynamic_score and self.with_contrastive:
            query_labels = [sp['query_label'] for sp in samples]
            batch['query_labels'] = query_labels
            batch['label_ids'] = np.arange(batch_size)

        if 'intent_label' in samples[0]['extra_info']:
            intent_label = [
                sample['extra_info']['intent_label'] for sample in samples
            ]
            intent_label = np.array(intent_label).astype('int64')
            batch['intent_label'] = intent_label

        return batch, batch_size
