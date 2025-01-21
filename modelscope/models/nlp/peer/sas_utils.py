import random

import nltk
import numpy as np
import torch


def get_random_states(device=None):
    random_states = {}

    random_states['rng_state_torch'] = torch.get_rng_state()
    random_states['rng_state_np'] = np.random.get_state()
    random_states['rng_state_rnd'] = random.getstate()
    if device is not None and device.type == 'cuda':
        random_states['rng_state_torch_cuda'] = torch.cuda.get_rng_state(
            device)

    return random_states


def set_random_states(random_states, device=None):

    torch.set_rng_state(random_states['rng_state_torch'])
    np.random.set_state(random_states['rng_state_np'])
    random.setstate(random_states['rng_state_rnd'])
    if device is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(random_states['rng_state_torch_cuda'])


# Check any nan or inf in the data. Return an array of two elements for nan and inf, respectively.
# Inputs
#   data: a tensor or a tuple of multiple tensors
# Outputs:
#   results: Each element shows the # of tensors that includes nan or inf.
#            If data is a "tuple" (instead of a single tensor),
#            we add 10 to the count if any nan or inf is detected.
def check_nan_inf(data):
    if data is None:
        return None

    result = [0, 0]
    if torch.is_tensor(data):
        if torch.isnan(data).any():
            result[0] = 1
        if torch.isinf(data).any():
            result[1] = 1

    elif type(data) is tuple:
        for i in range(len(data)):
            if torch.is_tensor(data[i]):
                if torch.isnan(data[i]).any():
                    result[0] += 1
                if torch.isinf(data[i]).any():
                    result[1] += 1

        if result[0] > 0:
            result[0] += 10
        if result[1] > 0:
            result[1] += 10

    return result if sum(result) > 0 else None


class SequenceSideInfo():

    def __init__(self, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import ElectraTokenizer
            self.tokenizer = ElectraTokenizer.from_pretrained(
                'google/electra-small-generator')

        self.sen_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

        tokens = [
            self.tokenizer.decode([i])
            for i in range(self.tokenizer.vocab_size)
        ]
        self.ind_subtokens = set(
            [i for i in range(len(tokens)) if tokens[i][0:2] == '##'])
        tmp = [
            0 if t[0] == '[' and t[-1] == ']' else
            (10 + min(5,
                      len(t) - 2) if t[0:2] == '##' else min(10, len(t)))
            for t in tokens
        ]
        self.len_tokens = torch.tensor(tmp, dtype=torch.int8)

    def getSenTokIdx(self, sentence_position_embedding, inputs_str,
                     seq_len_total):
        sentences = self.sen_tokenizer.tokenize(inputs_str)
        sen_lengths = np.array([
            len(x) - 2
            for x in self.tokenizer.batch_encode_plus(sentences)['input_ids']
        ])  # -2: to drop the extra [CLS] and [SEP] added by sen_tokenizer

        sen_lengths[0] = seq_len_total - sen_lengths[1:].sum()

        idx_sen = np.concatenate([
            i * np.ones(sen_lengths[i], dtype=np.int8)
            for i in range(len(sen_lengths))
        ])
        idx_tok = np.concatenate([
            np.arange(sen_lengths[i], dtype=np.int8)
            for i in range(len(sen_lengths))
        ])

        return np.concatenate((idx_sen, idx_tok))

    def generate_seq_side_info(self, sentence_position_embedding, inputs_id):
        is_np_array = False
        if isinstance(inputs_id[0], (list, np.ndarray)):
            is_np_array = True
            inputs_id = torch.tensor(inputs_id)

        if hasattr(self.tokenizer, 'batch_decode'):
            inputs_str = self.tokenizer.batch_decode(inputs_id)
            sen_tok_idx = torch.tensor(
                np.array([
                    self.getSenTokIdx(sentence_position_embedding, input_str,
                                      inputs_id.shape[1])
                    for input_str in inputs_str
                ]),
                device=inputs_id.device)
        else:
            sen_tok_idx = torch.tensor(
                np.array([
                    self.getSenTokIdx(sentence_position_embedding,
                                      self.tokenizer.decode(input_ori),
                                      inputs_id.shape[1])
                    for input_ori in inputs_id.numpy()
                ]),
                device=inputs_id.device)

        side_info_dict = dict()
        seq_length = inputs_id.shape[1]
        side_info_dict[
            'ss_sentence_position_in_sequence'] = sen_tok_idx[:, 0:seq_length]
        side_info_dict[
            'ss_token_position_in_sentence'] = sen_tok_idx[:, 1 * seq_length:2
                                                           * seq_length]

        if sentence_position_embedding >= 2:
            # consider sub-word tokens
            unique, _ = np.unique(inputs_id, return_inverse=True)
            ind_subtokens = self.ind_subtokens.intersection(set(unique))

            if len(ind_subtokens) > 0:
                idx_tok_ww = torch.stack([
                    inputs_id == st for st in ind_subtokens
                ]).any(axis=0).char()
            else:
                idx_tok_ww = torch.zeros(inputs_id.shape, dtype=torch.int8)

            idx_tok_ww[:, 0] = 0
            idx_tok_ww_1 = idx_tok_ww[:, 1:]
            for i in range(1, 11):
                pos = torch.logical_and(idx_tok_ww_1 == i,
                                        idx_tok_ww[:, 0:-1] == i)
                if len(pos) == 0:
                    break
                idx_tok_ww_1[pos] = i + 1
            side_info_dict['ss_token_position_in_whole_word'] = idx_tok_ww

            inputs_str_len = self.len_tokens[inputs_id.long()]
            side_info_dict['ss_token_string_length'] = inputs_str_len

        if is_np_array:
            for key in side_info_dict.keys():
                side_info_dict[key] = side_info_dict[key].numpy()

        return side_info_dict
