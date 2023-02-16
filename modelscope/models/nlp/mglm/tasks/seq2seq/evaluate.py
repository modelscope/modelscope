# Copyright (c) 2022 Zhipu.AI

import datetime
import random
import string

import torch
import torch.nn.functional as F
from generation_utils import (BeamSearchScorer, LogitsProcessorList,
                              MinLengthLogitsProcessor,
                              NoRepeatNGramLogitsProcessor)
from megatron_util import mpu, print_rank_0
from rouge_score import rouge_scorer


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ','):
            return False
    return True


gigaword_tok_dict = {
    '(': '-lrb-',
    ')': '-rrb-',
    '[': '-lsb-',
    ']': '-rsb-',
    '{': '-lcb-',
    '}': '-rcb-',
    '[UNK]': 'UNK',
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;'
}

cnndm_tok_dict = {
    '(': '-LRB-',
    ')': '-RRB-',
    '[': '-LSB-',
    ']': '-RSB-',
    '{': '-LCB-',
    '}': '-RCB-'
}


def fix_tokenization(text, dataset):
    if dataset == 'cnn_dm_org':
        return text
    if dataset == 'gigaword':
        text = text.replace('[UNK]', 'UNK')
        return text
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append('``')
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(
                output_tokens) > 0 and output_tokens[-1].endswith(
                    'n') and i < len(input_tokens) - 1 and input_tokens[
                        i + 1] == 't':  # noqa
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[
                i + 1] in ('s', 'd', 'll'):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append('`')
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == '.' and i < len(input_tokens) - 2 and input_tokens[
                i + 1] == '.' and input_tokens[i + 2] == '.':
            output_tokens.append('...')
            i += 3
        elif tok == ',' and len(output_tokens) > 0 and _is_digit(
                output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(
                    input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ',' + input_tokens[i + 1]
            i += 2
        elif tok == '.' and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and \
                input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.' + input_tokens[i + 1]
            i += 2
        elif tok == '.' and len(output_tokens) > 0 and len(
                output_tokens[-1]) == 1 and output_tokens[-1].isalpha(  # noqa
                ) and i < len(input_tokens) - 2 and len(  # noqa
                    input_tokens[i + 1]) == 1 and input_tokens[
                        i + 1].isalpha(  # noqa
                        ) and input_tokens[i + 2] == '.':  # noqa
            # U . N . -> U.N.
            k = i + 3
            while k + 2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[
                        k + 1].isalpha() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i = k
        elif tok == '-':
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == '-':
                output_tokens.append('--')
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append('-')
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[
                    i + 1][0] not in string.punctuation:
                output_tokens[-1] += '-'
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append('-')
                i += 1
        elif prev_dash and len(
                output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return ' '.join(output_tokens)


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]  # noqa
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set) / len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def rouge_metric(predictions,
                 labels,
                 examples,
                 metric='rouge-1',
                 duplicate_rate=0.7,
                 dataset='cnn_dm'):
    metric_dict = {
        'rouge-1': 'rouge1',
        'rouge-2': 'rouge2',
        'rouge-l': 'rougeLsum'
    }
    refs = [example.meta['ref'] for example in examples]
    ref_list = []
    for ref in refs:
        ref = ref.strip().split('[SEP]')
        ref = [fix_tokenization(sentence, dataset=dataset) for sentence in ref]
        ref = '\n'.join(ref)
        ref_list.append(ref)
    pred_list = []
    for prediction in predictions:
        buf = []
        for sentence in prediction.strip().split('[SEP]'):
            sentence = fix_tokenization(sentence, dataset=dataset)
            if any(get_f1(sentence, s) > 1.0 for s in buf):
                continue
            s_len = len(sentence.split())
            if s_len <= 4:
                continue
            buf.append(sentence)
        if duplicate_rate and duplicate_rate < 1:
            buf = remove_duplicate(buf, duplicate_rate)
        line = '\n'.join(buf)
        pred_list.append(line)
    if torch.distributed.get_rank() == 0:
        import json
        with open('./results.json', 'w') as output:
            for ref, pred in zip(ref_list, pred_list):
                output.write(json.dumps({'ref': ref, 'pred': pred}) + '\n')
    scorer = rouge_scorer.RougeScorer([metric_dict[metric]], use_stemmer=True)
    scores = [
        scorer.score(pred, ref) for pred, ref in zip(pred_list, ref_list)
    ]
    scores = [score[metric_dict[metric]].fmeasure for score in scores]
    scores = sum(scores) / len(scores)
    return scores


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    tokens = batch['text'].long().cuda()
    attention_mask = batch['attention_mask'].long().cuda()
    position_ids = batch['position_id'].long().cuda()
    return tokens, attention_mask, position_ids


class DecoderEvaluater:

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.start_token = tokenizer.get_command('sop').Id
        self.end_token = tokenizer.get_command('eop').Id
        self.mask_token = tokenizer.get_command(
            'sMASK').Id if args.task_mask else tokenizer.get_command('MASK').Id
        self.pad_token = tokenizer.get_command('pad').Id
        self.processors = LogitsProcessorList()
        if args.min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(args.min_tgt_length,
                                                 self.end_token)
            self.processors.append(processor)
        if args.no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size)
            self.processors.append(processor)

    def evaluate(self, model, dataloader, example_dict, args):
        """Calculate correct over total answers and return prediction if the
        `output_predictions` is true."""
        model.eval()
        store = torch.distributed.TCPStore(args.master_ip,
                                           18931 + random.randint(0, 10000),
                                           mpu.get_data_parallel_world_size(),
                                           torch.distributed.get_rank() == 0,
                                           datetime.timedelta(seconds=30))
        print_rank_0('Distributed store created')
        with torch.no_grad():
            # For all the batches in the dataset.
            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(
                    data, args)
                batch_size = tokens.size(0)
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=args.out_seq_length,
                    num_beams=args.num_beams,
                    device=tokens.device,
                    length_penalty=args.length_penalty,
                    do_early_stopping=False,
                )
                beam_scores = torch.zeros((batch_size, args.num_beams),
                                          dtype=torch.float,
                                          device=tokens.device)
                beam_scores[:, 1:] = -1e9
                beam_scores = beam_scores.view((batch_size * args.num_beams, ))
                # Run the model forward.
                counter = 0
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        next_token_logits, *mems = model(
                            tokens,
                            position_ids,
                            attention_mask,
                            return_memory=True)
                        seq_length = next_token_logits.size(1)
                        next_token_logits = next_token_logits[:, -1]
                        next_token_logits = next_token_logits.unsqueeze(
                            1).repeat(1, args.num_beams,
                                      1).view(batch_size * args.num_beams, -1)
                        mems = [
                            mem.unsqueeze(1).repeat(
                                1, args.num_beams, 1,
                                1).view(batch_size * args.num_beams,
                                        seq_length, -1) for mem in mems
                        ]
                        position_ids = tokens.new_ones(batch_size,
                                                       args.num_beams, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = text.index(self.mask_token)
                            position_ids[i, :, 0] = mask_pos
                        position_ids = position_ids.reshape(
                            batch_size * args.num_beams, 2, 1)
                        tokens = tokens.new_zeros(batch_size * args.num_beams,
                                                  0)
                        attention_mask = tokens.new_zeros(
                            [batch_size * args.num_beams])
                    else:
                        if not args.no_block_position:
                            position_ids[:, 1] = counter + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(
                            last_token,
                            position_ids,
                            attention_mask,
                            *mems,
                            return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(
                        next_token_logits, dim=-1)
                    next_token_scores = self.processors(
                        tokens, next_token_scores)
                    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                        next_token_scores)
                    vocab_size = next_token_scores.shape[-1]
                    next_token_scores = next_token_scores.view(
                        batch_size, args.num_beams * vocab_size)

                    probs = F.softmax(next_token_scores, dim=-1)
                    if args.select_topk:
                        _, next_tokens = torch.topk(
                            probs, k=2 * args.num_beams, dim=-1, largest=True)
                    else:
                        next_tokens = torch.multinomial(
                            probs, num_samples=2 * args.num_beams)
                    next_token_scores = torch.gather(next_token_scores, -1,
                                                     next_tokens)
                    next_token_scores, _indices = torch.sort(
                        next_token_scores, descending=True, dim=1)
                    next_tokens = torch.gather(next_tokens, -1, _indices)

                    next_indices = next_tokens // vocab_size
                    next_tokens = next_tokens % vocab_size
                    # stateless
                    beam_outputs = beam_scorer.process(
                        tokens,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        eos_token_id=self.end_token,
                        pad_token_id=self.pad_token)
                    beam_scores = beam_outputs['next_beam_scores']
                    beam_next_tokens = beam_outputs['next_beam_tokens']
                    beam_idx = beam_outputs['next_beam_indices']
                    beam_next_tokens = beam_next_tokens.unsqueeze(-1)
                    tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens],
                                       dim=-1)
                    mems = [mem[beam_idx] for mem in mems] if mems else []
                    if beam_scorer.is_done:
                        break
                    counter += 1
                tokens, _ = beam_scorer.finalize(
                    tokens,
                    beam_scores,
                    next_tokens,
                    next_indices,
                    eos_token_id=self.end_token,
                    pad_token_id=self.pad_token)
                predictions = []
                for text in tokens.tolist():
                    text = [
                        token for token in text
                        if token not in [self.end_token, self.pad_token]
                    ]
                    text = self.tokenizer.DecodeIds(text)
                    predictions.append(text)
                uid_list = data['uid']
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                for uid, prediction in zip(uid_list, predictions):
                    store.set(uid, prediction)
                if (idx + 1) % args.log_interval == 0:
                    print_rank_0(f'Iteration {idx + 1} / {len(dataloader)}')
        model.train()
        torch.distributed.barrier()
        print_rank_0('Evaluation completed')
        predictions, examples = [], []
        for uid, example in example_dict.items():
            predictions.append(store.get(uid).decode('utf-8'))
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples


def blanklm_fix_tokenization(text):
    text = text.replace('` `', '``')
    text = text.replace("\' \'", "\'\'")
    text = text.replace("n \' t", "n\'t")
    text = text.replace("\' s", "\'s")
    text = text.replace("\' m", "\'m")
    text = text.replace("\' re", "\'re")
    text = text.replace('. . .', '...')
    text = text.replace(' . .', ' ..')
    text = text.replace('- -', '--')
    text = text.replace('u . s .', 'u.s.')
    text = text.replace('u . k .', 'u.k.')
    text = text.replace('e . g .', 'e.g.')
    return text


class BlankLMEvaluater(DecoderEvaluater):

    def evaluate(self, model, dataloader, example_dict, args):
        model.eval()
        store = torch.distributed.TCPStore(args.master_ip,
                                           18931 + random.randint(0, 10000),
                                           mpu.get_data_parallel_world_size(),
                                           torch.distributed.get_rank() == 0,
                                           datetime.timedelta(seconds=30))
        print_rank_0('Distributed store created')

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(
                    data, args)
                src_tokens = tokens
                batch_size = tokens.size(0)
                mask_positions = []
                current_mask = []
                for text in tokens.tolist():
                    mask_positions.append([
                        i for i, x in enumerate(text) if x == self.mask_token
                    ])
                    current_mask.append(0)
                    # print(self.tokenizer.DecodeIds(text))
                    # print(mask_positions[-1])
                counter = 0
                done = [False] * batch_size
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        # print(tokens)
                        # print(position_ids)
                        next_token_logits, *mems = model(
                            tokens,
                            position_ids,
                            attention_mask,
                            return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                        position_ids = tokens.new_ones(batch_size, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = mask_positions[i][current_mask[i]]
                            position_ids[i, 0] = mask_pos
                        tokens = tokens.new_zeros(batch_size, 0)
                        attention_mask = tokens.new_zeros(batch_size)
                    else:
                        position_ids[:, 1] = position_ids[:, 1] + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(
                            last_token,
                            position_ids,
                            attention_mask,
                            *mems,
                            return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(
                        next_token_logits, dim=-1)
                    next_token_scores = self.processors(
                        tokens, next_token_scores)
                    next_tokens = next_token_scores.max(dim=-1)[1]
                    # print(self.tokenizer.DecodeIds(next_tokens.tolist()))
                    for i, next_token in enumerate(next_tokens.tolist()):
                        if next_token == self.end_token:
                            if current_mask[i] + 1 < len(mask_positions[i]):
                                current_mask[i] += 1
                                next_tokens[i] = self.start_token
                                position_ids[i, 0] = mask_positions[i][
                                    current_mask[i]]
                                position_ids[i, 1] = 0
                            else:
                                done[i] = True
                        if done[i]:
                            next_tokens[i] = self.pad_token
                    if all(done):
                        break
                    tokens = torch.cat(
                        [tokens, next_tokens.unsqueeze(-1)], dim=-1)
                    counter += 1
                predictions = []
                for i, text in enumerate(tokens.tolist()):
                    text = [
                        token for token in text
                        if token not in [self.end_token, self.pad_token]
                    ]
                    blanks = [[]]
                    for token in text:
                        if token == self.start_token:
                            blanks.append([])
                        else:
                            blanks[-1].append(token)
                    output_tokens = []
                    current_blank = 0
                    for token in src_tokens[i].tolist():
                        if token == self.mask_token:
                            if current_blank < len(blanks):
                                output_tokens += blanks[current_blank]
                            current_blank += 1
                        else:
                            if token not in [self.pad_token]:
                                output_tokens.append(token)
                    text = self.tokenizer.DecodeIds(output_tokens[:-1])
                    text = blanklm_fix_tokenization(text)
                    predictions.append(text)
                    # print(text)
                uid_list = data['uid']
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                for uid, prediction in zip(uid_list, predictions):
                    store.set(uid, prediction)
                if (idx + 1) % args.log_interval == 0:
                    print_rank_0(f'Iteration {idx + 1} / {len(dataloader)}')

        model.train()
        torch.distributed.barrier()
        print_rank_0('Evaluation completed')
        predictions, examples = [], []
        for uid, example in example_dict.items():
            predictions.append(store.get(uid).decode('utf-8'))
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples
