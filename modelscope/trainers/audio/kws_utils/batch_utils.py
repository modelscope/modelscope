# Copyright (c) 2021 Binbin Zhang
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

import datetime
import math
import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ReduceOp
from torch.nn.utils import clip_grad_norm_

from modelscope.utils.logger import get_logger

logger = get_logger()

# torch.set_printoptions(threshold=np.inf)


def executor_train(model, optimizer, data_loader, device, writer, args):
    ''' Train one epoch
    '''
    model.train()
    clip = args.get('grad_clip', 50.0)
    log_interval = args.get('log_interval', 10)
    epoch = args.get('epoch', 0)

    rank = args.get('rank', 0)
    local_rank = args.get('local_rank', 0)
    world_size = args.get('world_size', 1)
    accum_batchs = args.get('grad_accum', 1)

    # [For distributed] Because iteration counts are not always equals between
    # processes, send stop-flag to the other processes if iterator is finished
    iterator_stop = torch.tensor(0).to(device)

    for batch_idx, batch in enumerate(data_loader):
        if world_size > 1:
            dist.all_reduce(iterator_stop, ReduceOp.SUM)
        if iterator_stop > 0:
            break

        key, feats, target, feats_lengths, target_lengths = batch
        feats = feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        if target_lengths is not None:
            target_lengths = target_lengths.to(device)
        num_utts = feats_lengths.size(0)
        if num_utts == 0:
            continue
        logits, _ = model(feats)
        loss, acc = ctc_loss(logits, target, feats_lengths, target_lengths)
        loss = loss / num_utts

        # normlize loss to account for batch accumulation
        loss = loss / accum_batchs
        loss.backward()
        if (batch_idx + 1) % accum_batchs == 0:
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad()

        if batch_idx % log_interval == 0:
            logger.info(
                'RANK {}/{}/{} TRAIN Batch {}/{} size {} loss {:.6f}'.format(
                    world_size, rank, local_rank, epoch, batch_idx, num_utts,
                    loss.item()))
    else:
        iterator_stop.fill_(1)
        if world_size > 1:
            dist.all_reduce(iterator_stop, ReduceOp.SUM)


def executor_cv(model, data_loader, device, args):
    ''' Cross validation on
    '''
    model.eval()
    log_interval = args.get('log_interval', 10)
    epoch = args.get('epoch', 0)
    # in order to avoid division by 0
    num_seen_utts = 1
    num_seen_tokens = 1
    total_loss = 0.0
    # [For distributed] Because iteration counts are not always equals between
    # processes, send stop-flag to the other processes if iterator is finished
    iterator_stop = torch.tensor(0).to(device)
    counter = torch.zeros((4, ), device=device)

    rank = args.get('rank', 0)
    local_rank = args.get('local_rank', 0)
    world_size = args.get('world_size', 1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if world_size > 1:
                dist.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop > 0:
                break

            key, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            if target_lengths is not None:
                target_lengths = target_lengths.to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue
            logits, _ = model(feats)

            loss, acc = ctc_loss(logits, target, feats_lengths, target_lengths,
                                 True)
            if torch.isfinite(loss):
                num_seen_utts += num_utts
                num_seen_tokens += target_lengths.sum()
                total_loss += loss.item()
                counter[0] += loss.item()
                counter[1] += acc * num_utts
                # counter[1] += acc * target_lengths.sum()
                counter[2] += num_utts
                counter[3] += target_lengths.sum()

            if batch_idx % log_interval == 0:
                logger.info(
                    'RANK {}/{}/{} CV Batch {}/{} size {} loss {:.6f} acc {:.2f} history loss {:.6f}'
                    .format(world_size, rank, local_rank, epoch, batch_idx,
                            num_utts,
                            loss.item() / num_utts, acc,
                            total_loss / num_seen_utts))
        else:
            iterator_stop.fill_(1)
            if world_size > 1:
                dist.all_reduce(iterator_stop, ReduceOp.SUM)

    if world_size > 1:
        dist.all_reduce(counter, ReduceOp.SUM)
    logger.info('Total utts number is {}'.format(counter[2]))
    counter = counter.to('cpu')

    return counter[0].item() / counter[2].item(), counter[1].item(
    ) / counter[2].item()


def executor_test(model, data_loader, device, keywords_token, keywords_idxset,
                  args):
    ''' Test model with decoder
    '''
    assert args.get('test_dir', None) is not None, \
        'Please config param: test_dir, to store score file'
    score_abs_path = os.path.join(args['test_dir'], 'score.txt')
    log_interval = args.get('log_interval', 10)

    model.eval()
    infer_seconds = 0.0
    decode_seconds = 0.0
    with torch.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch in enumerate(data_loader):
            batch_start_time = datetime.datetime.now()

            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            feats_lengths = feats_lengths.to(device)
            if target_lengths is not None:
                target_lengths = target_lengths.to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue

            logits, _ = model(feats)
            logits = logits.softmax(2)  # (1, maxlen, vocab_size)
            logits = logits.cpu()

            infer_end_time = datetime.datetime.now()
            for i in range(len(keys)):
                key = keys[i]
                score = logits[i][:feats_lengths[i]]
                hyps = ctc_prefix_beam_search(score, feats_lengths[i],
                                              keywords_idxset)
                hit_keyword = None
                hit_score = 1.0
                # start = 0; end = 0
                for one_hyp in hyps:
                    prefix_ids = one_hyp[0]
                    # path_score = one_hyp[1]
                    prefix_nodes = one_hyp[2]
                    assert len(prefix_ids) == len(prefix_nodes)
                    for word in keywords_token.keys():
                        lab = keywords_token[word]['token_id']
                        offset = is_sublist(prefix_ids, lab)
                        if offset != -1:
                            hit_keyword = word
                            # start = prefix_nodes[offset]['frame']
                            # end = prefix_nodes[offset+len(lab)-1]['frame']
                            for idx in range(offset, offset + len(lab)):
                                hit_score *= prefix_nodes[idx]['prob']
                            break
                    if hit_keyword is not None:
                        hit_score = math.sqrt(hit_score)
                        break

                if hit_keyword is not None:
                    # fout.write('{} detected [{:.2f} {:.2f}] {} {:.3f}\n'\
                    #          .format(key, start*0.03, end*0.03, hit_keyword, hit_score))
                    fout.write('{} detected {} {:.3f}\n'.format(
                        key, hit_keyword, hit_score))
                else:
                    fout.write('{} rejected\n'.format(key))

            decode_end_time = datetime.datetime.now()
            infer_seconds += (infer_end_time
                              - batch_start_time).total_seconds()
            decode_seconds += (decode_end_time
                               - infer_end_time).total_seconds()

            if batch_idx % log_interval == 0:
                logger.info('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()
        logger.info(
            'Total infer cost {:.2f} mins, decode cost {:.2f} mins'.format(
                infer_seconds / 60.0,
                decode_seconds / 60.0,
            ))

    return score_abs_path


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
        return -1


def ctc_loss(logits: torch.Tensor,
             target: torch.Tensor,
             logits_lengths: torch.Tensor,
             target_lengths: torch.Tensor,
             need_acc: bool = False):
    """ CTC Loss
    Args:
        logits: (B, D), D is the number of keywords plus 1 (non-keyword)
        target: (B)
        logits_lengths: (B)
        target_lengths: (B)
    Returns:
        (float): loss of current batch
    """

    acc = 0.0
    if need_acc:
        acc = acc_utterance(logits, target, logits_lengths, target_lengths)

    # logits: (B, L, D) -> (L, B, D)
    logits = logits.transpose(0, 1)
    logits = logits.log_softmax(2)
    loss = F.ctc_loss(
        logits, target, logits_lengths, target_lengths, reduction='sum')
    # loss = loss / logits.size(1)

    return loss, acc


def acc_utterance(logits: torch.Tensor, target: torch.Tensor,
                  logits_length: torch.Tensor, target_length: torch.Tensor):
    if logits is None:
        return 0

    logits = logits.softmax(2)  # (1, maxlen, vocab_size)
    logits = logits.cpu()
    target = target.cpu()

    total_word = 0
    total_ins = 0
    total_sub = 0
    total_del = 0
    calculator = Calculator()
    for i in range(logits.size(0)):
        score = logits[i][:logits_length[i]]
        hyps = ctc_prefix_beam_search(score, logits_length[i], None, 3, 5)
        lab = [str(item) for item in target[i][:target_length[i]].tolist()]
        rec = []
        if len(hyps) > 0:
            rec = [str(item) for item in hyps[0][0]]
        result = calculator.calculate(lab, rec)
        # print(f'result:{result}')
        if result['all'] != 0:
            total_word += result['all']
            total_ins += result['ins']
            total_sub += result['sub']
            total_del += result['del']

    return float(total_word - total_ins - total_sub
                 - total_del) * 100.0 / total_word


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    logits_lengths: torch.Tensor,
    keywords_tokenset: set = None,
    score_beam_size: int = 3,
    path_beam_size: int = 20,
) -> Tuple[List[List[int]], torch.Tensor]:
    """ CTC prefix beam search inner implementation

    Args:
        logits (torch.Tensor): (1, max_len, vocab_size)
        logits_lengths (torch.Tensor): (1, )
        keywords_tokenset (set): token set for filtering score
        score_beam_size (int): beam size for score
        path_beam_size (int): beam size for path

    Returns:
        List[List[int]]: nbest results
    """
    maxlen = logits.size(0)
    # ctc_probs = logits.softmax(1)  # (1, maxlen, vocab_size)
    ctc_probs = logits

    cur_hyps = [(tuple(), (1.0, 0.0, []))]

    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        probs = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))

        # 2.1 First beam prune: select topk best
        top_k_probs, top_k_index = probs.topk(
            score_beam_size)  # (score_beam_size,)

        # filter prob score that is too small
        filter_probs = []
        filter_index = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
            if keywords_tokenset is not None:
                if prob > 0.05 and idx in keywords_tokenset:
                    filter_probs.append(prob)
                    filter_index.append(idx)
            else:
                if prob > 0.05:
                    filter_probs.append(prob)
                    filter_index.append(idx)

        if len(filter_index) == 0:
            continue

        for s in filter_index:
            ps = probs[s].item()

            for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pb = n_pb + pb * ps + pnb * ps
                    nodes = cur_nodes.copy()
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)
                elif s == last:
                    if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                        # Update *ss -> *s;
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pnb = n_pnb + pnb * ps
                        nodes = cur_nodes.copy()
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)

                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        cur_hyps = next_hyps[:path_beam_size]

    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps


class Calculator:

    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost['cor'] = 0
        self.cost['sub'] = 1
        self.cost['del'] = 1
        self.cost['ins'] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, '')
        rec.insert(0, '')
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(rec):
                row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)):
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(rec)):
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i - 1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['cor']
                    error = 'cor'
                else:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error
        # Tracing back
        result = {
            'lab': [],
            'rec': [],
            'all': 0,
            'cor': 0,
            'sub': 0,
            'ins': 0,
            'del': 0
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]['error'] == 'cor':  # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub':  # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del':  # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, '')
                i = i - 1
            elif self.space[i][j]['error'] == 'ins':  # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['lab'].insert(0, '')
                result['rec'].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non':  # starting point
                break
            else:  # shouldn't reach here
                print(
                    'this should not happen , i = {i} , j = {j} , error = {error}'
                    .format(i=i, j=j, error=self.space[i][j]['error']))
        return result

    def overall(self):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in self.data:
            result['all'] = result['all'] + self.data[token]['all']
            result['cor'] = result['cor'] + self.data[token]['cor']
            result['sub'] = result['sub'] + self.data[token]['sub']
            result['ins'] = result['ins'] + self.data[token]['ins']
            result['del'] = result['del'] + self.data[token]['del']
        return result

    def cluster(self, data):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in data:
            if token in self.data:
                result['all'] = result['all'] + self.data[token]['all']
                result['cor'] = result['cor'] + self.data[token]['cor']
                result['sub'] = result['sub'] + self.data[token]['sub']
                result['ins'] = result['ins'] + self.data[token]['ins']
                result['del'] = result['del'] + self.data[token]['del']
        return result

    def keys(self):
        return list(self.data.keys())
