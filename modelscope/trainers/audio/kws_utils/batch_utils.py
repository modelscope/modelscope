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
        loss = ctc_loss(logits, target, feats_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), clip)
        if torch.isfinite(grad_norm):
            optimizer.step()
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
    total_loss = 0.0
    # [For distributed] Because iteration counts are not always equals between
    # processes, send stop-flag to the other processes if iterator is finished
    iterator_stop = torch.tensor(0).to(device)
    counter = torch.zeros((3, ), device=device)

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
            loss = ctc_loss(logits, target, feats_lengths, target_lengths)
            if torch.isfinite(loss):
                num_seen_utts += num_utts
                total_loss += loss.item() * num_utts
                counter[0] += loss.item() * num_utts
                counter[1] += num_utts

            if batch_idx % log_interval == 0:
                logger.info(
                    'RANK {}/{}/{} CV Batch {}/{} size {} loss {:.6f} history loss {:.6f}'
                    .format(world_size, rank, local_rank, epoch, batch_idx,
                            num_utts, loss.item(), total_loss / num_seen_utts))
        else:
            iterator_stop.fill_(1)
            if world_size > 1:
                dist.all_reduce(iterator_stop, ReduceOp.SUM)

    if world_size > 1:
        dist.all_reduce(counter, ReduceOp.SUM)
    logger.info('Total utts number is {}'.format(counter[1]))
    counter = counter.to('cpu')

    return counter[0].item() / counter[1].item()


def executor_test(model, data_loader, device, keywords_token,
                  keywords_tokenset, args):
    ''' Test model with decoder
    '''
    assert args.get('test_dir', None) is not None, \
        'Please config param: test_dir, to store score file'
    score_abs_path = os.path.join(args['test_dir'], 'score.txt')
    log_interval = args.get('log_interval', 10)

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
                                              keywords_tokenset)

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


def ctc_loss(logits: torch.Tensor, target: torch.Tensor,
             logits_lengths: torch.Tensor, target_lengths: torch.Tensor):
    """ CTC Loss
    Args:
        logits: (B, D), D is the number of keywords plus 1 (non-keyword)
        target: (B)
        logits_lengths: (B)
        target_lengths: (B)
    Returns:
        (float): loss of current batch
    """

    # logits: (B, L, D) -> (L, B, D)
    logits = logits.transpose(0, 1)
    logits = logits.log_softmax(2)
    loss = F.ctc_loss(
        logits, target, logits_lengths, target_lengths, reduction='sum')
    loss = loss / logits.size(1)

    return loss


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
            if prob > 0.05 and idx in keywords_tokenset:
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
