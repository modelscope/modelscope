import os
import pprint
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import ujson as json
from torch import nn

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.models.nlp.ponet.configuration import PoNetConfig
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DocumentGroundedDialogRerankPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['DocumentGroundedDialogRerankPipeline']


@PIPELINES.register_module(
    Tasks.document_grounded_dialog_rerank,
    module_name=Pipelines.document_grounded_dialog_rerank)
class DocumentGroundedDialogRerankPipeline(Pipeline):

    def __init__(self,
                 model: Union[DocumentGroundedDialogRerankModel, str],
                 preprocessor: DocumentGroundedDialogRerankPreprocessor = None,
                 config_file: str = None,
                 device: str = 'cuda',
                 auto_collate=True,
                 seed: int = 88,
                 **kwarg):
        """The Rerank pipeline for document grounded dialog

        Args:
            model: A model instance or a model local dir or a model id in the model hub.
            preprocessor: A preprocessor instance.
            config_file: Path to config file.
            device: Device to run the model.
            auto_collate: Apply auto collate.
            seed: Random seeds of random parameters.
            **kwargs: The preprocessor kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipe_ins = pipeline('document_grounded_dialog_rerank', model='damo/nlp_convai_rerank')
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            seed=seed)
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        if kwarg['model_resize']:
            self.model.resize_token_embeddings(
                len(self.preprocessor.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        self.args = kwarg
        # self.model_cfg = self.model.model_cfg
        set_seed(seed)

    def one_instance(self, input_ids, attention_mask):
        all_probs = []
        for start_ndx in range(0, len(input_ids), self.args['max_batch_size']):
            probs = F.softmax(
                self.model({
                    'input_ids':
                    input_ids[start_ndx:start_ndx
                              + self.args['max_batch_size']],
                    'attention_mask':
                    attention_mask[start_ndx:start_ndx
                                   + self.args['max_batch_size']]
                }).logits.detach().cpu(),
                dim=-1)[:, 1].numpy().tolist()
            all_probs.extend(probs)
        return all_probs

    def forward(self, dataset: Union[list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        report = Reporting()
        self.guess = []
        with torch.no_grad():
            for jobj in dataset:
                inst_id = jobj['id']
                probs = self.one_instance(jobj['input_ids'],
                                          jobj['attention_mask'])
                passages = jobj['passages']
                query = jobj['query']
                scored_pids = [(p['pid'], prob)
                               for p, prob in zip(passages, probs)]
                scored_pids.sort(key=lambda x: x[1], reverse=True)
                wids = to_distinct_doc_ids([
                    pid for pid, prob in scored_pids
                ])  # convert to Wikipedia document ids
                pred_record = {
                    'id':
                    inst_id,
                    'input':
                    query,
                    'scored_pids':
                    scored_pids,
                    'output': [{
                        'answer':
                        '',
                        'provenance': [{
                            'wikipedia_id': wid
                        } for wid in wids]
                    }]
                }
                if self.args['include_passages']:
                    pred_record['passages'] = passages

                if report.is_time():
                    print(
                        f'Finished {report.check_count}; {report.check_count / report.elapsed_seconds()} per second.'
                    )
                self.guess.append(pred_record)
        # if args['kilt_data']:
        #     evaluate(dataset, args['output'])

    def postprocess(self, inputs: list):
        return {OutputKeys.OUTPUT: inputs}


class Reporting:

    def __init__(self,
                 *,
                 recency_weight=0.001,
                 report_interval_secs=300,
                 check_every=1,
                 gather_samples: Iterable = (),
                 num_samples=10000):
        """The Reporting to print parameter status

        Args:
            recency_weight: when computing the moving average, how much weight to give to the current sample.
            report_interval_secs: how many seconds between returning true for is_time.
            check_every: how often to check the time, when calling is_time.
            gather_samples: keep the last num_samples of the listed names (gathered from moving_averages).
            num_samples: how many samples to keep.
        """
        self.check_count = 0
        self.check_every = check_every
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_interval_secs = report_interval_secs
        # For tracking moving averages of various values
        self.names = None
        self.averages = None
        self.counts = None
        self.recency_weight = recency_weight
        self.per_value_recency_weight = dict()
        self.report_count = 0
        self._prev_check_count = 0
        self.sample_names = list(gather_samples)
        if len(self.sample_names) > 0:
            self.sample_values = np.zeros(
                (len(self.sample_names), num_samples), dtype=np.float32)
            self.sample_ndxs = np.zeros(len(self.sample_names), dtype=np.int32)
        else:
            self.sample_values = None
            self.sample_ndxs = None

    def reset(self):
        self.check_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_count = 0
        self._prev_check_count = 0
        if len(self.sample_names) > 0:
            self.sample_values[:, :] = 0
            self.sample_ndxs[:] = 0
        if self.counts is not None:
            self.counts[:] = 0
            self.averages[:] = 0

    def is_time(self):
        self.check_count += 1
        if self.check_count % self.check_every == 0:
            elapsed = time.time() - self.last_time
            if elapsed >= self.report_interval_secs:
                # check the time more or less often
                if self.check_every > 1 and self.check_count - self._prev_check_count < 5 * self.check_every:
                    self.check_every //= 2
                elif self.check_count - self._prev_check_count > 50 * self.check_every:
                    self.check_every *= 2
                self.last_time = time.time()
                self.report_count += 1
                self._prev_check_count = self.check_count
                return True
        return False

    def moving_averages(self, **values):
        # create entries in avgs and counts when needed
        # update the avgs and counts
        if self.names is None:
            self.names = list(values.keys())
            self.averages = np.zeros(len(self.names))
            self.counts = np.zeros(len(self.names))
        for name in values.keys():
            if name not in self.names:
                self.names.append(name)
        if self.averages.shape[0] < len(self.names):
            old_len = self.averages.shape[0]
            self.averages = np.resize(self.averages, len(self.names))
            self.averages[old_len:] = 0
            self.counts = np.resize(self.counts, len(self.names))
            self.counts[old_len:] = 0
        for ndx, name in enumerate(self.names):
            if name in values:
                self.counts[ndx] += 1
                # support per-name recency_weight
                if name in self.per_value_recency_weight:
                    rweight = max(self.per_value_recency_weight[name],
                                  1.0 / self.counts[ndx])
                else:
                    rweight = max(self.recency_weight, 1.0 / self.counts[ndx])
                self.averages[ndx] = rweight * values[name] + (
                    1.0 - rweight) * self.averages[ndx]
        for ndx, name in enumerate(self.sample_names):
            if name in values:
                self.sample_values[self.sample_ndxs[ndx]] = values[name]
                self.sample_ndxs[ndx] = (self.sample_ndxs[ndx]
                                         + 1) % self.sample_values.shape[1]

    def get_samples(self, name):
        for ndx, n in enumerate(self.sample_names):
            if n == name:
                count = self.get_count(name)
                if count is None:
                    count = 0
                return self.sample_values[ndx, 0:count]  # NOTE: not in order
        return None

    def get_moving_average(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.averages[ndx]
        return None

    def get_count(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.counts[ndx]
        return None

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def elapsed_time_str(self) -> str:
        return time_str(self.elapsed_seconds())

    def progress_str(self, instance_name='instance'):
        return f'On {instance_name} {self.check_count}, ' \
               f'{self.check_count / self.elapsed_seconds()} {instance_name}s per second.'

    def display(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.info(f'{prefix}{n} = {v}')

    def display_warn(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.warning(f'{prefix}{n} = {v}')


def _remove_duplicates(obj):
    obj_tmp = []
    for o in obj:
        if o not in obj_tmp:
            obj_tmp.append(o)
    return obj_tmp


def _get_ids_list(datapoint, rank_keys, verbose=False):
    # collect all gold ids
    ids_list = []
    for output in datapoint['output']:
        current_ids_list = []
        if 'provenance' in output:
            for provenance in output['provenance']:
                if any(rank_key not in provenance for rank_key in rank_keys):
                    missing = set(rank_keys) - set(list(
                        provenance.keys())).intersection(set(rank_keys))
                    if verbose:
                        print(
                            f'WARNING: missing key(s) {missing} in provenance, unable to compute retrieval for those.'
                        )
                else:
                    current_ids_list.append('+'.join([
                        str(provenance[rank_key]).strip()
                        for rank_key in rank_keys
                    ]))
        ids_list.append(
            _remove_duplicates(current_ids_list))  # remove duplicates

    # consider only unique ids
    return ids_list


def _computeRprec(guess_ids, gold_ids):
    R = len(gold_ids)
    num = 0

    for prediction in guess_ids[:R]:
        if str(prediction).strip() in gold_ids:
            num += 1

    Rprec = num / R if R > 0 else 0
    return Rprec


# 1. Precision computation
def _precision_at_k(rank, k):
    # precision @ k
    p = rank[:k].count(True) / k

    return p


# 2. Recall computation
def _recall_at_k(rank, num_distinct_evidence_sets, k):
    r = rank[:k].count(True) / num_distinct_evidence_sets

    return r


# 3. Success rate computation
def _success_rate_at_k(rank, k):
    # success rate @ k
    p = int(True in rank[:k])

    return p


def get_rank(guess_item, gold_item, k, rank_keys, verbose=False):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, 'k must be a positive integer grater than 0.'

    rank = []
    num_distinct_evidence_sets = 0

    guess_ids = _get_ids_list(guess_item, rank_keys)[0]

    if guess_ids and len(guess_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in gold_item['output']:
            if 'provenance' in output:
                e_set = {
                    '+'.join([
                        str(provenance[rank_key]).strip()
                        for rank_key in rank_keys
                    ])
                    for provenance in output['provenance']
                }
                if e_set not in evidence_sets:  # no duplicate evidence set
                    evidence_sets.append(e_set)
                    e_size[len(e_set)] += 1
        num_distinct_evidence_sets = len(evidence_sets)

        # 2. check what's the minimum number of predicted pages needed to get a robust P/R@k
        min_prediction_size = 0
        c = 0
        for size, freq in sorted(e_size.items(), reverse=True):
            for _ in range(freq):
                min_prediction_size += size
                c += 1
                if c == k:
                    break
            if c == k:
                break
        # if the number of evidence sets is smaller than k
        min_prediction_size += k - c

        if verbose and len(guess_ids) < min_prediction_size:
            print(
                f'WARNING: you should provide at least {min_prediction_size} provenance items '
                f'for a robust recall@{k} computation (you provided {len(guess_ids)} item(s)).'
            )

        # 3. rank by gruping pages in each evidence set (each evidence set count as 1),
        # the position in the rank of each evidence set is given by the last page in guess_ids
        # non evidence pages counts as 1
        rank = []
        for guess_id in guess_ids:
            guess_id = str(guess_id).strip()
            found = False
            for idx, e_set in enumerate(evidence_sets):

                e_set_id = f'evidence_set:{idx}'

                if guess_id in e_set:
                    found = True

                    # remove from the rank previous points referring to this evidence set
                    if e_set_id in rank:
                        rank.remove(e_set_id)

                    # remove the guess_id from the evidence set
                    e_set.remove(guess_id)

                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
                    else:
                        # add a point for this partial evidence set
                        rank.append(e_set_id)

            if not found:
                rank.append(False)

    return rank, num_distinct_evidence_sets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(filename):
    data = []
    file_in = open(filename, 'r')
    lines = file_in.readlines()
    for line in lines:
        data.append(json.loads(line))
    return data


def rprecision(guess_item, gold_item, rank_keys):
    gold_ids_list = _get_ids_list(gold_item, rank_keys)
    guess_ids = _get_ids_list(guess_item, rank_keys)[0]
    Rprec_vector = []
    for gold_ids in gold_ids_list:
        Rprec = _computeRprec(guess_ids, gold_ids)
        Rprec_vector.append(Rprec)
    return max(Rprec_vector)


def get_ranking_metrics(guess_item, gold_item, ks, rank_keys):
    Rprec = 0
    P_at_k = {'precision@{}'.format(k): 0 for k in sorted(ks) if k > 0}
    R_at_k = {'recall@{}'.format(k): 0 for k in sorted(ks) if k > 1}
    S_at_k = {'success_rate@{}'.format(k): 0 for k in sorted(ks) if k > 1}

    assert (
        'output' in guess_item and len(guess_item['output']) == 1
    ), f"guess should provide exactly one output for {guess_item['id']}"

    Rprec = rprecision(guess_item, gold_item, rank_keys=rank_keys)
    for k in ks:

        # 0. get rank
        rank, num_distinct_evidence_sets = get_rank(
            guess_item, gold_item, k, rank_keys=rank_keys)

        if num_distinct_evidence_sets > 0:
            # 1. precision
            P_at_k['precision@{}'.format(k)] = _precision_at_k(rank, k)

            # 2. recall
            R_at_k['recall@{}'.format(k)] = _recall_at_k(
                rank, num_distinct_evidence_sets, k)

            # 3. success rate
            S_at_k['success_rate@{}'.format(k)] = _success_rate_at_k(rank, k)

        # else:
        #     print(
        #         "WARNING: the number of distinct evidence sets is 0 for {}".format(
        #             gold_item
        #         )
        #     )

    return {'Rprec': Rprec, **P_at_k, **R_at_k, **S_at_k}


def compute(gold_dataset, guess_dataset, ks, rank_keys):
    ks = sorted([int(x) for x in ks])

    result = OrderedDict()
    result['Rprec'] = 0.0
    for k in ks:
        if k > 0:
            result['precision@{}'.format(k)] = 0.0
            # if k > 1:
            result['recall@{}'.format(k)] = 0.0
            result['success_rate@{}'.format(k)] = 0.0

    assert len(guess_dataset) == len(
        gold_dataset), 'different size gold: {} guess: {}'.format(
            len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert (str(gold['id']).strip() == str(
            guess['id']).strip()), 'Items must have same order with same IDs'

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks,
                                              rank_keys)
        result['Rprec'] += ranking_metrics['Rprec']
        for k in ks:
            if k > 0:
                result['precision@{}'.format(k)] += ranking_metrics[
                    'precision@{}'.format(k)]
                result['recall@{}'.format(k)] += ranking_metrics[
                    'recall@{}'.format(k)]
                result['success_rate@{}'.format(k)] += ranking_metrics[
                    'success_rate@{}'.format(k)]

    if len(guess_dataset) > 0:
        result['Rprec'] /= len(guess_dataset)
        for k in ks:
            if k > 0:
                result['precision@{}'.format(k)] /= len(guess_dataset)
                # if k > 1:
                result['recall@{}'.format(k)] /= len(guess_dataset)
                result['success_rate@{}'.format(k)] /= len(guess_dataset)

    return result


def to_distinct_doc_ids(passage_ids):
    doc_ids = []
    for pid in passage_ids:
        # MARK
        doc_id = pid
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def validate_input(gold_records, guess_records):
    if len(gold_records) != len(guess_records):
        print('WARNING: DIFFERENT SIZE gold: {} guess: {}'.format(
            len(gold_records), len(guess_records)))

    # align order
    gold_ids = []
    for gold in gold_records:
        assert str(
            gold['id']).strip() not in gold_ids, 'Gold IDs should be unique'
        gold_ids.append(str(gold['id']).strip())

    id2guess_record = {}
    for guess in guess_records:
        assert (str(guess['id']).strip()
                not in id2guess_record), 'Prediction IDs should be unique'
        id2guess_record[str(guess['id']).strip()] = guess

    guess_records = []
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])
        else:
            raise ValueError(
                'ERROR: no prediction provided for id: {}'.format(id))

    return gold_records, guess_records


# utility to get gold answers
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold['output']:
        if 'answer' in item and item['answer'] and len(
                item['answer'].strip()) > 0:
            ground_truths.add(item['answer'].strip())
    return ground_truths


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    if scores_for_ground_truths:
        return max(scores_for_ground_truths)
    else:
        return 0


def _calculate_metrics(gold_records, guess_records):
    assert len(gold_records) == len(
        guess_records), 'different size gold: {} guess: {}'.format(
            len(gold_records), len(guess_records))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    # kilt metrics
    kilt_accuracy = 0
    kilt_em = 0
    kilt_f1 = 0
    kilt_rougel = 0

    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (str(gold_item['id']).strip() == str(guess_item['id']).strip()
                ), 'Items must have same order with same IDs'

        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        conditions = (len(guess_item['output'])
                      == 1) and ('answer' in guess_item['output'][0])
        assert (
            conditions
        ), f"you should provide exactly one valid answer for {guess_item['id']}"
        guess_answer = str(guess_item['output'][0]['answer']).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(_exact_match_score,
                                                  guess_answer,
                                                  gold_candidate_answers)
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(_f1_score, guess_answer,
                                                  gold_candidate_answers)
        normalized_f1 += local_f1

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(_rougel_score,
                                                      guess_answer,
                                                      gold_candidate_answers)
        rougel += local_rougel

        # KILT-metrics
        Rprec = rprecision(guess_item, gold_item, rank_keys=['wikipedia_id'])
        if Rprec == 1:
            # 1. KILT-AC
            kilt_accuracy += local_accuracy

            # 2. KILT-EM
            kilt_em += local_em

            # 3. KILT-F1
            kilt_f1 += local_f1

            # 4. KILT-RL
            kilt_rougel += local_rougel

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        kilt_accuracy /= total_count
        kilt_em /= total_count
        kilt_f1 /= total_count
        kilt_rougel /= total_count

    return {
        'kilt': {
            'KILT-accuracy': kilt_accuracy,
            'KILT-em': kilt_em,
            'KILT-f1': kilt_f1,
            'KILT-rougel': kilt_rougel,
        },
        'downstream': {
            'accuracy': accuracy,
            'em': normalized_em,
            'f1': normalized_f1,
            'rougel': rougel,
        },
    }


def evaluate(gold, guess):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = gold
    guess_records = load_data(guess)

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    # 2. retrieval performance
    retrieval_results = compute(
        gold_records,
        guess_records,
        ks=[1, 5, 10, 100],
        rank_keys=['wikipedia_id'])
    result['retrieval'] = {
        'Rprec': retrieval_results['Rprec'],
        'recall@1': retrieval_results['recall@1'],
        'recall@5': retrieval_results['recall@5'],
        'recall@10': retrieval_results['recall@10'],
        'recall@100': retrieval_results['recall@100'],
    }

    pp.pprint(result)
    return result


if __name__ == '__main__':
    main()
