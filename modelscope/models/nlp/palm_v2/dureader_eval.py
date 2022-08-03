# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module computes evaluation metrics for DuReader dataset.
"""

import argparse
import copy
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict

import json
import numpy as np
from rouge import Rouge

EMPTY = ''
YESNO_LABELS = set(['Yes', 'No', 'Depends'])


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,
                                 len(sub) + 1)]
               for j in range(0,
                              len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Bleu:

    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):
        assert (list(gts.keys()) == list(res.keys()))
        imgIds = list(gts.keys())

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        return score, scores

    def method(self):
        return 'Bleu'


def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return (len(words), counts)


def cook_refs(refs, eff=None, n=4):  # lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    # Calculate effective reference sentence length.
    if eff == 'shortest':
        reflen = min(reflen)
    elif eff == 'average':
        reflen = float(sum(reflen)) / len(reflen)

    # lhuang: N.B.: leave reflen computaiton to the very end!!

    # lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return reflen, maxcounts


def cook_test(test, xxx_todo_changeme, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflen, refmaxcounts) = xxx_todo_changeme
    testlen, counts = precook(test, n, True)

    result = {}

    # Calculate effective reference sentence length.

    if eff == 'closest':
        result['reflen'] = min((abs(ref - testlen), ref) for ref in reflen)[1]
    else:  # i.e., "average" or "shortest" or None
        result['reflen'] = reflen

    result['testlen'] = testlen

    result['guess'] = [max(0, testlen - k + 1) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    for (ngram, count) in counts.items():
        result['correct'][len(ngram) - 1] += min(
            refmaxcounts.get(ngram, 0), count)

    return result


class BleuScorer(object):
    """Bleu scorer.
    """

    __slots__ = 'n', 'crefs', 'ctest', '_score', '_ratio', '_testlen', '_reflen', 'special_reflen'

    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)  # N.B.: -1
            else:
                self.ctest.append(
                    None)  # lens of crefs and ctest have to match

        self._score = None  # need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        '''return (bleu, len_ratio) pair'''
        return (self.fscore(option=option), self.ratio(option=option))

    def score_ratio_str(self, option=None):
        return '%.4f (%.2f)' % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''

        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(
            self.ctest), 'refs/test mismatch! %d<>%d' % (len(
                self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            # avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), 'incompatible BLEUs.'
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None  # need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option='average'):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):

        if option == 'shortest':
            reflen = min(reflens)
        elif option == 'average':
            reflen = float(sum(reflens)) / len(reflens)
        elif option == 'closest':
            reflen = min((abs(ref - testlen), ref) for ref in reflens)[1]
        else:
            assert False, 'unsupported reflen option %s' % option

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  # so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = 'average' if len(self.crefs) == 1 else 'closest'

        self._testlen = 0
        self._reflen = 0
        totalcomps = {
            'testlen': 0,
            'reflen': 0,
            'guess': [0] * n,
            'correct': [0] * n
        }

        # for each sentence
        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:  # need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) / (
                    float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu**(1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small
                                        )  # N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) / (
                totalcomps['guess'][k] + small)
            bleus.append(bleu**(1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small
                                          )  # N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print('ratio:', ratio)

        self._score = bleus
        return self._score, bleu_list


def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized


def data_check(obj, task):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    assert 'question_id' in obj, "Missing 'question_id' field."
    assert 'question_type' in obj, \
        "Missing 'question_type' field. question_id: {}".format(obj['question_type'])

    assert 'yesno_answers' in obj, \
        "Missing 'yesno_answers' field. question_id: {}".format(obj['question_id'])
    assert isinstance(obj['yesno_answers'], list), \
        r"""'yesno_answers' field must be a list, if the 'question_type' is not
            'YES_NO', then this field should be an empty list.
            question_id: {}""".format(obj['question_id'])

    assert 'entity_answers' in obj, \
        "Missing 'entity_answers' field. question_id: {}".format(obj['question_id'])
    assert isinstance(
        obj['entity_answers'],
        list) and len(obj['entity_answers']) > 0, r"""'entity_answers' field
            must be a list, and has at least one element, which can be a empty list.
            question_id: {}""".format(obj['question_id'])


def read_file(file_name, task, is_ref=False):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping question_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - question_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entity_answers: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """

    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode)

    results = {}
    keys = ['answers', 'yesno_answers', 'entity_answers', 'question_type']
    if is_ref:
        keys += ['source']

    zf = zipfile.ZipFile(file_name,
                         'r') if file_name.endswith('.zip') else None
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        for line in _open(fn, 'r', zip_obj=zf):
            try:
                obj = json.loads(line.strip())
            except ValueError:
                raise ValueError('Every line of data should be legal json')
            data_check(obj, task)
            qid = obj['question_id']
            assert qid not in results, 'Duplicate question_id: {}'.format(qid)
            results[qid] = {}
            for k in keys:
                results[qid][k] = obj[k]
    return results


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
        'missing keys: {}'.format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    # rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    rouge_score = Rouge().get_scores(
        list(map(lambda x: x[0], pred_dict.values())),
        list(map(lambda x: x[0], ref_dict.values())))
    rouge_score = sum([d['rouge-l']['f']
                       for d in rouge_score]) / len(rouge_score)
    scores['Rouge-L'] = rouge_score
    return scores


def local_prf(pred_list, ref_list):
    """
    Compute local precision recall and f1-score,
    given only one prediction list and one reference list
    """
    common = Counter(pred_list) & Counter(ref_list)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(pred_list)
    r = 1.0 * num_same / len(ref_list)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def compute_prf(pred_dict, ref_dict):
    """
    Compute precision recall and f1-score.
    """
    # pred_question_ids = set(pred_dict.keys())
    ref_question_ids = set(ref_dict.keys())
    correct_preds, total_correct, total_preds = 0, 0, 0
    for question_id in ref_question_ids:
        pred_entity_list = pred_dict.get(question_id, [[]])
        assert len(pred_entity_list) == 1, \
            'the number of entity list for question_id {} is not 1.'.format(question_id)
        pred_entity_list = pred_entity_list[0]
        all_ref_entity_lists = ref_dict[question_id]
        best_local_f1 = 0
        best_ref_entity_list = None
        for ref_entity_list in all_ref_entity_lists:
            local_f1 = local_prf(pred_entity_list, ref_entity_list)[2]
            if local_f1 > best_local_f1:
                best_ref_entity_list = ref_entity_list
                best_local_f1 = local_f1
        if best_ref_entity_list is None:
            if len(all_ref_entity_lists) > 0:
                best_ref_entity_list = sorted(
                    all_ref_entity_lists, key=lambda x: len(x))[0]
            else:
                best_ref_entity_list = []
        gold_entities = set(best_ref_entity_list)
        pred_entities = set(pred_entity_list)
        correct_preds += len(gold_entities & pred_entities)
        total_preds += len(pred_entities)
        total_correct += len(gold_entities)
    p = float(correct_preds) / total_preds if correct_preds > 0 else 0
    r = float(correct_preds) / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {'Precision': p, 'Recall': r, 'F1': f1}


def prepare_prf(pred_dict, ref_dict):
    """
    Prepares data for calculation of prf scores.
    """
    preds = {k: v['entity_answers'] for k, v in pred_dict.items()}
    refs = {k: v['entity_answers'] for k, v in ref_dict.items()}
    return preds, refs


def filter_dict(result_dict, key_tag):
    """
    Filter a subset of the result_dict, where keys ends with 'key_tag'.
    """
    filtered = {}
    for k, v in result_dict.items():
        if k.endswith(key_tag):
            filtered[k] = v
    return filtered


def get_metrics(pred_result, ref_result, task, source):
    """
    Computes metrics.
    """
    metrics = {}

    ref_result_filtered = {}
    pred_result_filtered = {}
    if source == 'both':
        ref_result_filtered = ref_result
        pred_result_filtered = pred_result
    else:
        for question_id, info in ref_result.items():
            if info['source'] == source:
                ref_result_filtered[question_id] = info
                if question_id in pred_result:
                    pred_result_filtered[question_id] = pred_result[
                        question_id]

    if task == 'main' or task == 'all' \
            or task == 'description':
        pred_dict, ref_dict = prepare_bleu(pred_result_filtered,
                                           ref_result_filtered, task)
        metrics = compute_bleu_rouge(pred_dict, ref_dict)
    elif task == 'yesno':
        pred_dict, ref_dict = prepare_bleu(pred_result_filtered,
                                           ref_result_filtered, task)
        keys = ['Yes', 'No', 'Depends']
        preds = [filter_dict(pred_dict, k) for k in keys]
        refs = [filter_dict(ref_dict, k) for k in keys]

        metrics = compute_bleu_rouge(pred_dict, ref_dict)

        for k, pred, ref in zip(keys, preds, refs):
            m = compute_bleu_rouge(pred, ref)
            k_metric = [(k + '|' + key, v) for key, v in m.items()]
            metrics.update(k_metric)

    elif task == 'entity':
        pred_dict, ref_dict = prepare_prf(pred_result_filtered,
                                          ref_result_filtered)
        pred_dict_bleu, ref_dict_bleu = prepare_bleu(pred_result_filtered,
                                                     ref_result_filtered, task)
        metrics = compute_prf(pred_dict, ref_dict)
        metrics.update(compute_bleu_rouge(pred_dict_bleu, ref_dict_bleu))
    else:
        raise ValueError('Illegal task name: {}'.format(task))

    return metrics


def prepare_bleu(pred_result, ref_result, task):
    """
    Prepares data for calculation of bleu and rouge scores.
    """
    pred_list, ref_list = [], []
    qids = ref_result.keys()
    for qid in qids:
        if task == 'main':
            pred, ref = get_main_result(qid, pred_result, ref_result)
        elif task == 'yesno':
            pred, ref = get_yesno_result(qid, pred_result, ref_result)
        elif task == 'all':
            pred, ref = get_all_result(qid, pred_result, ref_result)
        elif task == 'entity':
            pred, ref = get_entity_result(qid, pred_result, ref_result)
        elif task == 'description':
            pred, ref = get_desc_result(qid, pred_result, ref_result)
        else:
            raise ValueError('Illegal task name: {}'.format(task))
        if pred and ref:
            pred_list += pred
            ref_list += ref
    pred_dict = dict(pred_list)
    ref_dict = dict(ref_list)
    for qid, ans in ref_dict.items():
        ref_dict[qid] = normalize(ref_dict[qid])
        pred_dict[qid] = normalize(pred_dict.get(qid, [EMPTY]))
        if not ans or ans == [EMPTY]:
            del ref_dict[qid]
            del pred_dict[qid]

    for k, v in pred_dict.items():
        assert len(v) == 1, \
            'There should be only one predict answer. question_id: {}'.format(k)
    return pred_dict, ref_dict


def get_main_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'main'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    ref_ans = ref_result[qid]['answers']
    if not ref_ans:
        ref_ans = [EMPTY]
    pred_ans = pred_result.get(qid, {}).get('answers', [])[:1]
    if not pred_ans:
        pred_ans = [EMPTY]

    return [(qid, pred_ans)], [(qid, ref_ans)]


def get_entity_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'entity'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] != 'ENTITY':
        return None, None
    return get_main_result(qid, pred_result, ref_result)


def get_desc_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'description'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] != 'DESCRIPTION':
        return None, None
    return get_main_result(qid, pred_result, ref_result)


def get_yesno_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'yesno'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """

    def _uniq(li, is_ref):
        uniq_li = []
        left = []
        keys = set()
        for k, v in li:
            if k not in keys:
                uniq_li.append((k, v))
                keys.add(k)
            else:
                left.append((k, v))

        if is_ref:
            dict_li = dict(uniq_li)
            for k, v in left:
                dict_li[k] += v
            uniq_li = [(k, v) for k, v in dict_li.items()]
        return uniq_li

    def _expand_result(uniq_li):
        expanded = uniq_li[:]
        keys = set([x[0] for x in uniq_li])
        for k in YESNO_LABELS - keys:
            expanded.append((k, [EMPTY]))
        return expanded

    def _get_yesno_ans(qid, result_dict, is_ref=False):
        if qid not in result_dict:
            return [(str(qid) + '_' + k, v) for k, v in _expand_result([])]
        yesno_answers = result_dict[qid]['yesno_answers']
        answers = result_dict[qid]['answers']
        lbl_ans = _uniq([(k, [v]) for k, v in zip(yesno_answers, answers)],
                        is_ref)
        ret = [(str(qid) + '_' + k, v) for k, v in _expand_result(lbl_ans)]
        return ret

    if ref_result[qid]['question_type'] != 'YES_NO':
        return None, None

    ref_ans = _get_yesno_ans(qid, ref_result, is_ref=True)
    pred_ans = _get_yesno_ans(qid, pred_result)
    return pred_ans, ref_ans


def get_all_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'all'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] == 'YES_NO':
        return get_yesno_result(qid, pred_result, ref_result)
    return get_main_result(qid, pred_result, ref_result)


def format_metrics(metrics, task, err_msg):
    """
    Format metrics. 'err' field returns any error occured during evaluation.

    Args:
        metrics: A dict object contains metrics for different tasks.
        task: Task name.
        err_msg: Exception raised during evaluation.
    Returns:
        Formatted result.
    """
    result = {}
    sources = ['both', 'search', 'zhidao']
    if err_msg is not None:
        return {'errorMsg': str(err_msg), 'errorCode': 1, 'data': []}
    data = []
    if task != 'all' and task != 'main':
        sources = ['both']

    if task == 'entity':
        metric_names = ['Bleu-4', 'Rouge-L']
        metric_names_prf = ['F1', 'Precision', 'Recall']
        for name in metric_names + metric_names_prf:
            for src in sources:
                obj = {
                    'name': name,
                    'value': round(metrics[src].get(name, 0) * 100, 2),
                    'type': src,
                }
                data.append(obj)
    elif task == 'yesno':
        metric_names = ['Bleu-4', 'Rouge-L']
        details = ['Yes', 'No', 'Depends']
        src = sources[0]
        for name in metric_names:
            obj = {
                'name': name,
                'value': round(metrics[src].get(name, 0) * 100, 2),
                'type': 'All',
            }
            data.append(obj)
            for d in details:
                obj = {
                    'name': name,
                    'value': round(metrics[src].get(d + '|' + name, 0) * 100,
                                   2),
                    'type': d
                }
                data.append(obj)
    else:
        metric_names = ['Bleu-4', 'Rouge-L']
        for name in metric_names:
            for src in sources:
                obj = {
                    'name': name,
                    'value': round(metrics[src].get(name, 0) * 100, 2),
                    'type': src
                }
                data.append(obj)

    result['data'] = data
    result['errorCode'] = 0
    result['errorMsg'] = 'success'

    return result


def main(args):
    """
    Do evaluation.
    """
    err = None
    metrics = {}
    try:
        pred_result = read_file(args.pred_file, args.task)
        ref_result = read_file(args.ref_file, args.task, is_ref=True)
        sources = ['both', 'search', 'zhidao']
        if args.task not in set(['main', 'all']):
            sources = sources[:1]
        for source in sources:
            metrics[source] = get_metrics(pred_result, ref_result, args.task,
                                          source)
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae

    print(
        json.dumps(
            format_metrics(metrics, args.task, err),
            ensure_ascii=False).encode('utf8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='predict file')
    parser.add_argument('ref_file', help='reference file')
    parser.add_argument(
        'task', help='task name: Main|Yes_No|All|Entity|Description')

    args = parser.parse_args()
    args.task = args.task.lower().replace('_', '')
    main(args)
