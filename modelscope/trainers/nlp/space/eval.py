# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright from https://github.com/thu-spmi/LABES
# Copyright from https://github.com/TonyNemo/UBAR-MultiWOZ
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
import math
from collections import Counter

import json
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import f1_score

from modelscope.utils.nlp.space import ontology, utils
from modelscope.utils.nlp.space.clean_dataset import clean_slot_values


def similar(a, b):
    return a == b or a in b or b in a or a.split()[0] == b.split(
    )[0] or a.split()[-1] == b.split()[-1]


def setsub(a, b):
    junks_a = []
    useless_constraint = [
        'temperature', 'week', 'est ', 'quick', 'reminder', 'near'
    ]
    for i in a:
        flg = False
        for j in b:
            if similar(i, j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True


def setsim(a, b):
    a, b = set(a), set(b)
    return setsub(a, b) and setsub(b, a)


def DA_evaluate(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    results = {}

    for avg_name in ['micro']:
        my_f1_score = f1_score(y_true=labels, y_pred=preds, average=avg_name)
        results['f1_{}'.format(avg_name)] = my_f1_score

    return results


class BLEUScorer(object):
    # BLEU score calculator via GentScorer interface
    # it calculates the BLEU-4 by taking the entire corpus in
    # Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = \
                        dict((ng, min(count, max_counts[ng])) for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = \
            1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = \
            [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
        s = \
            math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


""""
For the data preparation and evaluation on MultiWOZ2.0/2.1,
we refer to the code of UBAR (https://github.com/TonyNemo/UBAR-MultiWOZ)
"""


class MultiWOZEvaluator(object):

    def __init__(self, reader, **kwargs):
        self.reader = reader
        self.domains = ontology.all_domains
        self.all_data = self.reader.data
        self.test_data = self.reader.test

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for d, s_list in ontology.informable_slots.items():
            for s in s_list:
                self.all_info_slot.append(d + '-' + s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        self.db_dir = kwargs['data_dir']

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def validation_metric(self, data, fout=None):
        bleu = self.bleu_metric(data)
        # accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
        success, match, req_offer_counts, dial_num = \
            self.context_to_response_eval(data, same_eval_as_cambridge=True, fout=fout)
        return bleu, success, match

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [], []
        for row in data:
            if eval_dial_list and row[
                    'dial_id'] + '.json' not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            try:
                sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            except Exception:
                sc = 0.0
        else:
            sc = 0.0
        return sc

    def context_to_response_eval(self,
                                 data,
                                 eval_dial_list=None,
                                 same_eval_as_cambridge=False,
                                 fout=None):
        dials = self.pack_dial(data)
        counts = {}
        for req in self.requestables:
            counts[req + '_total'] = 0
            counts[req + '_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id + '.json' not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}
            if '.json' not in dial_id and '.json' in list(
                    self.all_data.keys())[0]:
                dial_id = dial_id + '.json'
            for domain in ontology.all_domains:
                if self.all_data[dial_id]['goal'].get(domain):
                    true_goal = self.all_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)

            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            success, match, stats, counts = \
                self._evaluateGeneratedDialogue(dial, goal, reqs, counts,
                                                same_eval_as_cambridge=same_eval_as_cambridge, fout=fout)

            successes += success
            matches += match
            dial_num += 1

        succ_rate = successes / (float(dial_num) + 1e-10) * 100
        match_rate = matches / (float(dial_num) + 1e-10) * 100
        return succ_rate, match_rate, counts, dial_num

    def _evaluateGeneratedDialogue(self,
                                   dialog,
                                   goal,
                                   real_requestables,
                                   counts,
                                   soft_acc=False,
                                   same_eval_as_cambridge=False,
                                   fout=None):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        log = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0:
                continue
            if fout is not None:
                log.append({
                    'turn_num': turn['turn_num'],
                    'turn_domain': turn['dspn'],
                    'user': turn['user'],
                    'aspn': turn['aspn'],
                    'aspn_gen': turn['aspn_gen'],
                    'resp': turn['resp'],
                    'resp_gen': turn['resp_gen'],
                    'pointer': turn['pointer'],
                })

            sent_t = turn['resp_gen']

            for domain in goal.keys():
                # for computing success
                if same_eval_as_cambridge:
                    # [restaurant_name], [hotel_name] instead of [value_name]
                    if self.reader.use_true_domain_for_ctr_eval:
                        dom_pred = [d[1:-1] for d in turn['dspn'].split()]
                    else:
                        dom_pred = [d[1:-1] for d in turn['dspn_gen'].split()]

                    if domain not in dom_pred:  # fail
                        continue
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in [
                            'restaurant', 'hotel', 'attraction', 'train'
                    ]:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if not self.reader.use_true_curr_bspn and not self.reader.use_true_bspn_for_ctr_eval:
                            bspn = turn['bspn_gen']
                        else:
                            bspn = turn['bspn']

                        constraint_dict = self.reader.bspan_to_constraint_dict(
                            bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(
                                domain,
                                constraint_dict[domain],
                                return_name=True)
                        else:
                            venues = []

                        if len(venue_offered[domain]) == 0 and venues:

                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            flag = False
                            for ven in venues:
                                if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            if flag and venues:  # sometimes there are no results so sample won't work
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            if domain in ['restaurant', 'hotel', 'train']:
                                if 'booked' in turn['pointer'] or 'ok' in turn[
                                        'pointer'] or '[value_reference]' in turn[
                                            'resp']:
                                    # if pointer was allowing for that?
                                    provided_requestables[domain].append(
                                        'reference')
                            else:
                                provided_requestables[domain].append(
                                    'reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain][
                        'requestable']:
                    venue_offered[domain] = '[value_name]'
        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {
            'restaurant': [0, 0, 0],
            'hotel': [0, 0, 0],
            'attraction': [0, 0, 0],
            'train': [0, 0, 0],
            'taxi': [0, 0, 0],
            'hospital': [0, 0, 0],
            'police': [0, 0, 0]
        }

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(
                    domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]
                        ) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and len(
                        set(venue_offered[domain]) & set(goal_venues)) > 0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match) / len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request + '_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request + '_offer'] += 1

        # SUCCESS
        if fout is not None:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success) / len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0
        else:
            if match == 1.0:
                for domain in domains_in_goal:
                    success_stat = 0
                    domain_success = 0
                    if len(real_requestables[domain]) == 0:
                        success += 1
                        success_stat = 1
                        stats[domain][1] = success_stat
                        continue
                    # if values in sentences are super set of requestables
                    for request in real_requestables[domain]:
                        if request in provided_requestables[domain]:
                            domain_success += 1

                    if domain_success == len(real_requestables[domain]):
                        success += 1
                        success_stat = 1

                    stats[domain][1] = success_stat

                # final eval
                if soft_acc:
                    success = float(success) / len(real_requestables)
                else:
                    if success >= len(real_requestables):
                        success = 1
                    else:
                        success = 0

        if fout is not None and success == 0:
            sample = {
                dialog[0]['dial_id']: {
                    'log': log,
                    'real_requestables': real_requestables,
                    'provided_requestables': provided_requestables
                }
            }
            line = json.dumps(sample)
            fout.write(line)
            fout.write('\n')

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for s in true_goal[domain]['reqt']:  # addtional requests:
                        if s in [
                                'phone', 'address', 'postcode', 'reference',
                                'id'
                        ]:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')

            for s, v in true_goal[domain]['info'].items():
                s_, v_ = clean_slot_values(self.db_dir, domain, s, v)
                if len(v_.split()) > 1:
                    v_ = ' '.join(
                        [token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]['informable'][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]['booking'] = true_goal[domain]['book']
        return goal


class GenericEvaluator:

    def __init__(self, reader):
        self.reader = reader
        self.metric_dict = {}

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def run_metrics(self, results):
        raise ValueError('Please specify the evaluator first')

    def bleu_metric(self, data, type='bleu'):
        gen, truth = [], []
        for row in data:
            gen.append(self.clean(row['resp_gen']))
            # gen.append(self.clean(row['resp']))
            truth.append(self.clean(row['resp']))
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
        return sc

    def _normalize_constraint(self,
                              constraint,
                              ignore_dontcare=False,
                              intersection=True):
        """
        Normalize belief span, e.g. delete repeated words
        :param constraint - {'food': 'asian oritental', 'pricerange': 'cheap'}
        :param intersection: if true, only keeps the words that appear in th ontology
                                        we set intersection=True as in previous works
        :returns: normalized constraint dict
                      e.g. - {'food': 'asian oritental', 'pricerange': 'cheap', 'area': ''}
        """
        normalized = {}
        for s in self.informable_slots:
            normalized[s] = ''
        for s, v in constraint.items():
            if ignore_dontcare and v == 'dontcare':
                continue
            if intersection and v != 'dontcare' and v not in self.entities_flat:
                continue

            normalized[s] = v

        return normalized

    def _normalize_act(self, aspn, intersection=False):
        aspn_list = aspn.split('|')
        normalized = {}
        for i, v in enumerate(aspn_list):
            seq = v.strip()
            word_set = set()
            for w in seq.split():
                if intersection:
                    if self.reader.act_order[i] == 'av':
                        if '[value' in w:
                            word_set.add(w)
                    else:
                        if w in self.requestable_slots:
                            word_set.add(w)
                else:
                    word_set.add(w)
            normalized[self.reader.act_order[i]] = word_set
        return normalized

    def tracker_metric(self, data, normalize=True):
        # turn level metric
        tp, fp, fn, db_correct = 0, 0, 0, 0
        goal_accr, slot_accr, total = 0, {}, 1e-8
        for s in self.informable_slots:
            slot_accr[s] = 0

        for row in data:
            if normalize:
                gen = self._normalize_constraint(row['bspn_gen'])
                truth = self._normalize_constraint(row['bspn'])
            else:
                gen = self._normalize_constraint(
                    row['bspn_gen'], intersection=False)
                truth = self._normalize_constraint(
                    row['bspn'], intersection=False)
            valid = 'thank' not in row['user'] and 'bye' not in row['user']
            if valid:
                for slot, value in gen.items():
                    if value in truth[slot]:
                        tp += 1
                    else:
                        fp += 1
                for slot, value in truth.items():
                    if value not in gen[slot]:
                        fn += 1

            if truth and valid:
                total += 1
                for s in self.informable_slots:
                    if gen[s] == truth[s]:
                        slot_accr[s] += 1
                if gen == truth:
                    goal_accr += 1
                if row.get('db_gen') and row.get('db_match'):
                    if row['db_gen'] == row['db_match']:
                        db_correct += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        goal_accr /= total
        db_correct /= total
        for s in slot_accr:
            slot_accr[s] /= total
        return precision, recall, f1, goal_accr, slot_accr, db_correct

    def request_metric(self, data):
        # dialog level metric
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            truth_req, gen_req = set(), set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                resp_gen_token = self.clean(turn['resp_gen']).split()
                resp_token = self.clean(turn['resp']).split()
                for w in resp_gen_token:
                    if '[value_' in w and w.endswith(
                            ']') and w != '[value_name]':
                        gen_req.add(w[1:-1].split('_')[1])
                for w in resp_token:
                    if '[value_' in w and w.endswith(
                            ']') and w != '[value_name]':
                        truth_req.add(w[1:-1].split('_')[1])
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

    def act_metric(self, data):
        # turn level metric
        tp, fp, fn = {
            'all_s': 0,
            'all_v': 0
        }, {
            'all_s': 0,
            'all_v': 0
        }, {
            'all_s': 0,
            'all_v': 0
        }
        for s in self.requestable_slots:
            tp[s], fp[s], fn[s] = 0, 0, 0
            tp['[value_%s]' % s], fp['[value_%s]' % s], fn['[value_%s]'
                                                           % s] = 0, 0, 0

        for row in data:
            gen = self._normalize_act(row['aspn_gen'])
            truth = self._normalize_act(row['aspn'])
            valid = 'thank' not in row['user'] and 'bye' not in row['user']
            if valid:
                # how well the act decoder captures user's requests
                for value in gen['av']:
                    if value in truth['av']:
                        tp['all_v'] += 1
                        if tp.get(value):
                            tp[value] += 1
                    else:
                        fp['all_v'] += 1
                        if fp.get(value):
                            fp[value] += 1
                for value in truth['av']:
                    if value not in gen['av']:
                        fn['all_v'] += 1
                        if fn.get(value):
                            fn[value] += 1

                # how accurately the act decoder predicts system's question
                if 'as' not in gen:
                    continue
                for slot in gen['as']:
                    if slot in truth['as']:
                        tp['all_s'] += 1
                        if tp.get(slot):
                            tp[slot] += 1
                    else:
                        fp['all_s'] += 1
                        if fp.get(slot):
                            fp[slot] += 1
                for slot in truth['as']:
                    if slot not in gen['as']:
                        fn['all_s'] += 1
                        if fn.get(slot):
                            fn[slot] += 1

        result = {}
        for k, v in tp.items():
            precision, recall = tp[k] / (tp[k] + fp[k] + 1e-8), tp[k] / (
                tp[k] + fn[k] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            result[k] = [f1, precision, recall]
        return result


"""
For the data preparation and evaluation on In-Car Assistant/CamRest,
we refer to the code of LABES (https://github.com/thu-spmi/LABES)
"""


class CamRestEvaluator(GenericEvaluator):

    def __init__(self, reader):
        super().__init__(reader)
        self.entities_flat, self.entitiy_to_slot_dict = self.get_entities(
            self.reader.ontology_path)
        self.informable_slots = self.reader.otlg.informable_slots
        self.requestable_slots = self.reader.otlg.requestable_slots

    def run_metrics(self, results):
        metrics = {}
        bleu = self.bleu_metric(results)
        p, r, f1, goal_acc, slot_acc, db_acc = self.tracker_metric(results)
        match = self.match_metric(results)
        req_f1, req_p, req_r = self.request_metric(results)

        metrics['bleu'] = bleu
        metrics['match'] = match
        metrics['req_f1'] = req_f1
        metrics['joint_goal'] = goal_acc
        metrics['slot_accu'] = slot_acc
        metrics['slot-p/r/f1'] = (p, r, f1)
        metrics['db_acc'] = db_acc

        return metrics

    def get_entities(self, entity_path):
        entities_flat = []
        entitiy_to_slot_dict = {}
        raw_entities = json.loads(
            open(entity_path, encoding='utf-8').read().lower())
        for s in raw_entities['informable']:
            entities_flat.extend(raw_entities['informable'][s])
            for v in raw_entities['informable'][s]:
                entitiy_to_slot_dict[v] = s
        return entities_flat, entitiy_to_slot_dict

    def constraint_same(self, truth_cons, gen_cons):
        if not truth_cons and not gen_cons:
            return True
        if not truth_cons or not gen_cons:
            return False
        return setsim(gen_cons, truth_cons)

    def match_metric(self, data):
        dials = self.pack_dial(data)
        match, total = 0, 1e-8
        for dial_id in dials:
            dial = dials[dial_id]
            truth_cons, gen_cons = {'1': '', '2': '', '3': ''}, None
            for turn_num, turn in enumerate(dial):
                # find the last turn which the system provide an entity
                if '[value' in turn['resp_gen']:
                    gen_cons = self._normalize_constraint(
                        turn['bspn_gen'], ignore_dontcare=True)
                if '[value' in turn['resp']:
                    truth_cons = self._normalize_constraint(
                        turn['bspn'], ignore_dontcare=True)
            if not gen_cons:
                # if no entity is provided, choose the state of the last dialog turn
                gen_cons = self._normalize_constraint(
                    dial[-1]['bspn_gen'], ignore_dontcare=True)
            if list(truth_cons.values()) != ['', '', '']:
                if gen_cons == truth_cons:
                    match += 1
                total += 1

        return match / total

    def clean(self, resp):
        # we  use the same clean process as in Sequicity, SEDST, FSDM
        # to ensure comparable results
        resp = resp.replace(f'{self.reader.sos_r_token} ', '')
        resp = resp.replace(f' {self.reader.eos_r_token}', '')
        resp = f'{self.reader.sos_r_token} {resp} {self.reader.eos_r_token}'
        for value, slot in self.entitiy_to_slot_dict.items():

            resp = utils.clean_replace(resp, value, '[value_%s]' % slot)
        return resp


class KvretEvaluator(GenericEvaluator):

    def __init__(self, reader):
        super().__init__(reader)
        self.entities_flat, self.entitiy_to_slot_dict = self.get_entities(
            self.reader.ontology_path)
        self.informable_slots = self.reader.otlg.informable_slots
        self.requestable_slots = self.reader.otlg.requestable_slots

    def run_metrics(self, results):
        metrics = {}
        bleu = self.bleu_metric(results)
        p, r, f1, goal_acc, slot_acc, db_acc = self.tracker_metric(
            results, normalize=True)
        match = self.match_metric(results)
        req_f1, req_p, req_r = self.request_metric(results)

        metrics['bleu'] = bleu
        metrics['match'] = match
        metrics['req_f1'] = req_f1
        metrics['joint_goal'] = goal_acc
        metrics['slot_accu'] = slot_acc
        metrics['slot-p/r/f1'] = (p, r, f1)
        metrics['db_acc'] = db_acc

        return metrics

    def _normalize_constraint(self,
                              constraint,
                              ignore_dontcare=False,
                              intersection=True):
        """
        Normalize belief span, e.g. delete repeated words
        :param constraint - {'food': 'asian oritental', 'pricerange': 'cheap'}
        :param intersection: if true, only keeps the words that appear in th ontology
                                        we set intersection=True as in previous works
        :returns: normalized constraint dict
                      e.g. - {'food': 'asian oritental', 'pricerange': 'cheap', 'area': ''}
        """
        junk = [
            'good', 'great', 'quickest', 'shortest', 'route', 'week',
            'fastest', 'nearest', 'next', 'closest', 'way', 'mile', 'activity',
            'restaurant', 'appointment'
        ]
        normalized = {}
        for s in self.informable_slots:
            normalized[s] = ''
        for s, v in constraint.items():
            for j in junk:
                v = ' '.join(v.replace(j, '').split())
            if intersection and v not in self.entities_flat:
                continue

            if s in self.informable_slots:
                normalized[s] = v
            else:
                # TODO only use slot (not domain) in s for matching !!!
                pass

        return normalized

    def get_entities(self, entity_path):
        entities_flat = []
        entitiy_to_slot_dict = {}

        entitiy_to_slot_dict = self.reader.entity_dict
        for s in entitiy_to_slot_dict:
            if s not in entities_flat:
                entities_flat.append(s)
        return entities_flat, entitiy_to_slot_dict

    def constraint_same(self, truth_cons, gen_cons):
        if not truth_cons and not gen_cons:
            return True
        if not truth_cons or not gen_cons:
            return False
        return setsim(gen_cons, truth_cons)

    def match_metric(self, data):
        dials = self.pack_dial(data)
        match, total = 0, 1e-8
        for dial_id in dials:
            dial = dials[dial_id]
            truth_cons, gen_cons = {
                '1': '',
                '2': '',
                '3': '',
                '4': '',
                '5': '',
                '6': '',
                '7': '',
                '8': '',
                '9': '',
                '10': '',
                '11': ''
            }, None
            for turn_num, turn in enumerate(dial):
                # find the last turn which the system provide an entity
                if '[value' in turn['resp_gen']:
                    gen_cons = self._normalize_constraint(
                        turn['bspn_gen'], ignore_dontcare=True)
                if '[value' in turn['resp']:
                    truth_cons = self._normalize_constraint(
                        turn['bspn'], ignore_dontcare=True)

            if not gen_cons:
                # if no entity is provided, choose the state of the last dialog turn
                gen_cons = self._normalize_constraint(
                    dial[-1]['bspn_gen'], ignore_dontcare=True)

            if list(truth_cons.values()) != [''] * 11:
                gen_cons = [x for x in gen_cons.values() if x]
                truth_cons = [x for x in truth_cons.values() if x]
                if self.constraint_same(gen_cons, truth_cons):
                    match += 1
                total += 1

        return match / total

    def clean(self, resp):
        # we  use the same clean process as in Sequicity, SEDST, FSDM
        # to ensure comparable results
        resp = resp.replace(f'{self.reader.sos_r_token} ', '')
        resp = resp.replace(f' {self.reader.eos_r_token}', '')
        resp = f'{self.reader.sos_r_token} {resp} {self.reader.eos_r_token}'
        for value, slot in self.entitiy_to_slot_dict.items():
            resp = utils.clean_replace(resp, value, '[value_%s]' % slot)
        return resp
