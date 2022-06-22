"""
Field class
"""
import os
import random
from collections import OrderedDict
from itertools import chain

import numpy as np

from modelscope.preprocessors.space.tokenizer import Tokenizer
from modelscope.utils.nlp.space import ontology, utils
from modelscope.utils.nlp.space.db_ops import MultiWozDB
from modelscope.utils.nlp.space.utils import list2np


class BPETextField(object):

    pad_token = '[PAD]'
    bos_token = '[BOS]'
    eos_token = '[EOS]'
    unk_token = '[UNK]'
    sos_u_token = '<sos_u>'
    eos_u_token = '<eos_u>'
    sos_b_token = '<sos_b>'
    eos_b_token = '<eos_b>'
    sos_d_token = '<sos_d>'
    eos_d_token = '<eos_d>'
    sos_a_token = '<sos_a>'
    eos_a_token = '<eos_a>'
    sos_db_token = '<sos_db>'
    eos_db_token = '<eos_db>'
    sos_r_token = '<sos_r>'
    eos_r_token = '<eos_r>'

    @property
    def bot_id(self):
        """
        用于区分user和bot两个角色
        1和0不是词表中的index，而是专门针对role的index，大小就为2，对应超参数'num_type_embeddings'
        """
        return 0

    @property
    def user_id(self):
        """
        用于区分user和bot两个角色
        1和0不是词表中的index，而是专门针对role的index，大小就为2，对应超参数'num_type_embeddings'
        """
        return 1

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
    def sos_d_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_d_token])[0]

    @property
    def eos_d_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_d_token])[0]

    def __init__(self, config):
        self.gpu = 0
        self.tokenizer = None
        self.vocab = None
        self.db = None
        self.set_stats = {}

        self.prompt_num_for_understand = config.BPETextField.prompt_num_for_understand
        self.prompt_num_for_policy = config.BPETextField.prompt_num_for_policy
        self.understand_tokens = ontology.get_understand_tokens(
            self.prompt_num_for_understand)
        self.policy_tokens = ontology.get_policy_tokens(
            self.prompt_num_for_policy)

        self.with_query_bow = config.BPETextField.with_query_bow
        self.understand = config.BPETextField.understand
        self.policy = config.BPETextField.policy

        self.batch_size = config.Trainer.batch_size
        self.filtered = config.BPETextField.filtered
        self.max_len = config.BPETextField.max_len
        self.min_utt_len = config.BPETextField.min_utt_len
        self.max_utt_len = config.BPETextField.max_utt_len
        self.min_ctx_turn = config.BPETextField.min_ctx_turn
        self.max_ctx_turn = config.BPETextField.max_ctx_turn - 1  # subtract reply turn

        self.use_true_prev_bspn = config.Generator.use_true_prev_bspn
        self.use_true_prev_aspn = config.Generator.use_true_prev_aspn
        self.use_true_db_pointer = config.Generator.use_true_db_pointer
        self.use_true_prev_resp = config.Generator.use_true_prev_resp
        self.use_true_curr_bspn = config.Generator.use_true_curr_bspn
        self.use_true_curr_aspn = config.Generator.use_true_curr_aspn
        self.use_all_previous_context = config.Generator.use_all_previous_context
        self.use_true_bspn_for_ctr_eval = config.Generator.use_true_bspn_for_ctr_eval
        self.use_true_domain_for_ctr_eval = config.Generator.use_true_domain_for_ctr_eval

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)
        batch = {}

        src = [sp['src'][-self.max_ctx_turn:] for sp in samples]
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

        # src端序列和tgt端序列需要分开pad，以保证解码时第一个词对齐
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

        if self.understand_ids and self.understand:
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

        return batch, batch_size

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == self.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        # if (len(batch) % len(cfg.cuda_device)) != 0:
        #     batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        # TODO deal with deleted data
        if self.gpu <= 1:
            if len(batch) > 0.5 * self.batch_size:
                all_batches.append(batch)
            elif len(all_batches):
                all_batches[-1].extend(batch)
            else:
                all_batches.append(batch)

        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial

    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)


class MultiWOZBPETextField(BPETextField):

    def __init__(self, model_dir, config):
        super(MultiWOZBPETextField, self).__init__(config)
        import spacy
        self.nlp = spacy.load('en_core_web_sm')

        self.db = MultiWozDB(
            model_dir, {
                'attraction': 'db/attraction_db_processed.json',
                'hospital': 'db/hospital_db_processed.json',
                'hotel': 'db/hotel_db_processed.json',
                'police': 'db/police_db_processed.json',
                'restaurant': 'db/restaurant_db_processed.json',
                'taxi': 'db/taxi_db_processed.json',
                'train': 'db/train_db_processed.json',
            })
        self._build_vocab(model_dir)

        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token, self.unk_token
        ]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(
            vocab_path=os.path.join(model_dir, 'vocab.txt'),
            special_tokens=special_tokens,
            tokenizer_type=config.BPETextField.tokenizer_type)
        self.understand_ids = self.tokenizer.convert_tokens_to_ids(
            self.understand_tokens)
        self.policy_ids = self.tokenizer.convert_tokens_to_ids(
            self.policy_tokens)

        return

    def get_ids(self, data: str):
        result = [self.sos_u_id] + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(
                self._get_convert_str(data))) + [self.eos_u_id]
        return result

    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key == 'dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][
                            -1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_batches(self, set_name):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            try:
                log_str += 'turn num:%d, dial num: %d, batch num: %d last batch len: %d\n' % (
                    k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            except Exception:
                log_str += 'turn num:%d, dial num: %d, batch num: %d last batch len: %d\n' % (
                    k, len(turn_bucket[k]), len(batches), 0.0)
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches

        # log stats
        # logging.info(log_str)
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name][
            'num_training_steps_per_epoch'] = num_training_steps  # turn-level的steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        if set_name == 'train':
            random.shuffle(all_batches)
        return all_batches

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        prompt_tokens = self.understand_tokens + self.policy_tokens
        special_tokens.extend(
            ontology.get_special_tokens(other_tokens=prompt_tokens))

        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in self.vocab._word2idx.keys():
            if word.startswith('[value_') and word.endswith(']'):
                special_tokens.append(word)

        return special_tokens

    def _build_vocab(self, model_dir: str):
        self.vocab = utils.MultiWOZVocab(3000)
        vp = os.path.join('{}/vocab'.format(model_dir))
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size

    def _get_convert_str(self, sent):
        assert isinstance(sent, str)
        return ' '.join([
            self.tokenizer.spec_convert_dict.get(tok, tok)
            for tok in sent.split()
        ])

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        """
        ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel'] -> {'hotel': {'pricerange': 'cheap', 'type': 'hotel'}}
        """
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx + 1]
                        ns = self.vocab.decode(ns) if type(
                            ns) is not str else ns
                        if ns == "'s":
                            continue
                    except Exception:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def convert_batch_turn(self, turn_batch, pv_batch, first_turn=False):
        """
        URURU：这里的含义是指轮级别的训练（数据整理），区别于session级别的训练方式（convert_batch_session）；
        但不同于eval时的含义，eval时二者都是逐轮依次生成的，那时URURU的含义请见相关的函数注释；

        convert the current and the last turn
        concat [U_0,R_0,...,U_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [user, bspn, db, aspn, resp]

        """
        inputs = []
        if first_turn:
            batch_zipped = zip(turn_batch['user'], turn_batch['bspn'],
                               turn_batch['db'], turn_batch['aspn'],
                               turn_batch['resp'])
            for u, b, db, a, r in batch_zipped:
                if self.use_true_curr_bspn:
                    src = [u + b + db]
                    tgt = a + r
                else:
                    src = [u]
                    tgt = b + db + a + r
                inputs.append({'src': src, 'tgt': tgt})
                pv = [src[-1], tgt]
                pv_batch.append(pv)
        else:
            batch_zipped = zip(pv_batch, turn_batch['user'],
                               turn_batch['bspn'], turn_batch['db'],
                               turn_batch['aspn'], turn_batch['resp'])
            for i, (pv, u, b, db, a, r) in enumerate(batch_zipped):
                if self.use_true_curr_bspn:
                    src = pv + [u + b + db]
                    tgt = a + r
                else:
                    src = pv + [u]
                    tgt = b + db + a + r
                inputs.append({'src': src, 'tgt': tgt})
                pv = [src[-1], tgt]
                pv_batch[i].extend(pv)

        return inputs, pv_batch

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = [
            'dial_id', 'turn_num', 'user', 'bspn_gen', 'bsdx', 'resp_gen',
            'resp', 'aspn_gen', 'aspn', 'dspn_gen', 'dspn', 'bspn', 'pointer',
            'qspn_gen', 'qspn'
        ]

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'trun_num': len(turns)}
            for f in field[2:]:
                entry[f] = ''  # TODO ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        v = ' '.join(v)
                    else:
                        pass  # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
        firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        prompt_id = None
        if self.use_true_curr_bspn:
            if self.use_true_curr_aspn:  # only predict resp
                context_list = ['user', 'bspn', 'db', 'aspn']
                prompt_id = self.sos_r_id
            else:  # predicted aspn
                context_list = ['user', 'bspn', 'db']
                prompt_id = self.sos_a_id
        else:  # predict bspn aspn resp. db are not predicted. this part tbd.
            context_list = ['user']
            prompt_id = self.sos_b_id

        if first_turn:
            context = []
            for c in context_list:
                context += turn[c]

            inputs['src'] = [context]
            inputs['labels'] = [context]
        else:
            context = []
            for c in context_list:
                context += turn[c]

            if self.use_true_curr_bspn:
                pv_context = pv_turn['labels'] + [
                    pv_turn['aspn'] + pv_turn['resp']
                ]
            else:
                pv_info = pv_turn['bspn'] + pv_turn['db'] + pv_turn[
                    'aspn'] + pv_turn['resp']
                pv_context = pv_turn['labels'] + [pv_info]

            # prompt response, add sos_r
            inputs['src'] = pv_context + [context]

            if self.use_all_previous_context:
                inputs['labels'] = pv_context + [
                    context
                ]  # use all previous ubar history
            else:
                inputs['labels'] = [context]  # use previous turn

        return inputs, prompt_id
