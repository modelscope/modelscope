# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
from asyncio import constants
from collections import OrderedDict
from itertools import chain

import json
import numpy as np

from modelscope.preprocessors.nlp.space.tokenizer import Tokenizer
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp.space import ontology, utils
from modelscope.utils.nlp.space.db_ops import MultiWozDB
from modelscope.utils.nlp.space.utils import list2np

logger = get_logger()


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
        return 0

    @property
    def user_id(self):
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
        self.train, self.dev, self.test = [], [], []
        self.gpu = config.Trainer.gpu
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

        # src sequence and tgt sequence should be padded separately，to make sure the first word is aligned
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
                all_batches.append(batch)
                batch = []

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

    def __init__(self, config, **kwargs):
        super(MultiWOZBPETextField, self).__init__(config)

        import spacy
        try:
            import en_core_web_sm
        except ImportError:
            logger.warning('Miss module en_core_web_sm!')
            logger.warning('We will download en_core_web_sm automatically.')
            try:
                spacy.cli.download('en_core_web_sm')
            except Exception as e:
                logger.error(e)
                raise ImportError(
                    'Download en_core_web_sm error. '
                    'Please use \'python -m spacy download en_core_web_sm\' to download it by yourself!'
                )
        self.nlp = spacy.load('en_core_web_sm')

        if config.do_train:
            db_dir = kwargs['data_dir']
        else:
            db_dir = kwargs['model_dir']
        self.db = MultiWozDB(
            db_dir, {
                'attraction': 'db/attraction_db_processed.json',
                'hospital': 'db/hospital_db_processed.json',
                'hotel': 'db/hotel_db_processed.json',
                'police': 'db/police_db_processed.json',
                'restaurant': 'db/restaurant_db_processed.json',
                'taxi': 'db/taxi_db_processed.json',
                'train': 'db/train_db_processed.json',
            })
        self._build_vocab(db_dir)

        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token, self.unk_token
        ]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(
            vocab_path=os.path.join(kwargs['model_dir'], ModelFile.VOCAB_FILE),
            special_tokens=special_tokens,
            tokenizer_type=config.BPETextField.tokenizer_type)
        self.understand_ids = self.tokenizer.convert_tokens_to_ids(
            self.understand_tokens)
        self.policy_ids = self.tokenizer.convert_tokens_to_ids(
            self.policy_tokens)

        if config.do_train:
            test_list = [
                line.strip().lower() for line in open(
                    os.path.join(kwargs['data_dir'], 'testListFile.json'),
                    'r',
                    encoding='utf-8').readlines()
            ]
            dev_list = [
                line.strip().lower() for line in open(
                    os.path.join(kwargs['data_dir'], 'valListFile.json'),
                    'r',
                    encoding='utf-8').readlines()
            ]

            self.dev_files, self.test_files = {}, {}
            for fn in test_list:
                self.test_files[fn.replace('.json', '')] = 1
            for fn in dev_list:
                self.dev_files[fn.replace('.json', '')] = 1

            self._load_data(kwargs['data_dir'])

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

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)

        self.set_stats[set_name][
            'num_training_steps_per_epoch'] = num_training_steps  # turn-level steps
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

    def _load_data(self, data_dir, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """

        def load_data_from_resource(data_resource):
            data = json.loads(
                open(
                    os.path.join(data_dir, data_resource),
                    'r',
                    encoding='utf-8').read().lower())
            train, dev, test = [], [], []
            for fn, dial in data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if self.dev_files.get(fn):
                    dev.append(self._get_encoded_data(fn, dial))
                elif self.test_files.get(fn):
                    test.append(self._get_encoded_data(fn, dial))
                else:
                    train.append(self._get_encoded_data(fn, dial))
            return train, dev, test

        data_processed = 'new_db_se_blank_encoded_domain.data.json'
        data_resource = 'data_for_damd.json'
        if save_temp:  # save encoded data
            # encoded: no sos, se_encoded: sos and eos
            encoded_file = os.path.join(data_dir, data_processed)

            if os.path.exists(encoded_file):
                logger.info(
                    'Reading encoded data from {}'.format(encoded_file))
                self.data = json.loads(
                    open(
                        os.path.join(data_dir, data_resource),
                        'r',
                        encoding='utf-8').read().lower())
                encoded_data = json.loads(
                    open(encoded_file, 'r', encoding='utf-8').read())
                self.train = encoded_data['train']
                self.dev = encoded_data['dev']
                self.test = encoded_data['test']
            else:
                logger.info(
                    'Encoding data now and save the encoded data in {}'.format(
                        encoded_file))
                # not exists, encode data and save
                self.train, self.dev, self.test = load_data_from_resource(
                    data_resource)
                # save encoded data
                encoded_data = {
                    'train': self.train,
                    'dev': self.dev,
                    'test': self.test
                }
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
        else:  # directly read processed data and encode
            self.train, self.dev, self.test = load_data_from_resource(
                data_resource)

        random.seed(10)
        random.shuffle(self.train)
        logger.info('train size:{}, dev size:{}, test size:{}'.format(
            len(self.train), len(self.dev), len(self.test)))

    def _get_convert_str(self, sent):
        assert isinstance(sent, str)
        return ' '.join([
            self.tokenizer.spec_convert_dict.get(tok, tok)
            for tok in sent.split()
        ])

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn

            enc_info_list = [
                ('user', self.sos_u_id, 'user', self.eos_u_id),
                ('usdx', self.sos_u_id, 'user', self.eos_u_id),
                ('resp', self.sos_r_id, 'resp', self.eos_r_id),
                ('bspn', self.sos_b_id, 'constraint', self.eos_b_id),
                ('bsdx', self.sos_b_id, 'cons_delex', self.eos_b_id),
                ('aspn', self.sos_a_id, 'sys_act', self.eos_a_id)
            ]
            for enc_key, start_token, item_key, end_token in enc_info_list:
                enc[enc_key] = [
                    start_token
                ] + self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(
                        self._get_convert_str(t[item_key]))) + [end_token]

            enc['turn_num'] = t['turn_num']

            if idx > 0 and t['turn_domain'] == '[general]':
                enc['dspn'] = encoded_dial[idx - 1]['dspn']
                enc['pointer'] = encoded_dial[idx - 1]['pointer'][:4] + [
                    int(i) for i in t['pointer'].split(',')
                ][-2:]
                enc['turn_domain'] = encoded_dial[idx - 1]['turn_domain']
                enc['db'] = encoded_dial[idx - 1]['db']
            else:
                if t['turn_domain'] == '[general]':
                    assert not t['constraint'], f'{fn}-{idx}'
                enc['dspn'] = [
                    self.sos_d_id
                ] + self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(
                        self._get_convert_str(
                            t['turn_domain']))) + [self.eos_d_id]
                enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
                enc['turn_domain'] = t['turn_domain'].split()
                db_pointer = self.bspan_to_DBpointer(t['constraint'],
                                                     t['turn_domain'].split())
                enc['db'] = [
                    self.sos_db_id
                ] + self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(
                        self._get_convert_str(db_pointer))) + [self.eos_db_id]

            encoded_dial.append(enc)
        return encoded_dial

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]

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

    def restore(self, resp, domain, constraint_dict, mat_ents):
        restored = resp

        restored = restored.replace('[value_reference]', '53022')
        restored = restored.replace('[value_car]', 'BMW')

        for d in domain:
            constraint = constraint_dict.get(d, None)
            if constraint:
                replace_res_list = [('stay', '[value_stay]'),
                                    ('day', '[value_day]'),
                                    ('people', '[value_people]'),
                                    ('time', '[value_time]'),
                                    ('type', '[value_type]')]
                for key, value_key in replace_res_list:
                    if key in constraint:
                        restored = restored.replace(value_key, constraint[key])

                if d in mat_ents and len(mat_ents[d]) == 0:
                    for s in constraint:
                        if s == 'pricerange' and d in [
                                'hotel', 'restaurant'
                        ] and 'price]' in restored:
                            restored = restored.replace(
                                '[value_price]', constraint['pricerange'])
                        if s + ']' in restored:
                            restored = restored.replace(
                                '[value_%s]' % s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]',
                                            str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', '3')

        try:
            ent = mat_ents.get(domain[-1], [])
            if ent:
                ent = ent[0]

                for t in restored.split():
                    if '[value' in t:
                        slot = t[7:-1]
                        if ent.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            restored = restored.replace(t, ent[slot])
                        elif slot == 'price':
                            if ent.get('pricerange'):
                                restored = restored.replace(
                                    t, ent['pricerange'])
                            else:
                                logger.info(restored, domain)
        except Exception:
            logger.error(resp)
            logger.error(restored)
            quit()

        restored = restored.replace('[value_phone]', '62781111')
        restored = restored.replace('[value_postcode]', 'CG9566')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')

        return restored
