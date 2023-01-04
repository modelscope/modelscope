# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict

import numpy
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from modelscope.metainfo import Models
from modelscope.models.base import Model, Tensor
from modelscope.models.builder import MODELS
from modelscope.preprocessors.nlp.space_T_cn.fields.struct import Constant
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import Seq2SQL, SpaceTCnModel
from .configuration import SpaceTCnConfig

__all__ = ['TableQuestionAnswering']


@MODELS.register_module(
    Tasks.table_question_answering, module_name=Models.space_T_cn)
class TableQuestionAnswering(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the table-question-answering model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.tokenizer = BertTokenizer(
            os.path.join(model_dir, ModelFile.VOCAB_FILE))

        state_dict = torch.load(
            os.path.join(self.model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location='cpu')

        self.backbone_config = SpaceTCnConfig.from_json_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.backbone_model = SpaceTCnModel(
            config=self.backbone_config, schema_link_module='rat')
        self.backbone_model.load_state_dict(state_dict['backbone_model'])

        constant = Constant()
        self.agg_ops = constant.agg_ops
        self.cond_ops = constant.cond_ops
        self.cond_conn_ops = constant.cond_conn_ops
        self.action_ops = constant.action_ops
        self.max_select_num = constant.max_select_num
        self.max_where_num = constant.max_where_num
        self.col_type_dict = constant.col_type_dict
        self.schema_link_dict = constant.schema_link_dict
        self.n_cond_ops = len(self.cond_ops)
        self.n_agg_ops = len(self.agg_ops)
        self.n_action_ops = len(self.action_ops)
        iS = self.backbone_config.hidden_size
        self.head_model = Seq2SQL(
            iS,
            100,
            2,
            0.0,
            self.n_cond_ops,
            self.n_agg_ops,
            self.n_action_ops,
            self.max_select_num,
            self.max_where_num,
            device=self._device_name)
        self.device = self._device_name
        self.head_model.load_state_dict(state_dict['head_model'], strict=False)

    def to(self, device):
        self.device = device
        self.backbone_model.to(device)
        self.head_model.to(device)
        self.head_model.set_device(device)

    def convert_string(self, pr_wvi, nlu, nlu_tt):
        convs = []
        for b, nlu1 in enumerate(nlu):
            conv_dict = {}
            nlu_tt1 = nlu_tt[b]
            idx = 0
            convflag = True
            for i, ntok in enumerate(nlu_tt1):
                if idx >= len(nlu1):
                    convflag = False
                    break

                if ntok.startswith('##'):
                    ntok = ntok.replace('##', '')
                tok = nlu1[idx:idx + 1].lower()
                if ntok == tok:
                    conv_dict[i] = [idx, idx + 1]
                    idx += 1
                elif ntok == '#':
                    conv_dict[i] = [idx, idx]
                elif ntok == '[UNK]':
                    conv_dict[i] = [idx, idx + 1]
                    j = i + 1
                    idx += 1
                    if idx < len(nlu1) and j < len(
                            nlu_tt1) and nlu_tt1[j] != '[UNK]':
                        while idx < len(nlu1):
                            val = nlu1[idx:idx + 1].lower()
                            if nlu_tt1[j].startswith(val):
                                break
                            idx += 1
                        conv_dict[i][1] = idx
                elif tok in ntok:
                    startid = idx
                    idx += 1
                    while idx < len(nlu1):
                        tok += nlu1[idx:idx + 1].lower()
                        if ntok == tok:
                            conv_dict[i] = [startid, idx + 1]
                            break
                        idx += 1
                    idx += 1
                else:
                    convflag = False

            conv = []
            if convflag:
                for pr_wvi1 in pr_wvi[b]:
                    s1, e1 = conv_dict[pr_wvi1[0]]
                    s2, e2 = conv_dict[pr_wvi1[1]]
                    newidx = pr_wvi1[1]
                    while newidx + 1 < len(
                            nlu_tt1) and s2 == e2 and nlu_tt1[newidx] == '#':
                        newidx += 1
                        s2, e2 = conv_dict[newidx]
                    if newidx + 1 < len(nlu_tt1) and nlu_tt1[
                            newidx + 1].startswith('##'):
                        s2, e2 = conv_dict[newidx + 1]
                    phrase = nlu1[s1:e2]
                    conv.append(phrase)
            else:
                for pr_wvi1 in pr_wvi[b]:
                    phrase = ''.join(nlu_tt1[pr_wvi1[0]:pr_wvi1[1]
                                             + 1]).replace('##', '')
                    conv.append(phrase)
            convs.append(conv)

        return convs

    def get_fields_info(self, t1s, tables, train=True):
        nlu, nlu_t, sql_i, q_know, t_know, action, hs_t, types, units, his_sql, schema_link = \
            [], [], [], [], [], [], [], [], [], [], []
        for t1 in t1s:
            nlu.append(t1['question'])
            nlu_t.append(t1['question_tok'])
            hs_t.append(t1['header_tok'])
            q_know.append(t1['bertindex_knowledge'])
            t_know.append(t1['header_knowledge'])
            types.append(t1['types'])
            units.append(t1['units'])
            his_sql.append(t1.get('history_sql', None))
            schema_link.append(t1.get('schema_link', []))
            if train:
                action.append(t1.get('action', [0]))
                sql_i.append(t1['sql'])

        return nlu, nlu_t, sql_i, q_know, t_know, action, hs_t, types, units, his_sql, schema_link

    def get_history_select_where(self, his_sql, header_len):
        if his_sql is None:
            return [0], [0]

        sel = []
        for seli in his_sql['sel']:
            if seli + 1 < header_len and seli + 1 not in sel:
                sel.append(seli + 1)

        whe = []
        for condi in his_sql['conds']:
            if condi[0] + 1 < header_len and condi[0] + 1 not in whe:
                whe.append(condi[0] + 1)

        if len(sel) == 0:
            sel.append(0)
        if len(whe) == 0:
            whe.append(0)

        sel.sort()
        whe.sort()

        return sel, whe

    def get_types_ids(self, col_type):
        for key, type_ids in self.col_type_dict.items():
            if key in col_type.lower():
                return type_ids
        return self.col_type_dict['null']

    def generate_inputs(self, nlu1_tok, hs_t_1, type_t, unit_t, his_sql,
                        q_know, t_know, s_link):
        tokens = []
        orders = []
        types = []
        segment_ids = []
        matchs = []
        col_dict = {}
        schema_tok = []

        tokens.append('[CLS]')
        orders.append(0)
        types.append(0)
        i_st_nlu = len(tokens)

        matchs.append(0)
        segment_ids.append(0)
        for idx, token in enumerate(nlu1_tok):
            if q_know[idx] == 100:
                break
            elif q_know[idx] >= 5:
                matchs.append(1)
            else:
                matchs.append(q_know[idx] + 1)
            tokens.append(token)
            orders.append(0)
            types.append(0)
            segment_ids.append(0)

        i_ed_nlu = len(tokens)

        tokens.append('[SEP]')
        orders.append(0)
        types.append(0)
        matchs.append(0)
        segment_ids.append(0)

        sel, whe = self.get_history_select_where(his_sql, len(hs_t_1))

        if len(sel) == 1 and sel[0] == 0 \
                and len(whe) == 1 and whe[0] == 0:
            pass
        else:
            tokens.append('select')
            orders.append(0)
            types.append(0)
            matchs.append(10)
            segment_ids.append(0)

            for seli in sel:
                tokens.append('[PAD]')
                orders.append(0)
                types.append(0)
                matchs.append(10)
                segment_ids.append(0)
                col_dict[len(tokens) - 1] = seli

            tokens.append('where')
            orders.append(0)
            types.append(0)
            matchs.append(10)
            segment_ids.append(0)

            for whei in whe:
                tokens.append('[PAD]')
                orders.append(0)
                types.append(0)
                matchs.append(10)
                segment_ids.append(0)
                col_dict[len(tokens) - 1] = whei

            tokens.append('[SEP]')
            orders.append(0)
            types.append(0)
            matchs.append(10)
            segment_ids.append(0)

        column_start = len(tokens)
        i_hds_f = []
        header_flatten_tokens, header_flatten_index = [], []
        for i, hds11 in enumerate(hs_t_1):
            if len(unit_t[i]) == 1 and unit_t[i][0] == 'null':
                temp_header_tokens = hds11
            else:
                temp_header_tokens = hds11 + unit_t[i]
            schema_tok.append(temp_header_tokens)
            header_flatten_tokens.extend(temp_header_tokens)
            header_flatten_index.extend([i + 1] * len(temp_header_tokens))
            i_st_hd_f = len(tokens)
            tokens += ['[PAD]']
            orders.append(0)
            types.append(self.get_types_ids(type_t[i]))
            i_ed_hd_f = len(tokens)
            col_dict[len(tokens) - 1] = i
            i_hds_f.append((i_st_hd_f, i_ed_hd_f))
            if i == 0:
                matchs.append(6)
            else:
                matchs.append(t_know[i - 1] + 6)
            segment_ids.append(1)

        tokens.append('[SEP]')
        orders.append(0)
        types.append(0)
        matchs.append(0)
        segment_ids.append(1)

        # position where
        # [SEP]
        start_ids = len(tokens) - 1

        tokens.append('action')  # action
        orders.append(1)
        types.append(0)
        matchs.append(0)
        segment_ids.append(1)

        tokens.append('connect')  # column
        orders.append(1)
        types.append(0)
        matchs.append(0)
        segment_ids.append(1)

        tokens.append('allen')  # select len
        orders.append(1)
        types.append(0)
        matchs.append(0)
        segment_ids.append(1)

        for x in range(self.max_where_num):
            tokens.append('act')  # op
            orders.append(2 + x)
            types.append(0)
            matchs.append(0)
            segment_ids.append(1)

        tokens.append('size')  # where len
        orders.append(1)
        types.append(0)
        matchs.append(0)
        segment_ids.append(1)

        for x in range(self.max_select_num):
            tokens.append('focus')  # agg
            orders.append(2 + x)
            types.append(0)
            matchs.append(0)
            segment_ids.append(1)

        i_nlu = (i_st_nlu, i_ed_nlu)

        schema_link_matrix = numpy.zeros((len(tokens), len(tokens)),
                                         dtype='int32')
        schema_link_mask = numpy.zeros((len(tokens), len(tokens)),
                                       dtype='float32')
        for relation in s_link:
            if relation['label'] in ['col', 'val']:
                [q_st, q_ed] = relation['question_index']
                cid = max(0, relation['column_index'])
                schema_link_matrix[
                    i_st_nlu + q_st: i_st_nlu + q_ed + 1,
                    column_start + cid + 1: column_start + cid + 1 + 1] = \
                    self.schema_link_dict[relation['label'] + '_middle']
                schema_link_matrix[
                    i_st_nlu + q_st,
                    column_start + cid + 1: column_start + cid + 1 + 1] = \
                    self.schema_link_dict[relation['label'] + '_start']
                schema_link_matrix[
                    i_st_nlu + q_ed,
                    column_start + cid + 1: column_start + cid + 1 + 1] = \
                    self.schema_link_dict[relation['label'] + '_end']
                schema_link_mask[i_st_nlu + q_st:i_st_nlu + q_ed + 1,
                                 column_start + cid + 1:column_start + cid + 1
                                 + 1] = 1.0

        return tokens, orders, types, segment_ids, matchs, \
            i_nlu, i_hds_f, start_ids, column_start, col_dict, schema_tok, \
            header_flatten_tokens, header_flatten_index, schema_link_matrix, schema_link_mask

    def gen_l_hpu(self, i_hds):
        """
        Treat columns as if it is a batch of natural language utterance
        with batch-size = # of columns * # of batch_size
        i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
        """
        l_hpu = []
        for i_hds1 in i_hds:
            for i_hds11 in i_hds1:
                l_hpu.append(i_hds11[1] - i_hds11[0])

        return l_hpu

    def get_bert_output(self, model_bert, tokenizer, nlu_t, hs_t, col_types,
                        units, his_sql, q_know, t_know, schema_link):
        """
        Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.

        INPUT
        :param model_bert:
        :param tokenizer: WordPiece toknizer
        :param nlu: Question
        :param nlu_t: CoreNLP tokenized nlu.
        :param hds: Headers
        :param hs_t: None or 1st-level tokenized headers
        :param max_seq_length: max input token length

        OUTPUT
        tokens: BERT input tokens
        nlu_tt: WP-tokenized input natural language questions
        orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
        tok_to_orig_index: inverse map.

        """

        l_n = []
        l_hs = []  # The length of columns for each batch

        input_ids = []
        order_ids = []
        type_ids = []
        segment_ids = []
        match_ids = []
        input_mask = []

        i_nlu = [
        ]  # index to retreive the position of contextual vector later.
        i_hds = []
        tokens = []
        orders = []
        types = []
        matchs = []
        segments = []
        schema_link_matrix_list = []
        schema_link_mask_list = []
        start_index = []
        column_index = []
        col_dict_list = []
        header_list = []
        header_flatten_token_list = []
        header_flatten_tokenid_list = []
        header_flatten_index_list = []

        header_tok_max_len = 0
        cur_max_length = 0

        for b, nlu_t1 in enumerate(nlu_t):
            hs_t1 = [hs_t[b][-1]] + hs_t[b][:-1]
            type_t1 = [col_types[b][-1]] + col_types[b][:-1]
            unit_t1 = [units[b][-1]] + units[b][:-1]
            l_hs.append(len(hs_t1))

            # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
            # 2. Generate BERT inputs & indices.
            tokens1, orders1, types1, segment1, match1, i_nlu1, i_hds_1, \
                start_idx, column_start, col_dict, schema_tok, \
                header_flatten_tokens, header_flatten_index, schema_link_matrix, schema_link_mask = \
                self.generate_inputs(
                    nlu_t1, hs_t1, type_t1, unit_t1, his_sql[b],
                    q_know[b], t_know[b], schema_link[b])

            l_n.append(i_nlu1[1] - i_nlu1[0])
            start_index.append(start_idx)
            column_index.append(column_start)
            col_dict_list.append(col_dict)
            tokens.append(tokens1)
            orders.append(orders1)
            types.append(types1)
            segments.append(segment1)
            matchs.append(match1)
            i_nlu.append(i_nlu1)
            i_hds.append(i_hds_1)
            schema_link_matrix_list.append(schema_link_matrix)
            schema_link_mask_list.append(schema_link_mask)
            header_flatten_token_list.append(header_flatten_tokens)
            header_flatten_index_list.append(header_flatten_index)
            header_list.append(schema_tok)
            header_max = max([len(schema_tok1) for schema_tok1 in schema_tok])
            if header_max > header_tok_max_len:
                header_tok_max_len = header_max

            if len(tokens1) > cur_max_length:
                cur_max_length = len(tokens1)

            if len(tokens1) > 512:
                print('input too long!!! total_num:%d\t question:%s' %
                      (len(tokens1), ''.join(nlu_t1)))

        assert cur_max_length <= 512

        for i, tokens1 in enumerate(tokens):
            segment_ids1 = segments[i]
            order_ids1 = orders[i]
            type_ids1 = types[i]
            match_ids1 = matchs[i]
            input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            input_mask1 = [1] * len(input_ids1)

            while len(input_ids1) < cur_max_length:
                input_ids1.append(0)
                input_mask1.append(0)
                segment_ids1.append(0)
                order_ids1.append(0)
                type_ids1.append(0)
                match_ids1.append(0)

            if len(input_ids1) != cur_max_length:
                print('Error: ', nlu_t1, tokens1, len(input_ids1),
                      cur_max_length)

            assert len(input_ids1) == cur_max_length
            assert len(input_mask1) == cur_max_length
            assert len(order_ids1) == cur_max_length
            assert len(segment_ids1) == cur_max_length
            assert len(match_ids1) == cur_max_length
            assert len(type_ids1) == cur_max_length

            input_ids.append(input_ids1)
            order_ids.append(order_ids1)
            type_ids.append(type_ids1)
            segment_ids.append(segment_ids1)
            input_mask.append(input_mask1)
            match_ids.append(match_ids1)

        header_len = []
        header_ids = []
        header_max_len = max(
            [len(header_list1) for header_list1 in header_list])
        for header1 in header_list:
            header_len1 = []
            header_ids1 = []
            for header_tok in header1:
                header_len1.append(len(header_tok))
                header_tok_ids1 = tokenizer.convert_tokens_to_ids(header_tok)
                while len(header_tok_ids1) < header_tok_max_len:
                    header_tok_ids1.append(0)
                header_ids1.append(header_tok_ids1)
            while len(header_ids1) < header_max_len:
                header_ids1.append([0] * header_tok_max_len)
            header_len.append(header_len1)
            header_ids.append(header_ids1)

        for i, header_flatten_token in enumerate(header_flatten_token_list):
            header_flatten_tokenid = tokenizer.convert_tokens_to_ids(
                header_flatten_token)
            header_flatten_tokenid_list.append(header_flatten_tokenid)

        # Convert to tensor
        all_input_ids = torch.tensor(
            input_ids, dtype=torch.long).to(self.device)
        all_order_ids = torch.tensor(
            order_ids, dtype=torch.long).to(self.device)
        all_type_ids = torch.tensor(type_ids, dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor(
            input_mask, dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor(
            segment_ids, dtype=torch.long).to(self.device)
        all_match_ids = torch.tensor(
            match_ids, dtype=torch.long).to(self.device)
        all_header_ids = torch.tensor(
            header_ids, dtype=torch.long).to(self.device)
        all_ids = torch.arange(
            all_input_ids.shape[0], dtype=torch.long).to(self.device)

        bS = len(header_flatten_tokenid_list)
        max_header_flatten_token_length = max(
            [len(x) for x in header_flatten_tokenid_list])
        all_header_flatten_tokens = numpy.zeros(
            (bS, max_header_flatten_token_length), dtype='int32')
        all_header_flatten_index = numpy.zeros(
            (bS, max_header_flatten_token_length), dtype='int32')
        for i, header_flatten_tokenid in enumerate(
                header_flatten_tokenid_list):
            for j, tokenid in enumerate(header_flatten_tokenid):
                all_header_flatten_tokens[i, j] = tokenid
            for j, hdindex in enumerate(header_flatten_index_list[i]):
                all_header_flatten_index[i, j] = hdindex
        all_header_flatten_output = numpy.zeros((bS, header_max_len + 1),
                                                dtype='int32')
        all_header_flatten_tokens = torch.tensor(
            all_header_flatten_tokens, dtype=torch.long).to(self.device)
        all_header_flatten_index = torch.tensor(
            all_header_flatten_index, dtype=torch.long).to(self.device)
        all_header_flatten_output = torch.tensor(
            all_header_flatten_output, dtype=torch.float32).to(self.device)

        all_token_column_id = numpy.zeros((bS, cur_max_length), dtype='int32')
        all_token_column_mask = numpy.zeros((bS, cur_max_length),
                                            dtype='float32')
        for bi, col_dict in enumerate(col_dict_list):
            for ki, vi in col_dict.items():
                all_token_column_id[bi, ki] = vi + 1
                all_token_column_mask[bi, ki] = 1.0
        all_token_column_id = torch.tensor(
            all_token_column_id, dtype=torch.long).to(self.device)
        all_token_column_mask = torch.tensor(
            all_token_column_mask, dtype=torch.float32).to(self.device)

        all_schema_link_matrix = numpy.zeros(
            (bS, cur_max_length, cur_max_length), dtype='int32')
        all_schema_link_mask = numpy.zeros(
            (bS, cur_max_length, cur_max_length), dtype='float32')
        for i, schema_link_matrix in enumerate(schema_link_matrix_list):
            temp_len = schema_link_matrix.shape[0]
            all_schema_link_matrix[i, 0:temp_len,
                                   0:temp_len] = schema_link_matrix
            all_schema_link_mask[i, 0:temp_len,
                                 0:temp_len] = schema_link_mask_list[i]
        all_schema_link_matrix = torch.tensor(
            all_schema_link_matrix, dtype=torch.long).to(self.device)
        all_schema_link_mask = torch.tensor(
            all_schema_link_mask, dtype=torch.long).to(self.device)

        # 5. generate l_hpu from i_hds
        l_hpu = self.gen_l_hpu(i_hds)

        # 4. Generate BERT output.
        all_encoder_layer, pooled_output = model_bert(
            all_input_ids,
            all_header_ids,
            token_order_ids=all_order_ids,
            token_type_ids=all_segment_ids,
            attention_mask=all_input_mask,
            match_type_ids=all_match_ids,
            l_hs=l_hs,
            header_len=header_len,
            type_ids=all_type_ids,
            col_dict_list=col_dict_list,
            ids=all_ids,
            header_flatten_tokens=all_header_flatten_tokens,
            header_flatten_index=all_header_flatten_index,
            header_flatten_output=all_header_flatten_output,
            token_column_id=all_token_column_id,
            token_column_mask=all_token_column_mask,
            column_start_index=column_index,
            headers_length=l_hs,
            all_schema_link_matrix=all_schema_link_matrix,
            all_schema_link_mask=all_schema_link_mask,
            output_all_encoded_layers=False)

        return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
            l_n, l_hpu, l_hs, start_index, column_index, all_ids

    def predict(self, querys):
        self.head_model.eval()
        self.backbone_model.eval()

        nlu, nlu_t, sql_i, q_know, t_know, tb, hs_t, types, units, his_sql, schema_link = \
            self.get_fields_info(querys, None, train=False)

        with torch.no_grad():
            all_encoder_layer, _, tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, start_index, column_index, ids = \
                self.get_bert_output(
                    self.backbone_model, self.tokenizer,
                    nlu_t, hs_t, types, units, his_sql, q_know, t_know, schema_link)

            s_action, s_sc, s_sa, s_cco, s_wc, s_wo, s_wvs, s_len = self.head_model(
                all_encoder_layer, l_n, l_hs, start_index, column_index,
                tokens, ids)

        action_batch = torch.argmax(F.softmax(s_action, -1), -1).cpu().tolist()
        scco_batch = torch.argmax(F.softmax(s_cco, -1), -1).cpu().tolist()
        sc_batch = torch.argmax(F.softmax(s_sc, -1), -1).cpu().tolist()
        sa_batch = torch.argmax(F.softmax(s_sa, -1), -1).cpu().tolist()
        wc_batch = torch.argmax(F.softmax(s_wc, -1), -1).cpu().tolist()
        wo_batch = torch.argmax(F.softmax(s_wo, -1), -1).cpu().tolist()
        s_wvs_s, s_wvs_e = s_wvs
        wvss_batch = torch.argmax(F.softmax(s_wvs_s, -1), -1).cpu().tolist()
        wvse_batch = torch.argmax(F.softmax(s_wvs_e, -1), -1).cpu().tolist()
        s_slen, s_wlen = s_len
        slen_batch = torch.argmax(F.softmax(s_slen, -1), -1).cpu().tolist()
        wlen_batch = torch.argmax(F.softmax(s_wlen, -1), -1).cpu().tolist()

        pr_wvi = []
        for i in range(len(querys)):
            wvi = []
            for j in range(wlen_batch[i]):
                wvi.append([
                    max(0, wvss_batch[i][j] - 1),
                    max(0, wvse_batch[i][j] - 1)
                ])
            pr_wvi.append(wvi)
        pr_wvi_str = self.convert_string(pr_wvi, nlu, nlu_t)

        pre_results = []
        for ib in range(len(querys)):
            res_one = {}
            sql = {}
            sql['cond_conn_op'] = scco_batch[ib]
            sl = slen_batch[ib]
            sql['sel'] = list(
                numpy.array(sc_batch[ib][:sl]).astype(numpy.int32) - 1)
            sql['agg'] = list(
                numpy.array(sa_batch[ib][:sl]).astype(numpy.int32))
            sels = []
            aggs = []
            for ia, sel in enumerate(sql['sel']):
                if sel == -1:
                    if sql['agg'][ia] > 0:
                        sels.append(l_hs[ib] - 1)
                        aggs.append(sql['agg'][ia])
                    continue
                sels.append(int(sel))
                if sql['agg'][ia] == -1:
                    aggs.append(0)
                else:
                    aggs.append(int(sql['agg'][ia]))
            if len(sels) == 0:
                sels.append(l_hs[ib] - 1)
                aggs.append(0)
            assert len(sels) == len(aggs)
            sql['sel'] = sels
            sql['agg'] = aggs

            conds = []
            wl = wlen_batch[ib]
            wc_os = list(
                numpy.array(wc_batch[ib][:wl]).astype(numpy.int32) - 1)
            wo_os = list(numpy.array(wo_batch[ib][:wl]).astype(numpy.int32))
            res_one['question_tok'] = querys[ib]['question_tok']
            for i in range(wl):
                if wc_os[i] == -1:
                    continue
                conds.append([int(wc_os[i]), int(wo_os[i]), pr_wvi_str[ib][i]])
            if len(conds) == 0:
                conds.append([l_hs[ib] - 1, 2, 'Nulll'])
            sql['conds'] = conds
            res_one['question'] = querys[ib]['question']
            res_one['table_id'] = querys[ib]['table_id']
            res_one['sql'] = sql
            res_one['action'] = action_batch[ib]
            res_one['model_out'] = [
                sc_batch[ib], sa_batch[ib], wc_batch[ib], wo_batch[ib],
                wvss_batch[ib], wvse_batch[ib]
            ]
            pre_results.append(res_one)

        return pre_results

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data


        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'result':
                            {
                                'question_tok': ['有', '哪', '些', '风', '险', '类', '型', '？'],
                                'question': '有哪些风险类型？',
                                'table_id': 'fund',
                                'sql': {
                                    'cond_conn_op': 0,
                                    'sel': [5],
                                    'agg': [0],
                                    'conds': [[10, 2, 'Nulll']]
                                },
                                'action': 10,
                                'model_out': [
                                    [6, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [2, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]
                                ]
                            },
                        'history_sql': None
                    }

        Example:
            >>> from modelscope.models.nlp import TableQuestionAnswering
            >>> from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
            >>> model = TableQuestionAnswering.from_pretrained('damo/nlp_convai_text2sql_pretrain_cn')
            >>> preprocessor = TableQuestionAnsweringPreprocessor(model_dir=model.model_dir)
            >>> print(model(preprocessor({'question': '有哪些风险类型？'})))
        """
        result = self.predict(input['datas'])[0]

        return {
            'result': result,
            'history_sql': input['datas'][0]['history_sql']
        }
