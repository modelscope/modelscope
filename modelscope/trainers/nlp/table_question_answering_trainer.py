# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import time
from typing import Dict, Optional

import json
import numpy
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.models.nlp.space_T_cn.table_question_answering import \
    TableQuestionAnswering
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.table_question_answering_trainer
                          )
class TableQuestionAnsweringTrainer(BaseTrainer):

    def __init__(self, model: str, cfg_file: str = None, *args, **kwargs):
        self.model = Model.from_pretrained(model)
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']

    def get_linear_schedule_with_warmup(self,
                                        optimizer,
                                        num_warmup_steps,
                                        num_training_steps,
                                        last_epoch=-1):
        """
        set scheduler.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_wc1(self, conds):
        """
        [ [wc, wo, wv],
        [wc, wo, wv], ...
        ]
        """
        wc1 = []
        for cond in conds:
            wc1.append(int(cond[0]))
        return wc1

    def get_wo1(self, conds):
        """
        [ [wc, wo, wv],
        [wc, wo, wv], ...
        ]
        """
        wo1 = []
        for cond in conds:
            wo1.append(int(cond[1]))
        return wo1

    def get_wv1(self, conds):
        """
        [ [wc, wo, wv],
        [wc, wo, wv], ...
        ]
        """
        wv1 = []
        for cond in conds:
            wv1.append(str(cond[2]))
        return wv1

    def set_from_to(self, data, start, end, value):
        for i in range(start, end + 1):
            data[i] = value
        return data

    def get_g(self, sql_i, l_hs, action):
        """
        for backward compatibility, separated with get_g
        """
        g_sc = []
        g_sa = []
        g_wn = []
        g_wc = []
        g_wo = []
        g_wv = []
        g_slen = []
        g_action = []
        g_cond_conn_op = []
        idxs = []
        for b, psql_i1 in enumerate(sql_i):
            # g_sc.append(psql_i1["sel"][0])
            # g_sa.append(psql_i1["agg"][0])
            psql_i1['sel'] = numpy.asarray(psql_i1['sel'])
            idx = numpy.argsort(psql_i1['sel'])
            # put back one
            slen = len(psql_i1['sel'])
            sid_list = list(psql_i1['sel'][idx] + 1)
            said_list = list(numpy.asarray(psql_i1['agg'])[idx])
            for i, sid in enumerate(sid_list):
                if sid >= l_hs[b]:
                    sid_list[i] = 0
                    if said_list[i] == 0:
                        slen -= 1
            sid_list += [
                0 for _ in range(self.model.max_select_num - len(sid_list))
            ]
            # put back one
            said_list += [
                0 for _ in range(self.model.max_select_num - len(said_list))
            ]
            g_sc.append(sid_list)
            g_sa.append(said_list)
            g_slen.append(0 if slen <= 0 else slen)

            psql_i1['sel'] = numpy.sort(psql_i1['sel'])
            psql_i1['agg'] = numpy.sort(psql_i1['agg'])
            assert len(psql_i1['sel']) == len(psql_i1['agg'])

            g_action.append(action[b][0])
            g_cond_conn_op.append(psql_i1['cond_conn_op'])

            conds = numpy.asarray(psql_i1['conds'])
            conds_num = [int(x) for x in conds[:, 0]]
            idx = numpy.argsort(conds_num)
            idxs.append(idx)
            psql_i1['conds'] = conds[idx]
            if not len(psql_i1['agg']) < 0:
                # put back one
                wlen = len(conds)
                wcd_list = list(
                    numpy.array(self.get_wc1(list(conds[idx]))) + 1)
                wod_list = list(numpy.array(self.get_wo1(list(conds[idx]))))
                for i, wcd in enumerate(wcd_list):
                    if wcd >= l_hs[b]:
                        wcd_list[i] = 0
                        wlen -= 1
                wcd_list += [
                    0 for _ in range(self.model.max_where_num - len(wcd_list))
                ]
                wod_list += [
                    0 for _ in range(self.model.max_where_num - len(wod_list))
                ]
                g_wc.append(wcd_list)
                g_wn.append(0 if wlen <= 0 else wlen)
                g_wo.append(wod_list)
                g_wv.append(self.get_wv1(list(conds[idx])))
            else:
                raise EnvironmentError

        return g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_cond_conn_op, g_slen, g_action, idxs

    def get_g_wvi_bert_from_g_wvi_corenlp(self, g_wvi_corenlp, l_n, idxs):
        """
        Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

        Assumption: where_str always presents in the nlu.
        """
        max_l = 0
        for elem in l_n:
            if elem > max_l:
                max_l = elem

        # for first [CLS] and end [SEP]
        max_l += 2
        g_wvi = []
        g_wv_ps = []
        g_wv_pe = []
        for b, t_obj in enumerate(g_wvi_corenlp):
            g_wvi1 = [0] * max_l
            g_wvss1 = [0] * self.model.max_where_num
            g_wvse1 = [0] * self.model.max_where_num
            for i_wn, g_wvi_corenlp11 in enumerate(
                    list(numpy.asarray(t_obj['wvi_corenlp'])[idxs[b]])):
                st_idx, ed_idx = g_wvi_corenlp11

                if st_idx == -100 and ed_idx == -100:
                    continue
                else:
                    # put back one
                    self.set_from_to(g_wvi1, st_idx + 1, ed_idx + 1, i_wn + 1)
                    g_wvss1[i_wn] = st_idx + 1
                    g_wvse1[i_wn] = ed_idx + 1

            g_wvi.append(g_wvi1)
            g_wv_ps.append(g_wvss1)
            g_wv_pe.append(g_wvse1)

        return g_wvi, (g_wv_ps, g_wv_pe)

    def loss_scco(self, s_cco, g_cond_conn_op):
        loss = torch.nn.functional.cross_entropy(
            s_cco,
            torch.tensor(g_cond_conn_op).to(self.model.device))
        return loss

    def loss_sw_se(self, s_action, s_sc, s_sa, s_cco, s_wc, s_wo, s_wvs, g_sc,
                   g_sa, g_wn, g_wc, g_wo, g_wvi, g_cond_conn_op, g_slen,
                   g_wvp, max_h_len, s_len, g_action):
        loss = 0

        loss += torch.nn.functional.cross_entropy(
            s_sc.reshape(-1, max_h_len),
            torch.tensor(g_sc).reshape(-1).to(self.model.device))
        loss += torch.nn.functional.cross_entropy(
            s_sa.reshape(-1, self.model.n_agg_ops),
            torch.tensor(g_sa).reshape(-1).to(self.model.device))

        s_slen, s_wlen = s_len
        loss += self.loss_scco(s_cco, g_cond_conn_op)
        loss += self.loss_scco(s_slen, g_slen)
        loss += self.loss_scco(s_wlen, g_wn)

        loss += self.loss_scco(s_action, g_action)

        loss += torch.nn.functional.cross_entropy(
            s_wc.reshape(-1, max_h_len),
            torch.tensor(g_wc).reshape(-1).to(self.model.device))
        loss += torch.nn.functional.cross_entropy(
            s_wo.reshape(-1, self.model.n_cond_ops),
            torch.tensor(g_wo).reshape(-1).to(self.model.device))

        s_wvs_s, s_wvs_e = s_wvs
        loss += torch.nn.functional.cross_entropy(
            s_wvs_s.reshape(-1, s_wvs_s.shape[-1]),
            torch.tensor(g_wvp[0]).reshape(-1).to(self.model.device))
        loss += torch.nn.functional.cross_entropy(
            s_wvs_e.reshape(-1, s_wvs_e.shape[-1]),
            torch.tensor(g_wvp[1]).reshape(-1).to(self.model.device))

        return loss

    def sort_agg_sel(self, aggs, sels):
        if len(aggs) != len(sels):
            return aggs, sels
        seldic = {}
        for i, sel in enumerate(sels):
            seldic[sel] = aggs[i]
        aps = sorted(seldic.items(), key=lambda d: d[0])
        new_aggs = []
        new_sels = []
        for ap in aps:
            new_sels.append(ap[0])
            new_aggs.append(ap[1])
        return new_aggs, new_sels

    def sort_conds(self, nlu, conds):
        newconds = []
        for cond in conds:
            if len(newconds) == 0:
                newconds.append(cond)
                continue
            idx = len(newconds)
            for i, newcond in enumerate(newconds):
                if cond[0] < newcond[0]:
                    idx = i
                    break
                elif cond[0] == newcond[0]:
                    val = cond[2]
                    newval = newcond[2]
                    validx = nlu.find(val)
                    newvalidx = nlu.find(newval)
                    if validx != -1 and newvalidx != -1 and validx < newvalidx:
                        idx = i
                        break
            if idx == len(newconds):
                newconds.append(cond)
            else:
                newconds.insert(idx, cond)
        return newconds

    def calculate_scores(self, answers, results, epoch=0):
        if len(answers) != len(results) or len(results) == 0:
            return

        all_sum, all_right, sc_len, cco, wc_len = 0, 0, 0, 0, 0
        act, s_agg, all_col, s_col = 0, 0, 0, 0
        all_w, w_col, w_op, w_val = 0, 0, 0, 0
        for idx, item in enumerate(tqdm.tqdm(answers, desc='evaluate')):
            nlu = item['question']
            qaSQL = item['sql']
            result = results[idx]
            sql = result['sql']
            question = result['question']
            questionToken = result['question_tok']
            rights, errors = {}, {}
            if nlu != question:
                continue
            all_sum += 1
            right = True
            if len(sql['sel']) == len(qaSQL['sel']) and len(sql['agg']) == len(
                    qaSQL['agg']):
                sc_len += 1
                rights['select number'] = None
            else:
                right = False
                errors['select number'] = None

            if item['action'][0] == result['action']:
                act += 1
                rights['action'] = None
            else:
                right = False
                errors['action'] = None

            if sql['cond_conn_op'] == qaSQL['cond_conn_op']:
                cco += 1
                rights['condition operator'] = None
            else:
                right = False
                errors['condition operator'] = None

            if len(sql['conds']) == len(qaSQL['conds']):
                wc_len += 1
                rights['where number'] = None
            else:
                right = False
                errors['where number'] = None

            all_col += max(len(sql['agg']), len(qaSQL['agg']))
            aaggs, asels = self.sort_agg_sel(qaSQL['agg'], qaSQL['sel'])
            raggs, rsels = self.sort_agg_sel(sql['agg'], sql['sel'])
            for j, agg in enumerate(aaggs):
                if j < len(raggs) and raggs[j] == agg:
                    s_agg += 1
                    rights['select aggregation'] = None
                else:
                    right = False
                    errors['select aggregation'] = None
                if j < len(rsels) and j < len(asels) and rsels[j] == asels[j]:
                    s_col += 1
                    rights['select column'] = None
                else:
                    right = False
                    errors['select column'] = None

            all_w += max(len(sql['conds']), len(qaSQL['conds']))
            aconds = self.sort_conds(nlu, qaSQL['conds'])
            rconds = self.sort_conds(nlu, sql['conds'])

            for j, cond in enumerate(aconds):
                if j >= len(rconds):
                    break

                pcond = rconds[j]
                if cond[0] == pcond[0]:
                    w_col += 1
                    rights['where column'] = None
                else:
                    right = False
                    errors['where column'] = None
                if cond[1] == pcond[1]:
                    w_op += 1
                    rights['where operator'] = None
                else:
                    right = False
                    errors['where operator'] = None
                value = ''
                try:
                    for k in range(pcond['startId'], pcond['endId'] + 1, 1):
                        value += questionToken[k].strip()
                except Exception:
                    value = ''
                valuelow = value.strip().lower()
                normal = cond[2].strip().lower()
                valuenormal = pcond[2].strip().lower()
                if (normal in valuenormal) or (normal in valuelow) or (
                        valuelow in normal) or (valuenormal in normal):
                    w_val += 1
                    rights['where value'] = None
                else:
                    right = False
                    errors['where value'] = None

            if right:
                all_right += 1

        all_ratio = all_right / (all_sum + 0.01)
        act_ratio = act / (all_sum + 0.01)
        sc_len_ratio = sc_len / (all_sum + 0.01)
        cco_ratio = cco / (all_sum + 0.01)
        wc_len_ratio = wc_len / (all_sum + 0.01)
        s_agg_ratio = s_agg / (all_col + 0.01)
        s_col_ratio = s_col / (all_col + 0.01)
        w_col_ratio = w_col / (all_w + 0.01)
        w_op_ratio = w_op / (all_w + 0.01)
        w_val_ratio = w_val / (all_w + 0.01)
        logger.info(
            '{STATIS} [epoch=%d] all_ratio: %.3f, act_ratio: %.3f, sc_len_ratio: %.3f, '
            'cco_ratio: %.3f, wc_len_ratio: %.3f, s_agg_ratio: %.3f, s_col_ratio: %.3f, '
            'w_col_ratio: %.3f, w_op_ratio: %.3f, w_val_ratio: %.3f' %
            (epoch, all_ratio, act_ratio, sc_len_ratio, cco_ratio,
             wc_len_ratio, s_agg_ratio, s_col_ratio, w_col_ratio, w_op_ratio,
             w_val_ratio))

        metrics = {
            'accuracy': all_ratio,
            'action_accuracy': act_ratio,
            'select_length_accuracy': sc_len_ratio,
            'connector_accuracy': cco_ratio,
            'where_length_accuracy': wc_len_ratio,
            'select_aggregation_accuracy': s_agg_ratio,
            'select_column_accuracy': s_col_ratio,
            'where_column_accuracy': w_col_ratio,
            'where_operator_accuracy': w_op_ratio,
            'where_value_accuracy': w_val_ratio
        }

        return metrics

    def evaluate(self, checkpoint_path=None):
        """
        Evaluate testsets
        """
        metrics = {'all_ratio': 0.0}
        if checkpoint_path is not None:
            # load model
            state_dict = torch.load(checkpoint_path)
            self.model.backbone_model.load_state_dict(
                state_dict['backbone_model'])
            self.model.head_model.load_state_dict(
                state_dict['head_model'], strict=False)

            # predict
            results = []
            for data in tqdm.tqdm(self.eval_dataset, desc='predict'):
                result = self.model.predict([data])[0]
                results.append(result)

            metrics = self.calculate_scores(self.eval_dataset, results)

        return metrics

    def train(
        self,
        batch_size=16,
        total_epoches=20,
        backbone_learning_rate=1e-5,
        head_learning_rate=5e-4,
        backbone_weight_decay=0.01,
        head_weight_decay=0.01,
        warmup_ratio=0.1,
    ):
        """
        Fine-tuning trainsets
        """
        # obtain train loader
        train_loader = DataLoader(
            batch_size=batch_size,
            dataset=self.train_dataset,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: x)

        # some params
        total_train_steps = len(train_loader) * total_epoches
        warmup_steps = int(warmup_ratio * total_train_steps)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad,
                   self.model.head_model.parameters()),
            lr=head_learning_rate,
            weight_decay=head_weight_decay)
        opt_bert = torch.optim.AdamW(
            filter(lambda p: p.requires_grad,
                   self.model.backbone_model.parameters()),
            lr=backbone_learning_rate,
            weight_decay=backbone_weight_decay)
        lr_scheduler = self.get_linear_schedule_with_warmup(
            opt, warmup_steps, total_train_steps)
        lr_scheduler_bert = self.get_linear_schedule_with_warmup(
            opt_bert, warmup_steps, total_train_steps)

        # start training
        max_accuracy = 0.0
        for epoch in range(1, total_epoches + 1):

            # train model
            self.model.head_model.train()
            self.model.backbone_model.train()
            for iB, item in enumerate(train_loader):
                nlu, nlu_t, sql_i, q_know, t_know, action, hs_t, types, units, his_sql, schema_link = \
                    self.model.get_fields_info(item, None, train=True)

                # forward process
                all_encoder_layer, _, tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, start_index, column_index, ids = \
                    self.model.get_bert_output(
                        self.model.backbone_model, self.model.tokenizer, nlu_t, hs_t,
                        types, units, his_sql, q_know, t_know, schema_link)
                g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_cond_conn_op, g_slen, g_action, idxs = \
                    self.get_g(sql_i, l_hs, action)
                g_wvi, g_wvp = self.get_g_wvi_bert_from_g_wvi_corenlp(
                    item, l_n, idxs)
                s_action, s_sc, s_sa, s_cco, s_wc, s_wo, s_wvs, s_len = self.model.head_model(
                    all_encoder_layer, l_n, l_hs, start_index, column_index,
                    tokens, ids)

                # calculate loss
                max_h_len = max(l_hs)
                loss_all = self.loss_sw_se(s_action, s_sc, s_sa, s_cco, s_wc,
                                           s_wo, s_wvs, g_sc, g_sa, g_wn, g_wc,
                                           g_wo, g_wvi, g_cond_conn_op, g_slen,
                                           g_wvp, max_h_len, s_len, g_action)

                logger.info('{train} [epoch=%d/%d] [batch=%d/%d] loss: %.4f' %
                            (epoch, total_epoches, iB, len(train_loader),
                             loss_all.item()))

                # backward process
                opt.zero_grad()
                opt_bert.zero_grad()
                loss_all.backward()
                opt.step()
                lr_scheduler.step()
                opt_bert.step()
                lr_scheduler_bert.step()

            # evaluate model
            results = []
            for data in tqdm.tqdm(self.eval_dataset, desc='predict'):
                result = self.model.predict([data])[0]
                results.append(result)
            metrics = self.calculate_scores(
                self.eval_dataset, results, epoch=epoch)
            if metrics['accuracy'] >= max_accuracy:
                max_accuracy = metrics['accuracy']
                model_path = os.path.join(self.model.model_dir,
                                          'finetuned_model.bin')
                state_dict = {
                    'head_model': self.model.head_model.state_dict(),
                    'backbone_model': self.model.backbone_model.state_dict(),
                }
                torch.save(state_dict, model_path)
                logger.info(
                    'epoch %d obtain max score: %.4f, saving model to %s' %
                    (epoch, metrics['accuracy'], model_path))
