# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import json
import torch
from transformers import BertTokenizer

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import TableQuestionAnswering
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.preprocessors.nlp.space_T_cn.fields.struct import (Constant,
                                                                   SQLQuery)
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['TableQuestionAnsweringPipeline']


@PIPELINES.register_module(
    Tasks.table_question_answering,
    module_name=Pipelines.table_question_answering_pipeline)
class TableQuestionAnsweringPipeline(Pipeline):

    def __init__(self,
                 model: Union[TableQuestionAnswering, str],
                 preprocessor: TableQuestionAnsweringPreprocessor = None,
                 db: Database = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """use `model` and `preprocessor` to create a table question answering prediction pipeline

        Args:
            model (TableQuestionAnswering): a model instance
            preprocessor (TableQuestionAnsweringPreprocessor): a preprocessor instance
            db (Database): a database to store tables in the database
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            self.preprocessor = TableQuestionAnsweringPreprocessor(
                self.model.model_dir, **kwargs)

        # initilize tokenizer
        self.tokenizer = BertTokenizer(
            os.path.join(self.model.model_dir, ModelFile.VOCAB_FILE))

        # initialize database
        if db is None:
            self.db = Database(
                tokenizer=self.tokenizer,
                table_file_path=os.path.join(self.model.model_dir,
                                             'table.json'),
                syn_dict_file_path=os.path.join(self.model.model_dir,
                                                'synonym.txt'))
        else:
            self.db = db

        constant = Constant()
        self.agg_ops = constant.agg_ops
        self.cond_ops = constant.cond_ops
        self.cond_conn_ops = constant.cond_conn_ops
        self.action_ops = constant.action_ops
        self.max_select_num = constant.max_select_num
        self.max_where_num = constant.max_where_num
        self.col_type_dict = constant.col_type_dict
        self.schema_link_dict = constant.schema_link_dict
        self.limit_dict = constant.limit_dict

    def prepare_model(self):
        """ Place model on certain device for pytorch models before first inference
                """
        self._model_prepare_lock.acquire(timeout=600)
        self.model.to(self.device)
        self._model_prepare_lock.release()

    def post_process_multi_turn(self, history_sql, result, table):
        action = self.action_ops[result['action']]
        headers = table['header_name']
        current_sql = result['sql']
        current_sql['from'] = [table['table_id']]

        if history_sql is None:
            return current_sql

        if action == 'out_of_scripts':
            return history_sql

        elif action == 'switch_table':
            return current_sql

        elif action == 'restart':
            return current_sql

        elif action == 'firstTurn':
            return current_sql

        elif action == 'del_focus':
            pre_final_sql = history_sql
            pre_sels = []
            pre_aggs = []
            for idx, seli in enumerate(pre_final_sql['sel']):
                if seli not in current_sql['sel']:
                    pre_sels.append(seli)
                    pre_aggs.append(pre_final_sql['agg'][idx])

            if len(pre_sels) < 1:
                pre_sels.append(len(headers))
                pre_aggs.append(0)
            pre_final_sql['sel'] = pre_sels
            pre_final_sql['agg'] = pre_aggs

            final_conds = []
            for condi in pre_final_sql['conds']:
                if condi[0] < len(headers):
                    final_conds.append(condi)
            if len(final_conds) < 1:
                final_conds.append([len(headers), 2, 'Null'])
            pre_final_sql['conds'] = final_conds

            return pre_final_sql

        elif action == 'change_agg_only':
            pre_final_sql = history_sql
            pre_sels = []
            pre_aggs = []
            for idx, seli in enumerate(pre_final_sql['sel']):
                if seli in current_sql['sel']:
                    pre_sels.append(seli)
                    changed_aggi = -1
                    for idx_single, aggi in enumerate(current_sql['agg']):
                        if current_sql['sel'][idx_single] == seli:
                            changed_aggi = aggi
                    pre_aggs.append(changed_aggi)
                else:
                    pre_sels.append(seli)
                    pre_aggs.append(pre_final_sql['agg'][idx])
            pre_final_sql['sel'] = pre_sels
            pre_final_sql['agg'] = pre_aggs

            return pre_final_sql

        elif action == 'change_focus_total':
            pre_final_sql = history_sql
            pre_sels = current_sql['sel']
            pre_aggs = current_sql['agg']

            pre_final_sql['sel'] = pre_sels
            pre_final_sql['agg'] = pre_aggs
            for pre_condi in current_sql['conds']:
                if pre_condi[0] < len(headers):
                    in_flag = False
                    for history_condi in history_sql['conds']:
                        if pre_condi[0] == history_condi[0]:
                            in_flag = True
                    if not in_flag:
                        pre_final_sql['conds'].append(pre_condi)

            return pre_final_sql

        elif action == 'del_cond':
            pre_final_sql = history_sql

            final_conds = []

            for idx, condi in enumerate(pre_final_sql['conds']):
                if condi[0] not in current_sql['sel']:
                    final_conds.append(condi)
            pre_final_sql['conds'] = final_conds

            final_conds = []
            for condi in pre_final_sql['conds']:
                if condi[0] < len(headers):
                    final_conds.append(condi)
            if len(final_conds) < 1:
                final_conds.append([len(headers), 2, 'Null'])
            pre_final_sql['conds'] = final_conds

            return pre_final_sql

        elif action == 'change_cond':
            pre_final_sql = history_sql
            final_conds = []

            for idx, condi in enumerate(pre_final_sql['conds']):
                in_single_flag = False
                for single_condi in current_sql['conds']:
                    if condi[0] == single_condi[0]:
                        in_single_flag = True
                        final_conds.append(single_condi)
                if not in_single_flag:
                    final_conds.append(condi)
            pre_final_sql['conds'] = final_conds

            final_conds = []
            for condi in pre_final_sql['conds']:
                if condi[0] < len(headers):
                    final_conds.append(condi)
            if len(final_conds) < 1:
                final_conds.append([len(headers), 2, 'Null', 'Null'])
            pre_final_sql['conds'] = final_conds

            return pre_final_sql

        elif action == 'add_cond':
            pre_final_sql = history_sql
            final_conds = pre_final_sql['conds']
            for idx, condi in enumerate(current_sql['conds']):
                if condi[0] < len(headers):
                    final_conds.append(condi)
            pre_final_sql['conds'] = final_conds

            final_conds = []
            for condi in pre_final_sql['conds']:
                if condi[0] < len(headers):
                    final_conds.append(condi)
            if len(final_conds) < 1:
                final_conds.append([len(headers), 2, 'Null'])
            pre_final_sql['conds'] = final_conds

            return pre_final_sql

        else:
            return current_sql

    def sql_dict_to_str(self, result, tables):
        """
        convert sql struct to string
        """
        table = tables[result['sql']['from'][0]]
        header_names = table['header_name'] + ['空列']
        header_ids = table['header_id'] + ['null']
        sql = result['sql']

        str_cond_list, sql_cond_list = [], []
        where_conds, orderby_conds = [], []
        for cond in sql['conds']:
            if cond[1] in [4, 5]:
                orderby_conds.append(cond)
            else:
                where_conds.append(cond)
        for cond in where_conds:
            header_name = header_names[cond[0]]
            if header_name == '空列':
                continue
            header_id = '`%s`.`%s`' % (table['table_id'], header_ids[cond[0]])
            op = self.cond_ops[cond[1]]
            value = cond[2]
            str_cond_list.append('( ' + header_name + ' ' + op + ' "' + value
                                 + '" )')
            sql_cond_list.append('( ' + header_id + ' ' + op + ' "' + value
                                 + '" )')
        cond_str = ' ' + self.cond_conn_ops[sql['cond_conn_op']] + ' '
        str_where_conds = cond_str.join(str_cond_list)
        sql_where_conds = cond_str.join(sql_cond_list)
        if len(orderby_conds) != 0:
            str_orderby_column = ', '.join(
                [header_names[cond[0]] for cond in orderby_conds])
            sql_orderby_column = ', '.join([
                '`%s`.`%s`' % (table['table_id'], header_ids[cond[0]])
                for cond in orderby_conds
            ])
            str_orderby_op = self.cond_ops[orderby_conds[0][1]]
            str_orderby = '%s %s' % (str_orderby_column, str_orderby_op)
            sql_orderby = '%s %s' % (sql_orderby_column, str_orderby_op)
            limit_key = orderby_conds[0][2]
            is_in, limit_num = False, -1
            for key in self.limit_dict:
                if key in limit_key:
                    is_in = True
                    limit_num = self.limit_dict[key]
                    break
            if is_in:
                str_orderby += ' LIMIT %d' % (limit_num)
                sql_orderby += ' LIMIT %d' % (limit_num)
            # post process null column
            for idx, sel in enumerate(sql['sel']):
                if sel == len(header_ids) - 1:
                    primary_sel = 0
                    for index, attrib in enumerate(table['header_attribute']):
                        if attrib == 'PRIMARY':
                            primary_sel = index
                            break
                    if primary_sel not in sql['sel']:
                        sql['sel'][idx] = primary_sel
                    else:
                        del sql['sel'][idx]
        else:
            str_orderby = ''

        str_sel_list, sql_sel_list = [], []
        for idx, sel in enumerate(sql['sel']):
            header_name = header_names[sel]
            header_id = '`%s`.`%s`' % (table['table_id'], header_ids[sel])
            if sql['agg'][idx] == 0:
                str_sel_list.append(header_name)
                sql_sel_list.append(header_id)
            elif sql['agg'][idx] == 4:
                str_sel_list.append(self.agg_ops[sql['agg'][idx]]
                                    + '(DISTINCT ' + header_name + ')')
                sql_sel_list.append(self.agg_ops[sql['agg'][idx]]
                                    + '(DISTINCT ' + header_id + ')')
            else:
                str_sel_list.append(self.agg_ops[sql['agg'][idx]] + '('
                                    + header_name + ')')
                sql_sel_list.append(self.agg_ops[sql['agg'][idx]] + '('
                                    + header_id + ')')

        if len(str_cond_list) != 0 and len(str_orderby) != 0:
            final_str = 'SELECT %s FROM %s WHERE %s ORDER BY %s' % (
                ', '.join(str_sel_list), table['table_name'], str_where_conds,
                str_orderby)
            final_sql = 'SELECT %s FROM `%s` WHERE %s ORDER BY %s' % (
                ', '.join(sql_sel_list), table['table_id'], sql_where_conds,
                sql_orderby)
        elif len(str_cond_list) != 0:
            final_str = 'SELECT %s FROM %s WHERE %s' % (
                ', '.join(str_sel_list), table['table_name'], str_where_conds)
            final_sql = 'SELECT %s FROM `%s` WHERE %s' % (
                ', '.join(sql_sel_list), table['table_id'], sql_where_conds)
        elif len(str_orderby) != 0:
            final_str = 'SELECT %s FROM %s ORDER BY %s' % (
                ', '.join(str_sel_list), table['table_name'], str_orderby)
            final_sql = 'SELECT %s FROM `%s` ORDER BY %s' % (
                ', '.join(sql_sel_list), table['table_id'], sql_orderby)
        else:
            final_str = 'SELECT %s FROM %s' % (', '.join(str_sel_list),
                                               table['table_name'])
            final_sql = 'SELECT %s FROM `%s`' % (', '.join(sql_sel_list),
                                                 table['table_id'])

        sql = SQLQuery(
            string=final_str, query=final_sql, sql_result=result['sql'])

        return sql

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        result = inputs['result']
        history_sql = inputs['history_sql']
        try:
            result['sql'] = self.post_process_multi_turn(
                history_sql=history_sql,
                result=result,
                table=self.db.tables[result['table_id']])
        except Exception:
            result['sql'] = history_sql
        sql = self.sql_dict_to_str(result=result, tables=self.db.tables)

        # add sqlite
        if self.db.is_use_sqlite:
            try:
                cursor = self.db.connection_obj.cursor().execute(sql.query)
                header_ids, header_names = [], []
                for description in cursor.description:
                    header_names.append(self.db.tables[result['table_id']]
                                        ['headerid2name'].get(
                                            description[0], description[0]))
                    header_ids.append(description[0])
                rows = []
                for res in cursor.fetchall():
                    rows.append(list(res))
                tabledata = {
                    'header_id': header_ids,
                    'header_name': header_names,
                    'rows': rows
                }
            except Exception as e:
                logger.error(e)
                tabledata = {'header_id': [], 'header_name': [], 'rows': []}
        else:
            tabledata = {'header_id': [], 'header_name': [], 'rows': []}

        output = {
            OutputKeys.SQL_STRING: sql.string,
            OutputKeys.SQL_QUERY: sql.query,
            OutputKeys.HISTORY: result['sql'],
            OutputKeys.QUERT_RESULT: tabledata,
        }

        return {OutputKeys.OUTPUT: output}

    def _collate_fn(self, data):
        return data
