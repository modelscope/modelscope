# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

from transformers import BertTokenizer

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import TableQuestionAnswering
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.star3.fields.database import Database
from modelscope.preprocessors.star3.fields.struct import Constant, SQLQuery
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['TableQuestionAnsweringPipeline']


@PIPELINES.register_module(
    Tasks.table_question_answering,
    module_name=Pipelines.table_question_answering_pipeline)
class TableQuestionAnsweringPipeline(Pipeline):

    def __init__(self,
                 model: Union[TableQuestionAnswering, str],
                 preprocessor: TableQuestionAnsweringPreprocessor = None,
                 db: Database = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a table question answering prediction pipeline

        Args:
            model (TableQuestionAnswering): a model instance
            preprocessor (TableQuestionAnsweringPreprocessor): a preprocessor instance
            db (Database): a database to store tables in the database
        """
        model = model if isinstance(
            model, TableQuestionAnswering) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TableQuestionAnsweringPreprocessor(model.model_dir)

        # initilize tokenizer
        self.tokenizer = BertTokenizer(
            os.path.join(model.model_dir, ModelFile.VOCAB_FILE))

        # initialize database
        if db is None:
            self.db = Database(
                tokenizer=self.tokenizer,
                table_file_path=os.path.join(model.model_dir, 'table.json'),
                syn_dict_file_path=os.path.join(model.model_dir,
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

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def post_process_multi_turn(self, history_sql, result, table):
        action = self.action_ops[result['action']]
        headers = table['header_name']
        current_sql = result['sql']

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

    def sql_dict_to_str(self, result, table):
        """
        convert sql struct to string
        """
        header_names = table['header_name'] + ['空列']
        header_ids = table['header_id'] + ['null']
        sql = result['sql']

        str_sel_list, sql_sel_list = [], []
        for idx, sel in enumerate(sql['sel']):
            header_name = header_names[sel]
            header_id = '`%s`.`%s`' % (table['table_id'], header_ids[sel])
            if sql['agg'][idx] == 0:
                str_sel_list.append(header_name)
                sql_sel_list.append(header_id)
            else:
                str_sel_list.append(self.agg_ops[sql['agg'][idx]] + '( '
                                    + header_name + ' )')
                sql_sel_list.append(self.agg_ops[sql['agg'][idx]] + '( '
                                    + header_id + ' )')

        str_cond_list, sql_cond_list = [], []
        for cond in sql['conds']:
            header_name = header_names[cond[0]]
            header_id = '`%s`.`%s`' % (table['table_id'], header_ids[cond[0]])
            op = self.cond_ops[cond[1]]
            value = cond[2]
            str_cond_list.append('( ' + header_name + ' ' + op + ' "' + value
                                 + '" )')
            sql_cond_list.append('( ' + header_id + ' ' + op + ' "' + value
                                 + '" )')

        cond = ' ' + self.cond_conn_ops[sql['cond_conn_op']] + ' '

        final_str = 'SELECT %s FROM %s WHERE %s' % (', '.join(str_sel_list),
                                                    table['table_name'],
                                                    cond.join(str_cond_list))
        final_sql = 'SELECT %s FROM `%s` WHERE %s' % (', '.join(sql_sel_list),
                                                      table['table_id'],
                                                      cond.join(sql_cond_list))
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
        result['sql'] = self.post_process_multi_turn(
            history_sql=history_sql,
            result=result,
            table=self.db.tables[result['table_id']])
        sql = self.sql_dict_to_str(
            result=result, table=self.db.tables[result['table_id']])
        output = {OutputKeys.OUTPUT: sql, OutputKeys.HISTORY: result['sql']}
        return output

    def _collate_fn(self, data):
        return data
