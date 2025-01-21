# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import re
from typing import Any, Dict, List

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor
from .utils.bridge_content_encoder import get_database_matches
from .utils.get_tables import dump_db_json_schema


class OfaTextToSqlPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for text to sql tasks
    """

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaTextToSqlPreprocessor, self).__init__(cfg, model_dir, mode,
                                                       *args, **kwargs)

        self.instruction_text = self.cfg.model.get('prompt',
                                                   ' . generating sql code.')
        self.max_struct_length = self.cfg.get('max_struct_length', 256)
        self.separator = '\t'
        self.db_schema_cache = {}
        self.database_path = os.path.join(
            os.path.abspath(model_dir), 'database')

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        build sample for training tasks.

        step 1. Get the input question and database id from text input
        step 2. Get the database structure input
        step 3. Add a pseudo ids for every input.
        step 4. Calculate the target and previous output items.
        """
        assert 'text' in self.column_map and 'text' in data, \
            'there must be `text` column in task key map and source data'
        text = data[self.column_map['text']]  # equal data['text']
        texts = text.split(self.separator)
        assert len(
            texts
        ) == 3, 'invalid input, should contain query, question and database id'
        query, question, db_id = texts

        # construct struct input
        if db_id not in self.db_schema_cache:
            self.db_schema_cache[db_id] = dump_db_json_schema(
                self.database_path + '/' + db_id + '/' + db_id + '.sqlite',
                db_id)

        question = ' '.join(question.strip().split()[:self.max_src_length])

        seq_inputs = seq2seq_input(query, question, db_id, self.database_path,
                                   self.db_schema_cache[db_id], self.cfg.model,
                                   True)
        struct_in = seq_inputs['struct_in']
        text = seq_inputs['text_in']
        seq_out = seq_inputs['seq_out']
        db_struct = seq_inputs['db_struct']

        text = '{} ; structured knowledge: {}'.format(
            text, struct_in) + self.instruction_text
        src_item = self.tokenize_text(text + self.instruction_text)
        src_item = src_item[:(self.max_src_length + self.max_struct_length
                              + 20)]

        tgt_item = self.tokenize_text(
            ' {}'.format(seq_out), add_bos=False,
            add_eos=False)[:self.max_tgt_length]
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        sample = {
            'id': 0.0,
            'source': src_item,
            'target': target_item,
            'prev_output_tokens': prev_output_item,
            'db_struct': db_struct
        }

        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        build sample for inference tasks.

        step 1. Get the input question and database id from text input
        step 2. Get the database structure input
        step 3. Add a pseudo ids for every input.
        """
        assert 'text' in self.column_map and 'text' in data, \
            'there must be `text` column in task key map and source data'
        text = data[self.column_map['text']]  # equal data['text']
        db_id = data.get(self.column_map['database'], 'culture_company')
        db_id = db_id.strip()

        # construct struct input
        if db_id not in self.db_schema_cache:
            self.db_schema_cache[db_id] = dump_db_json_schema(
                self.database_path + '/' + db_id + '/' + db_id + '.sqlite',
                db_id)

        text = ' '.join(text.strip().split()[:self.max_src_length])

        seq_inputs = seq2seq_input(None, text, db_id, self.database_path,
                                   self.db_schema_cache[db_id], self.cfg.model)
        struct_in = seq_inputs['struct_in']
        db_struct = seq_inputs['db_struct']
        text = '{} ; structured knowledge: {}'.format(
            text, struct_in) + self.instruction_text
        src_item = self.tokenize_text(text + self.instruction_text)
        src_item = src_item[:(self.max_src_length + self.max_struct_length
                              + 20)]

        sample = {'id': 0.0, 'source': src_item, 'db_struct': db_struct}

        if 'solution' in self.column_map and self.column_map[
                'solution'] in data:
            sample['label'] = ' {}'.format(data[self.column_map['solution']])
        return sample


def seq2seq_input(query,
                  question,
                  db_id,
                  db_path,
                  schema,
                  args,
                  is_train=False):
    ex = form_input_for_construction(query, question, db_id, db_path, schema)
    serialized_schema = spider_add_serialized_schema(
        ex, args)['serialized_schema'].strip()
    if not is_train:
        return {
            'struct_in': serialized_schema,
            'text_in': question,
            'db_struct': ex
        }
    question, seq_out = spider_pre_process_one_function(ex, args)
    return {
        'struct_in': serialized_schema,
        'text_in': question,
        'seq_out': seq_out,
        'db_struct': ex
    }


def spider_pre_process_one_function(item: dict, args):
    prefix = ''

    seq_out = spider_get_target(
        query=item['query'],
        db_id=item['db_id'],
        normalize_query=True,
        target_with_db_id=args.target_with_db_id,
    )

    return prefix + item['question'].strip(), seq_out


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f'{db_id} | {_normalize(query)}' if target_with_db_id else _normalize(
        query)


def normalize(query: str) -> str:

    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(' , ', ', ')

    def white_space_fix(s):
        # Remove double and triple spaces
        return ' '.join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b",
                      lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def spider_add_serialized_schema(ex: dict, args) -> dict:
    if getattr(args, 'schema_serialization_with_nl'):
        serialized_schema = serialize_schema_natural_language(
            question=ex['question'],
            db_path=ex['db_path'],
            db_id=ex['db_id'],
            db_column_names=ex['db_column_names'],
            db_table_names=ex['db_table_names'],
            db_primary_keys=ex['db_primary_keys'],
            db_foreign_keys=ex['db_foreign_keys'],
            schema_serialization_with_db_content=args.
            schema_serialization_with_db_content,
            normalize_query=True,
        )
    else:
        serialized_schema = serialize_schema(
            question=ex['question'],
            db_path=ex['db_path'],
            db_id=ex['db_id'],
            db_column_names=ex['db_column_names'],
            db_table_names=ex['db_table_names'],
            schema_serialization_type='peteshaw',
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=args.
            schema_serialization_with_db_content,
            normalize_query=True,
        )
    return {'serialized_schema': serialized_schema}


def serialize_schema_natural_language(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    db_primary_keys,
    db_foreign_keys,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    overall_description = f'{db_id} contains tables such as ' \
                          f'{", ".join([name.lower() if normalize_query else name for name in db_table_names])}.'

    def table_description_primary_key_template(primary_key):
        return f'{primary_key} is the primary key.'

    def table_description(name, column_names):
        return f'Table {name} has columns such as {", ".join(column_names)}.'

    def value_description(cv_pairs):
        return f'{"".join(["The {} contains values such as {}.".format(column, value) for column, value in cv_pairs])}'

    def foreign_key_description(table_1, column_1, table_2, column_2):
        return f'The {column_1} of {table_1} is the foreign key of {column_2} of {table_2}.'

    db_primary_keys = db_primary_keys['column_id']
    db_foreign_keys = list(
        zip(db_foreign_keys['column_id'], db_foreign_keys['other_column_id']))

    descriptions = [overall_description]
    db_table_name_strs = []
    db_column_name_strs = []
    value_sep = ', '
    for table_id, table_name in enumerate(db_table_names):
        table_name_str = table_name.lower() if normalize_query else table_name
        db_table_name_strs.append(table_name_str)
        columns = []
        column_value_pairs = []
        primary_keys = []
        for column_id, (x, y) in enumerate(
                zip(db_column_names['table_id'],
                    db_column_names['column_name'])):
            if column_id == 0:
                continue
            column_str = y.lower() if normalize_query else y
            db_column_name_strs.append(column_str)
            if x == table_id:
                columns.append(column_str)
                if column_id in db_primary_keys:
                    primary_keys.append(column_str)
                if schema_serialization_with_db_content:
                    matches = get_database_matches(
                        question=question,
                        table_name=table_name,
                        column_name=y,
                        db_path=(db_path + '/' + db_id + '/' + db_id
                                 + '.sqlite'),
                    )
                    if matches:
                        column_value_pairs.append(
                            (column_str, value_sep.join(matches)))

        table_description_columns_str = table_description(
            table_name_str, columns)
        descriptions.append(table_description_columns_str)
        table_description_primary_key_str = table_description_primary_key_template(
            ', '.join(primary_keys))
        descriptions.append(table_description_primary_key_str)
        if len(column_value_pairs) > 0:
            value_description_str = value_description(column_value_pairs)
            descriptions.append(value_description_str)

    for x, y in db_foreign_keys:
        # get the table and column of x
        x_table_name = db_table_name_strs[db_column_names['table_id'][x]]
        x_column_name = db_column_name_strs[x]
        # get the table and column of y
        y_table_name = db_table_name_strs[db_column_names['table_id'][y]]
        y_column_name = db_column_name_strs[y]
        foreign_key_description_str = foreign_key_description(
            x_table_name, x_column_name, y_table_name, y_column_name)
        descriptions.append(foreign_key_description_str)
    return ' '.join(descriptions)


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = 'peteshaw',
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    if schema_serialization_type == 'verbose':
        db_id_str = 'Database: {db_id}. '
        table_sep = '. '
        table_str = 'Table: {table}. Columns: {columns}'
        column_sep = ', '
        column_str_with_values = '{column} ({values})'
        column_str_without_values = '{column}'
        value_sep = ', '
    elif schema_serialization_type == 'peteshaw':
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = ' | {db_id}'
        table_sep = ''
        table_str = ' | {table} : {columns}'
        column_sep = ' , '
        column_str_with_values = '{column} ( {values} )'
        column_str_without_values = '{column}'
        value_sep = ' , '
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower(
        ) if normalize_query else column_name
        if schema_serialization_with_db_content:
            # print("testing")
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + '/' + db_id + '/' + db_id + '.sqlite'),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches))
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(
                        table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names['table_id'],
                            db_column_names['column_name'],
                        ),
                    ),
                )),
        ) for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(
            db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema


def form_input_for_construction(query, question, db_id, db_path, schema):
    return {
        'query':
        query,
        'question':
        question,
        'db_id':
        db_id,
        'db_path':
        db_path,
        'db_table_names':
        schema['table_names_original'],
        'db_column_names': {
            'table_id': [
                table_id
                for table_id, column_name in schema['column_names_original']
            ],
            'column_name': [
                column_name
                for table_id, column_name in schema['column_names_original']
            ]
        },
        'db_column_types':
        schema['column_types'],
        'db_primary_keys': [{
            'column_id': column_id
        } for column_id in schema['primary_keys']],
        'db_foreign_keys': {
            'column_id': [
                column_id
                for column_id, other_column_id in schema['foreign_keys']
            ],
            'other_column_id': [
                other_column_id
                for column_id, other_column_id in schema['foreign_keys']
            ]
        },
    }
