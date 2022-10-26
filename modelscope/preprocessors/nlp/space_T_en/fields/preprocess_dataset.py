# Copyright (c) Alibaba, Inc. and its affiliates.

from text2sql_lgesql.preprocess.parse_raw_json import Schema, get_schemas
from text2sql_lgesql.process_sql import get_sql

from .parse import get_label


def preprocess_dataset(processor, dataset, output_tables, database_id, tables):

    schemas, db_names, thetables = get_schemas(tables)
    intables = output_tables[database_id]
    schema = schemas[database_id]
    table = thetables[database_id]
    sql_label = []
    if len(dataset['history']) == 0 or dataset['last_sql'] == '':
        sql_label = [''] * len(intables['column_names'])
    else:
        schema = Schema(schema, table)
        try:
            sql_label = get_sql(schema, dataset['last_sql'])
        except Exception:
            sql_label = [''] * len(intables['column_names'])
        sql_label = get_label(sql_label, len(table['column_names_original']))
    theone = {'db_id': database_id}
    theone['query'] = ''
    theone['query_toks_no_value'] = []
    theone['sql'] = {}
    if len(dataset['history']) != 0:
        theone['question'] = dataset['utterance'] + ' [CLS] ' + ' [CLS] '.join(
            dataset['history'][::-1][:4])
        theone['question_toks'] = theone['question'].split()
    else:
        theone['question'] = dataset['utterance']
        theone['question_toks'] = dataset['utterance'].split()

    return [theone], sql_label
