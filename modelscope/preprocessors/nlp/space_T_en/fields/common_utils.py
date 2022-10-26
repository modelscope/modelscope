# Copyright (c) rhythmcao modified from https://github.com/rhythmcao/text2sql-lgesql.

import os
import sqlite3
from itertools import combinations, product

import nltk
import numpy as np
from text2sql_lgesql.utils.constants import MAX_RELATIVE_DIST

from modelscope.utils.logger import get_logger

mwtokenizer = nltk.MWETokenizer(separator='')
mwtokenizer.add_mwe(('[', 'CLS', ']'))
logger = get_logger()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], [
        "'", '"', '`', '‘', '’', '“', '”', '``', "''", '‘‘', '’’'
    ]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[
                -1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\""]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx
                                                    + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question


class SubPreprocessor():

    def __init__(self, model_dir, use_gpu=False, db_content=True):
        super(SubPreprocessor, self).__init__()
        self.model_dir = model_dir
        self.db_dir = os.path.join(model_dir, 'db')
        self.db_content = db_content

        from nltk import data
        from nltk.corpus import stopwords
        data.path.append(os.path.join(self.model_dir, 'nltk_data'))
        self.stopwords = stopwords.words('english')

        import stanza
        from stanza.resources import common
        from stanza.pipeline import core
        self.nlp = stanza.Pipeline(
            'en',
            use_gpu=use_gpu,
            dir=self.model_dir,
            processors='tokenize,pos,lemma',
            tokenize_pretokenized=True,
            download_method=core.DownloadMethod.REUSE_RESOURCES)
        self.nlp1 = stanza.Pipeline(
            'en',
            use_gpu=use_gpu,
            dir=self.model_dir,
            processors='tokenize,pos,lemma',
            download_method=core.DownloadMethod.REUSE_RESOURCES)

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, db, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.extract_subgraph(entry, db, verbose=verbose)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        table_toks, table_names = [], []
        for tab in db['table_names']:
            doc = self.nlp1(tab)
            tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            table_names.append(' '.join(tab))
        db['processed_table_toks'], db[
            'processed_table_names'] = table_toks, table_names
        column_toks, column_names = [], []
        for _, c in db['column_names']:
            doc = self.nlp1(c)
            c = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            column_names.append(' '.join(c))
        db['processed_column_toks'], db[
            'processed_column_names'] = column_toks, column_names
        column2table = list(map(lambda x: x[0], db['column_names']))
        table2columns = [[] for _ in range(len(table_names))]
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0:
                continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(
            db['column_names']), '<U100'

        tab_mat = np.array([['table-table-generic'] * t_num
                            for _ in range(t_num)],
                           dtype=dtype)
        table_fks = set(
            map(lambda pair: (column2table[pair[0]], column2table[pair[1]]),
                db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[
                    tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
            else:
                tab_mat[tab1, tab2], tab_mat[
                    tab2, tab1] = 'table-table-fk', 'table-table-fkr'
        tab_mat[list(range(t_num)),
                list(range(t_num))] = 'table-table-identity'

        col_mat = np.array([['column-column-generic'] * c_num
                            for _ in range(c_num)],
                           dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'column-column-sametable'
        col_mat[list(range(c_num)),
                list(range(c_num))] = 'column-column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[
                col2, col1] = 'column-column-fk', 'column-column-fkr'
        col_mat[0, list(range(c_num))] = '*-column-generic'
        col_mat[list(range(c_num)), 0] = 'column-*-generic'
        col_mat[0, 0] = '*-*-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-generic'] * c_num
                                for _ in range(t_num)],
                               dtype=dtype)
        col_tab_mat = np.array([['column-table-generic'] * t_num
                                for _ in range(c_num)],
                               dtype=dtype)
        cols, tabs = list(
            zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num)))))
        col_tab_mat[cols, tabs], tab_col_mat[
            tabs, cols] = 'column-table-has', 'table-column-has'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(
                zip(*list(
                    map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[
                tabs, cols] = 'column-table-pk', 'table-column-pk'
        col_tab_mat[0, list(range(t_num))] = '*-table-generic'
        tab_col_mat[list(range(t_num)), 0] = 'table-*-generic'

        relations = \
            np.concatenate([
                np.concatenate([tab_mat, tab_col_mat], axis=1),
                np.concatenate([col_tab_mat, col_mat], axis=1)
            ], axis=0)
        db['relations'] = relations.tolist()

        if verbose:
            print('Tables:', ', '.join(db['table_names']))
            print('Lemmatized:', ', '.join(table_names))
            print('Columns:',
                  ', '.join(list(map(lambda x: x[1], db['column_names']))))
            print('Lemmatized:', ', '.join(column_names), '\n')
        return db

    def preprocess_question(self,
                            entry: dict,
                            db: dict,
                            verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # stanza tokenize, lemmatize and POS tag
        question = ' '.join(quote_normalization(entry['question_toks']))

        from nltk import data
        data.path.append(os.path.join(self.model_dir, 'nltk_data'))

        zippath = os.path.join(self.model_dir, 'nltk_data/tokenizers/punkt')
        if os.path.exists(zippath):
            print('punkt has already exist!')
        else:
            import zipfile
            with zipfile.ZipFile(zippath + '.zip') as zf:
                zf.extractall(
                    os.path.join(self.model_dir, 'nltk_data/tokenizers/'))
        question = nltk.word_tokenize(question)
        question = mwtokenizer.tokenize(question)

        doc = self.nlp([question])
        raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
        toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
        pos_tags = [w.xpos for s in doc.sentences for w in s.words]

        entry['raw_question_toks'] = raw_toks
        entry['processed_question_toks'] = toks
        entry['pos_tags'] = pos_tags

        q_num, dtype = len(toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = [
                'question-question-dist'
                + str(i) if i != 0 else 'question-question-identity'
                for i in range(-MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)
            ]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] \
                * (q_num - MAX_RELATIVE_DIST - 1) + \
                [
                    'question-question-dist' + str(i)
                    if i != 0 else 'question-question-identity'
                    for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1,
                                   1)]\
                + ['question-question-generic'] \
                * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        list_data = \
            [dist_vec[starting - i:starting - i + q_num] for i in range(q_num)]
        q_mat = \
            np.array(
                list_data,
                dtype=dtype
            )
        entry['relations'] = q_mat.tolist()

        if verbose:
            print('Question:', entry['question'])
            print('Tokenized:', ' '.join(entry['raw_question_toks']))
            print('Lemmatized:', ' '.join(entry['processed_question_toks']))
            print('Pos tags:', ' '.join(entry['pos_tags']), '\n')
        return entry

    def extract_subgraph(self, entry: dict, db: dict, verbose: bool = False):
        used_schema = {'table': set(), 'column': set()}
        entry['used_tables'] = sorted(list(used_schema['table']))
        entry['used_columns'] = sorted(list(used_schema['column']))

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used columns:', entry['used_columns'], '\n')
        return entry

    def extract_subgraph_from_sql(self, sql: dict, used_schema: dict):
        select_items = sql['select'][1]
        # select clause
        for _, val_unit in select_items:
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
        # from clause conds
        table_units = sql['from']['table_units']
        for _, t in table_units:
            if type(t) == dict:
                used_schema = self.extract_subgraph_from_sql(t, used_schema)
            else:
                used_schema['table'].add(t)
        # from, where and having conds
        used_schema = self.extract_subgraph_from_conds(sql['from']['conds'],
                                                       used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['where'],
                                                       used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['having'],
                                                       used_schema)
        # groupBy and orderBy clause
        groupBy = sql['groupBy']
        for col_unit in groupBy:
            used_schema['column'].add(col_unit[1])
        orderBy = sql['orderBy']
        if len(orderBy) > 0:
            orderBy = orderBy[1]
            for val_unit in orderBy:
                if val_unit[0] == 0:
                    col_unit = val_unit[1]
                    used_schema['column'].add(col_unit[1])
                else:
                    col_unit1, col_unit2 = val_unit[1:]
                    used_schema['column'].add(col_unit1[1])
                    used_schema['column'].add(col_unit2[1])
        # union, intersect and except clause
        if sql['intersect']:
            used_schema = self.extract_subgraph_from_sql(
                sql['intersect'], used_schema)
        if sql['union']:
            used_schema = self.extract_subgraph_from_sql(
                sql['union'], used_schema)
        if sql['except']:
            used_schema = self.extract_subgraph_from_sql(
                sql['except'], used_schema)
        return used_schema

    def extract_subgraph_from_conds(self, conds: list, used_schema: dict):
        if len(conds) == 0:
            return used_schema
        for cond in conds:
            if cond in ['and', 'or']:
                continue
            val_unit, val1, val2 = cond[2:]
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
            if type(val1) == list:
                used_schema['column'].add(val1[1])
            elif type(val1) == dict:
                used_schema = self.extract_subgraph_from_sql(val1, used_schema)
            if type(val2) == list:
                used_schema['column'].add(val1[1])
            elif type(val2) == dict:
                used_schema = self.extract_subgraph_from_sql(val2, used_schema)
        return used_schema

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        raw_question_toks, question_toks = entry['raw_question_toks'], entry[
            'processed_question_toks']
        table_toks, column_toks = db['processed_table_toks'], db[
            'processed_column_toks']
        table_names, column_names = db['processed_table_names'], db[
            'processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(
            column_toks), '<U100'

        # relations between questions and tables, q_num*t_num and t_num*q_num
        table_matched_pairs = {'partial': [], 'exact': []}
        q_tab_mat = np.array([['question-table-nomatch'] * t_num
                              for _ in range(q_num)],
                             dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num
                              for _ in range(t_num)],
                             dtype=dtype)
        max_len = max([len(t) for t in table_toks])
        index_pairs = list(
            filter(lambda x: x[1] - x[0] <= max_len,
                   combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i:j])
            if phrase in self.stopwords:
                continue
            for idx, name in enumerate(table_names):
                if phrase == name:
                    q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
                    if verbose:
                        table_matched_pairs['exact'].append(
                            str((name, idx, phrase, i, j)))
                elif (j - i == 1
                      and phrase in name.split()) or (j - i > 1
                                                      and phrase in name):
                    q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'
                    if verbose:
                        table_matched_pairs['partial'].append(
                            str((name, idx, phrase, i, j)))

        # relations between questions and columns
        column_matched_pairs = {'partial': [], 'exact': [], 'value': []}
        q_col_mat = np.array([['question-column-nomatch'] * c_num
                              for _ in range(q_num)],
                             dtype=dtype)
        col_q_mat = np.array([['column-question-nomatch'] * q_num
                              for _ in range(c_num)],
                             dtype=dtype)
        max_len = max([len(c) for c in column_toks])
        index_pairs = list(
            filter(lambda x: x[1] - x[0] <= max_len,
                   combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i:j])
            if phrase in self.stopwords:
                continue
            for idx, name in enumerate(column_names):
                if phrase == name:
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                    if verbose:
                        column_matched_pairs['exact'].append(
                            str((name, idx, phrase, i, j)))
                elif (j - i == 1
                      and phrase in name.split()) or (j - i > 1
                                                      and phrase in name):
                    q_col_mat[range(i, j),
                              idx] = 'question-column-partialmatch'
                    col_q_mat[idx,
                              range(i, j)] = 'column-question-partialmatch'
                    if verbose:
                        column_matched_pairs['partial'].append(
                            str((name, idx, phrase, i, j)))
        if self.db_content:
            db_file = os.path.join(self.db_dir, db['db_id'],
                                   db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                raise ValueError('[ERROR]: database file %s not found ...' %
                                 (db_file))
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            conn.execute('pragma foreign_keys=ON')
            for i, (tab_id,
                    col_name) in enumerate(db['column_names_original']):
                if i == 0 or 'id' in column_toks[
                        i]:  # ignore * and special token 'id'
                    continue
                tab_name = db['table_names_original'][tab_id]
                try:
                    cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";"
                                          % (col_name, tab_name))
                    cell_values = cursor.fetchall()
                    cell_values = [str(each[0]) for each in cell_values]
                    cell_values = [[str(float(each))] if is_number(each) else
                                   each.lower().split()
                                   for each in cell_values]
                except Exception as e:
                    print(e)
                for j, word in enumerate(raw_question_toks):
                    word = str(float(word)) if is_number(word) else word
                    for c in cell_values:
                        if word in c and 'nomatch' in q_col_mat[
                                j, i] and word not in self.stopwords:
                            q_col_mat[j, i] = 'question-column-valuematch'
                            col_q_mat[i, j] = 'column-question-valuematch'
                            if verbose:
                                column_matched_pairs['value'].append(
                                    str((column_names[i], i, word, j, j + 1)))
                            break
            conn.close()

        q_col_mat[:, 0] = 'question-*-generic'
        col_q_mat[0] = '*-question-generic'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Table matched: (table name, column id, \
                question span, start id, end id)')
            print(
                'Exact match:', ', '.join(table_matched_pairs['exact'])
                if table_matched_pairs['exact'] else 'empty')
            print(
                'Partial match:', ', '.join(table_matched_pairs['partial'])
                if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, \
                question span, start id, end id)')
            print(
                'Exact match:', ', '.join(column_matched_pairs['exact'])
                if column_matched_pairs['exact'] else 'empty')
            print(
                'Partial match:', ', '.join(column_matched_pairs['partial'])
                if column_matched_pairs['partial'] else 'empty')
            print(
                'Value match:', ', '.join(column_matched_pairs['value'])
                if column_matched_pairs['value'] else 'empty', '\n')
        return entry
