# Copyright (c) Alibaba, Inc. and its affiliates.
import sqlite3

import json
import tqdm

from .struct import Trie


class Database:

    def __init__(self,
                 tokenizer,
                 table_file_path,
                 syn_dict_file_path,
                 is_use_sqlite=True):
        self.tokenizer = tokenizer
        self.is_use_sqlite = is_use_sqlite
        if self.is_use_sqlite:
            self.connection_obj = sqlite3.connect(
                ':memory:', check_same_thread=False)
            self.type_dict = {'text': 'TEXT', 'number': 'INT', 'date': 'TEXT'}
        self.syn_dict = self.init_syn_dict(
            syn_dict_file_path=syn_dict_file_path)
        self.tables = self.init_tables(table_file_path=table_file_path)

    def __del__(self):
        if self.is_use_sqlite:
            self.connection_obj.close()

    def init_tables(self, table_file_path):
        tables = {}
        lines = []
        if type(table_file_path) == str:
            with open(table_file_path, 'r', encoding='utf-8') as fo:
                for line in fo:
                    lines.append(line)
        elif type(table_file_path) == list:
            for path in table_file_path:
                with open(path, 'r', encoding='utf-8') as fo:
                    for line in fo:
                        lines.append(line)
        else:
            raise ValueError()

        for line in tqdm.tqdm(lines, desc='Load Tables'):
            table = json.loads(line.strip())

            table_header_length = 0
            headers_tokens = []
            for header in table['header_name']:
                header_tokens = self.tokenizer.tokenize(header)
                table_header_length += len(header_tokens)
                headers_tokens.append(header_tokens)
            empty_column = self.tokenizer.tokenize('空列')
            table_header_length += len(empty_column)
            headers_tokens.append(empty_column)
            table['tablelen'] = table_header_length
            table['header_tok'] = headers_tokens
            table['headerid2name'] = {}
            for hid, hname in zip(table['header_id'], table['header_name']):
                table['headerid2name'][hid] = hname

            table['header_types'].append('null')
            table['header_units'] = [
                self.tokenizer.tokenize(unit) for unit in table['header_units']
            ] + [[]]

            trie_set = [Trie() for _ in table['header_name']]
            for row in table['rows']:
                for ii, cell in enumerate(row):
                    if 'real' in table['header_types'][ii].lower() or \
                        'number' in table['header_types'][ii].lower() or \
                            'duration' in table['header_types'][ii].lower():
                        continue
                    word = str(cell).strip().lower()
                    trie_set[ii].insert(word, word)
                    if word in self.syn_dict.keys():
                        for term in self.syn_dict[word]:
                            if term.strip() != '':
                                trie_set[ii].insert(term, word)

            table['value_trie'] = trie_set

            # create sqlite
            if self.is_use_sqlite:
                cursor_obj = self.connection_obj.cursor()
                cursor_obj.execute('DROP TABLE IF EXISTS %s' %
                                   (table['table_id']))
                header_string = ', '.join([
                    '%s %s' %
                    (name, self.type_dict[htype]) for name, htype in zip(
                        table['header_id'], table['header_types'])
                ])
                create_table_string = 'CREATE TABLE %s (%s);' % (
                    table['table_id'], header_string)
                cursor_obj.execute(create_table_string)
                for row in table['rows']:
                    value_string = ', '.join(['"%s"' % (val) for val in row])
                    insert_row_string = 'INSERT INTO %s VALUES(%s)' % (
                        table['table_id'], value_string)
                    cursor_obj.execute(insert_row_string)

            tables[table['table_id']] = table

        return tables

    def init_syn_dict(self, syn_dict_file_path):
        lines = []
        with open(syn_dict_file_path, encoding='utf-8') as fo:
            for line in fo:
                lines.append(line)

        syn_dict = {}
        for line in tqdm.tqdm(lines, desc='Load Synonym Dict'):
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            keys = tokens[0].strip().split('|')
            values = tokens[1].strip().split('|')
            for key in keys:
                key = key.lower().strip()
                syn_dict.setdefault(key, [])
                for value in values:
                    syn_dict[key].append(value.lower().strip())

        return syn_dict
