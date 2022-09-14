# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import tqdm

from modelscope.preprocessors.star3.fields.struct import Trie


class Database:

    def __init__(self, tokenizer, table_file_path, syn_dict_file_path):
        self.tokenizer = tokenizer
        self.tables = self.init_tables(table_file_path=table_file_path)
        self.syn_dict = self.init_syn_dict(
            syn_dict_file_path=syn_dict_file_path)

    def init_tables(self, table_file_path):
        tables = {}
        lines = []
        with open(table_file_path, 'r') as fo:
            for line in fo:
                lines.append(line)

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

            table['value_trie'] = trie_set
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
