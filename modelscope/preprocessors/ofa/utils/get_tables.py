# Copyright (c) Alibaba, Inc. and its affiliates.

import sqlite3
import sys
import traceback

EXIST = {'atis', 'geo', 'advising', 'yelp', 'restaurants', 'imdb', 'academic'}


def convert_fk_index(data):
    fk_holder = []
    for fk in data['foreign_keys']:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        ref_cid, cid = None, None
        try:
            tid = data['table_names_original'].index(tn)
            ref_tid = data['table_names_original'].index(ref_tn)

            for i, (tab_id,
                    col_org) in enumerate(data['column_names_original']):
                if tab_id == ref_tid and ref_col == col_org:
                    ref_cid = i
                elif tid == tab_id and col == col_org:
                    cid = i
            if ref_cid and cid:
                fk_holder.append([cid, ref_cid])
        except ValueError:
            traceback.print_exc()
            print('table_names_original: ', data['table_names_original'])
            print('finding tab name: ', tn, ref_tn)
            sys.exit()
    return fk_holder


def dump_db_json_schema(db, f):
    """read table and column info"""
    conn = sqlite3.connect(db)
    conn.execute('pragma foreign_keys=ON')
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {
        'db_id': f,
        'table_names_original': [],
        'table_names': [],
        'column_names_original': [(-1, '*')],
        'column_names': [(-1, '*')],
        'column_types': ['text'],
        'primary_keys': [],
        'foreign_keys': [],
    }

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace('_', ' '))
        fks = conn.execute(
            "PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data['column_names_original'].append((i, col[1]))
            data['column_names'].append((i, col[1].lower().replace('_', ' ')))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if ('char' in col_type or col_type == '' or 'text' in col_type
                    or 'var' in col_type):
                data['column_types'].append('text')
            elif ('int' in col_type or 'numeric' in col_type
                  or 'decimal' in col_type or 'number' in col_type
                  or 'id' in col_type or 'real' in col_type
                  or 'double' in col_type or 'float' in col_type):
                data['column_types'].append('number')
            elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                data['column_types'].append('time')
            elif 'boolean' in col_type:
                data['column_types'].append('boolean')
            else:
                data['column_types'].append('others')

            if col[5] == 1:
                data['primary_keys'].append(len(data['column_names']) - 1)

    data['foreign_keys'] = fk_holder
    data['foreign_keys'] = convert_fk_index(data)

    return data
