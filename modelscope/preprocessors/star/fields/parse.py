# Copyright (c) Alibaba, Inc. and its affiliates.

CLAUSE_KEYWORDS = ('SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT',
                   'INTERSECT', 'UNION', 'EXCEPT')
JOIN_KEYWORDS = ('JOIN', 'ON', 'AS')

WHERE_OPS = ('NOT_IN', 'BETWEEN', '=', '>', '<', '>=', '<=', '!=', 'IN',
             'LIKE', 'IS', 'EXISTS')
UNIT_OPS = ('NONE', '-', '+', '*', '/')
AGG_OPS = ('', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG')
TABLE_TYPE = {
    'sql': 'sql',
    'table_unit': 'table_unit',
}
COND_OPS = ('AND', 'OR')
SQL_OPS = ('INTERSECT', 'UNION', 'EXCEPT')
ORDER_OPS = ('DESC', 'ASC')


def get_select_labels(select, slot, cur_nest):
    for item in select[1]:
        if AGG_OPS[item[0]] != '':
            if slot[item[1][1][1]] == '':
                slot[item[1][1][1]] += (cur_nest + ' ' + AGG_OPS[item[0]])
            else:
                slot[item[1][1][1]] += (' ' + cur_nest + ' '
                                        + AGG_OPS[item[0]])
        else:
            if slot[item[1][1][1]] == '':
                slot[item[1][1][1]] += (cur_nest)
            else:
                slot[item[1][1][1]] += (' ' + cur_nest)
    return slot


def get_groupby_labels(groupby, slot, cur_nest):
    for item in groupby:
        if slot[item[1]] == '':
            slot[item[1]] += (cur_nest)
        else:
            slot[item[1]] += (' ' + cur_nest)
    return slot


def get_orderby_labels(orderby, limit, slot, cur_nest):
    if limit is None:
        thelimit = ''
    else:
        thelimit = ' LIMIT'
    for item in orderby[1]:
        if AGG_OPS[item[1][0]] != '':
            agg = ' ' + AGG_OPS[item[1][0]] + ' '
        else:
            agg = ' '
        if slot[item[1][1]] == '':
            slot[item[1][1]] += (
                cur_nest + agg + orderby[0].upper() + thelimit)
        else:
            slot[item[1][1]] += (' ' + cur_nest + agg + orderby[0].upper()
                                 + thelimit)

    return slot


def get_intersect_labels(intersect, slot, cur_nest):
    if isinstance(intersect, dict):
        if cur_nest != '':
            slot = get_labels(intersect, slot, cur_nest)
        else:
            slot = get_labels(intersect, slot, 'INTERSECT')
    else:
        return slot
    return slot


def get_except_labels(texcept, slot, cur_nest):
    if isinstance(texcept, dict):
        if cur_nest != '':
            slot = get_labels(texcept, slot, cur_nest)
        else:
            slot = get_labels(texcept, slot, 'EXCEPT')
    else:
        return slot
    return slot


def get_union_labels(union, slot, cur_nest):
    if isinstance(union, dict):
        if cur_nest != '':
            slot = get_labels(union, slot, cur_nest)
        else:
            slot = get_labels(union, slot, 'UNION')
    else:
        return slot
    return slot


def get_from_labels(tfrom, slot, cur_nest):
    if tfrom['table_units'][0][0] == 'sql':
        slot = get_labels(tfrom['table_units'][0][1], slot, 'OP_SEL')
    else:
        return slot
    return slot


def get_having_labels(having, slot, cur_nest):
    if len(having) == 1:
        item = having[0]
        if item[0] is True:
            neg = ' NOT'
        else:
            neg = ''
        if isinstance(item[3], dict):
            if AGG_OPS[item[2][1][0]] != '':
                agg = ' ' + AGG_OPS[item[2][1][0]]
            else:
                agg = ''
            if slot[item[2][1][1]] == '':
                slot[item[2][1][1]] += (
                    cur_nest + agg + neg + ' ' + WHERE_OPS[item[1]])
            else:
                slot[item[2][1][1]] += (' ' + cur_nest + agg + neg + ' '
                                        + WHERE_OPS[item[1]])
            slot = get_labels(item[3], slot, 'OP_SEL')
        else:
            if AGG_OPS[item[2][1][0]] != '':
                agg = ' ' + AGG_OPS[item[2][1][0]] + ' '
            else:
                agg = ' '
            if slot[item[2][1][1]] == '':
                slot[item[2][1][1]] += (cur_nest + agg + WHERE_OPS[item[1]])
            else:
                slot[item[2][1][1]] += (' ' + cur_nest + agg
                                        + WHERE_OPS[item[1]])
    else:
        for index, item in enumerate(having):
            if item[0] is True:
                neg = ' NOT'
            else:
                neg = ''
            if (index + 1 < len(having) and having[index + 1]) == 'or' or (
                    index - 1 >= 0 and having[index - 1] == 'or'):
                if AGG_OPS[item[2][1][0]] != '':
                    agg = ' ' + AGG_OPS[item[2][1][0]]
                else:
                    agg = ''
                if isinstance(item[3], dict):
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + agg + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + agg + neg
                                                + ' ' + WHERE_OPS[item[1]])
                    slot = get_labels(item[3], slot, 'OP_SEL')
                else:
                    if AGG_OPS[item[2][1][0]] != '':
                        agg = ' ' + AGG_OPS[item[2][1][0]] + ' '
                    else:
                        agg = ' '
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + ' OR' + agg + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + ' OR' + agg
                                                + WHERE_OPS[item[1]])
            elif item == 'and' or item == 'or':
                continue
            else:
                if isinstance(item[3], dict):
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                                + WHERE_OPS[item[1]])
                    slot = get_labels(item[3], slot, 'OP_SEL')
                else:
                    if AGG_OPS[item[2][1][0]] != '':
                        agg = ' ' + AGG_OPS[item[2][1][0]] + ' '
                    else:
                        agg = ' '
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + agg + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + agg
                                                + WHERE_OPS[item[1]])
    return slot


def get_where_labels(where, slot, cur_nest):
    if len(where) == 1:
        item = where[0]
        if item[0] is True:
            neg = ' NOT'
        else:
            neg = ''
        if isinstance(item[3], dict):
            if slot[item[2][1][1]] == '':
                slot[item[2][1][1]] += (
                    cur_nest + neg + ' ' + WHERE_OPS[item[1]])
            else:
                slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                        + WHERE_OPS[item[1]])
            slot = get_labels(item[3], slot, 'OP_SEL')
        else:
            if slot[item[2][1][1]] == '':
                slot[item[2][1][1]] += (
                    cur_nest + neg + ' ' + WHERE_OPS[item[1]])
            else:
                slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                        + WHERE_OPS[item[1]])
    else:
        for index, item in enumerate(where):
            if item[0] is True:
                neg = ' NOT'
            else:
                neg = ''
            if (index + 1 < len(where) and where[index + 1]) == 'or' or (
                    index - 1 >= 0 and where[index - 1] == 'or'):
                if isinstance(item[3], dict):
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                                + WHERE_OPS[item[1]])
                    slot = get_labels(item[3], slot, 'OP_SEL')
                else:
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + ' OR' + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + ' OR' + neg
                                                + ' ' + WHERE_OPS[item[1]])
            elif item == 'and' or item == 'or':
                continue
            else:
                if isinstance(item[3], dict):
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                                + WHERE_OPS[item[1]])
                    slot = get_labels(item[3], slot, 'OP_SEL')
                else:
                    if slot[item[2][1][1]] == '':
                        slot[item[2][1][1]] += (
                            cur_nest + neg + ' ' + WHERE_OPS[item[1]])
                    else:
                        slot[item[2][1][1]] += (' ' + cur_nest + neg + ' '
                                                + WHERE_OPS[item[1]])
    return slot


def get_labels(sql_struct, slot, cur_nest):

    if len(sql_struct['select']) > 0:
        if cur_nest != '':
            slot = get_select_labels(sql_struct['select'], slot,
                                     cur_nest + ' SELECT')
        else:
            slot = get_select_labels(sql_struct['select'], slot, 'SELECT')

    if sql_struct['from']:
        if cur_nest != '':
            slot = get_from_labels(sql_struct['from'], slot, 'FROM')
        else:
            slot = get_from_labels(sql_struct['from'], slot, 'FROM')

    if len(sql_struct['where']) > 0:
        if cur_nest != '':
            slot = get_where_labels(sql_struct['where'], slot,
                                    cur_nest + ' WHERE')
        else:
            slot = get_where_labels(sql_struct['where'], slot, 'WHERE')

    if len(sql_struct['groupBy']) > 0:
        if cur_nest != '':
            slot = get_groupby_labels(sql_struct['groupBy'], slot,
                                      cur_nest + ' GROUP_BY')
        else:
            slot = get_groupby_labels(sql_struct['groupBy'], slot, 'GROUP_BY')

    if len(sql_struct['having']) > 0:
        if cur_nest != '':
            slot = get_having_labels(sql_struct['having'], slot,
                                     cur_nest + ' HAVING')
        else:
            slot = get_having_labels(sql_struct['having'], slot, 'HAVING')

    if len(sql_struct['orderBy']) > 0:
        if cur_nest != '':
            slot = get_orderby_labels(sql_struct['orderBy'],
                                      sql_struct['limit'], slot,
                                      cur_nest + ' ORDER_BY')
        else:
            slot = get_orderby_labels(sql_struct['orderBy'],
                                      sql_struct['limit'], slot, 'ORDER_BY')

    if sql_struct['intersect']:
        if cur_nest != '':
            slot = get_intersect_labels(sql_struct['intersect'], slot,
                                        cur_nest + ' INTERSECT')
        else:
            slot = get_intersect_labels(sql_struct['intersect'], slot,
                                        'INTERSECT')

    if sql_struct['except']:
        if cur_nest != '':
            slot = get_except_labels(sql_struct['except'], slot,
                                     cur_nest + ' EXCEPT')
        else:
            slot = get_except_labels(sql_struct['except'], slot, 'EXCEPT')

    if sql_struct['union']:
        if cur_nest != '':
            slot = get_union_labels(sql_struct['union'], slot,
                                    cur_nest + ' UNION')
        else:
            slot = get_union_labels(sql_struct['union'], slot, 'UNION')
    return slot


def get_label(sql, column_len):
    thelabel = []
    slot = {}
    for idx in range(column_len):
        slot[idx] = ''
    for value in get_labels(sql, slot, '').values():
        thelabel.append(value)
    return thelabel
