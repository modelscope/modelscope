# Copyright (c) Alibaba, Inc. and its affiliates.

cond_ops = ['>', '<', '==', '!=', 'ASC', 'DESC']
agg_ops = [
    '', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM', 'COMPARE', 'GROUP BY', 'SAME'
]
conn_ops = ['', 'AND', 'OR']


class Context:

    def __init__(self):
        self.history_sql = None

    def set_history_sql(self, sql):
        self.history_sql = sql


class SQLQuery:

    def __init__(self, string, query, sql_result):
        self.string = string
        self.query = query
        self.sql_result = sql_result


class TrieNode(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.is_word = False
        self.term = None


class Trie(object):

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, term):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True
        node.term = term

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return None, False
        return node.term, True

    def match(self, query):
        start = 0
        end = 1
        length = len(query)
        ans = {}
        while start < length and end < length:
            sub = query[start:end]
            term, flag = self.search(sub)
            if flag:
                if term is not None:
                    ans[sub] = term
                end += 1
            else:
                start += 1
                end = start + 1
        return ans

    def starts_with(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
        Returns words started with prefix
        :param prefix:
        :return: words (list)
        """

        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.extend(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list

        words = []
        if not self.starts_with(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)


class TypeInfo:

    def __init__(self, label, index, linktype, value, orgvalue, pstart, pend,
                 weight):
        self.label = label
        self.index = index
        self.linktype = linktype
        self.value = value
        self.orgvalue = orgvalue
        self.pstart = pstart
        self.pend = pend
        self.weight = weight


class Constant:

    def __init__(self):
        self.action_ops = [
            'add_cond', 'change_cond', 'del_cond', 'change_focus_total',
            'change_agg_only', 'del_focus', 'restart', 'switch_table',
            'out_of_scripts', 'repeat', 'firstTurn'
        ]

        self.agg_ops = [
            '', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM', 'COMPARE', 'GROUP BY',
            'SAME', 'M2M', 'Y2Y', 'TREND'
        ]

        self.cond_ops = ['>', '<', '==', '!=', 'ASC', 'DESC']

        self.cond_conn_ops = ['', 'AND', 'OR']

        self.col_type_dict = {
            'null': 0,
            'text': 1,
            'number': 2,
            'duration': 3,
            'bool': 4,
            'date': 5
        }

        self.schema_link_dict = {
            'col_start': 1,
            'col_middle': 2,
            'col_end': 3,
            'val_start': 4,
            'val_middle': 5,
            'val_end': 6
        }

        self.max_select_num = 4

        self.max_where_num = 6

        self.limit_dict = {
            '最': 1,
            '1': 1,
            '一': 1,
            '2': 2,
            '二': 2,
            '3': 3,
            '三': 3,
            '4': 4,
            '四': 4,
            '5': 5,
            '五': 5,
            '6': 6,
            '六': 6,
            '7': 7,
            '七': 7,
            '8': 8,
            '八': 8,
            '9': 9,
            '九': 9
        }
