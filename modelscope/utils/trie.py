# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import defaultdict


class TreeNode:

    def __init__(self):
        self.child = defaultdict(TreeNode)


class Trie:

    def __init__(self, eos):
        self.root = TreeNode()
        self.eos = eos

    def insert(self, word):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    def get_next_layer(self, word):
        cur = self.root
        for c in word:
            cur = cur.child.get(c)
            if cur is None:
                return [self.eos]
        return list(cur.child.keys())
