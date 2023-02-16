# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, './')


def edge2mat(link, num_node):
    """According to the directed edge link, the adjacency matrix is constructed.
        link: [V, 2], each row is a tuple(start node, end node).
    """
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_incidence_matrix(im: np.ndarray) -> np.ndarray:
    Dl = im.sum(-1)
    num_node = im.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    res = Dn @ im
    return res


def build_digraph_incidence_matrix(num_nodes: int,
                                   edges: List[Tuple]) -> np.ndarray:
    source_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    target_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    source_graph = normalize_incidence_matrix(source_graph)
    target_graph = normalize_incidence_matrix(target_graph)
    return source_graph, target_graph


class DiGraph():

    def __init__(self, skeleton):
        super().__init__()
        self.num_nodes = len(skeleton.parents())
        self.directed_edges_hop1 = [
            (parrent, child)
            for child, parrent in enumerate(skeleton.parents()) if parrent >= 0
        ]
        self.directed_edges_hop2 = [(0, 1, 2), (0, 4, 5), (0, 7, 8), (1, 2, 3),
                                    (4, 5, 6), (7, 8, 9),
                                    (7, 8, 11), (7, 8, 14), (8, 9, 10),
                                    (8, 11, 12), (8, 14, 15), (11, 12, 13),
                                    (14, 15, 16)]  # (parrent, child)
        self.directed_edges_hop3 = [(0, 1, 2, 3), (0, 4, 5, 6), (0, 7, 8, 9),
                                    (7, 8, 9, 10), (7, 8, 11, 12),
                                    (7, 8, 14, 15), (8, 11, 12, 13),
                                    (8, 14, 15, 16)]
        self.directed_edges_hop4 = [(0, 7, 8, 9, 10), (0, 7, 8, 11, 12),
                                    (0, 7, 8, 14, 15), (7, 8, 11, 12, 13),
                                    (7, 8, 14, 15, 16)]

        self.num_edges = len(self.directed_edges_hop1)
        self.edge_left = [0, 1, 2, 10, 11, 12]
        self.edge_right = [3, 4, 5, 13, 14, 15]
        self.edge_middle = [6, 7, 8, 9]
        self.center = 0  # for h36m data skeleton
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.directed_edges_hop1)


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        - agcn: AGCN Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 skeleton=None,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        assert strategy in ['uniform', 'distance', 'spatial', 'agcn']
        self.get_edge(skeleton)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, skeleton):
        # edge is a list of [child, parent] paris
        self.num_node = len(skeleton.parents())
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(child, parrent)
                         for child, parrent in enumerate(skeleton.parents())]
        self.self_link = self_link
        self.neighbor_link = neighbor_link
        self.edge = self_link + neighbor_link
        self.center = 0  # for h36m data skeleton, root node idx

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = \
                    normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        elif strategy == 'agcn':
            A = []
            link_mat = edge2mat(self.self_link, self.num_node)
            In = normalize_digraph(edge2mat(self.neighbor_link, self.num_node))
            outward = [(j, i) for (i, j) in self.neighbor_link]
            Out = normalize_digraph(edge2mat(outward, self.num_node))
            A = np.stack((link_mat, In, Out))
            self.A = A
        else:
            raise ValueError('Do Not Exist This Strategy')


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
