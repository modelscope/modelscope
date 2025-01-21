# Copyright (c) Alibaba, Inc. and its affiliates.

import collections

import networkx as nx

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    Nodes = []
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        Nodes.append(Node(node, [n for n in tmp if n < node], type))
    return Nodes, input_nodes, output_nodes


def nodeid_trans(id, cur_level, num_levels):
    if id % 2 == 1:
        gap = int(((id + 1) // 2) * num_levels * 2)
    else:
        a = (num_levels - cur_level) * 2 - 1
        b = ((id + 1) // 2) * num_levels * 2
        gap = int(a + b)
    return cur_level + gap


def gen_log2n_graph_file(log2n_graph_file, depth_multiplier):
    f = open(log2n_graph_file, 'w')
    for i in range(depth_multiplier):
        for j in [1, 2, 4, 8, 16, 32]:
            if i - j < 0:
                break
            else:
                f.write('%d,%d\n' % (i - j, i))
    f.close()


def get_log2n_graph(depth_multiplier):
    nodes = []
    connnections = []

    for i in range(depth_multiplier):
        nodes.append(i)
        for j in [1, 2, 4, 8, 16, 32]:
            if i - j < 0:
                break
            else:
                connnections.append((i - j, i))
    return nodes, connnections


def get_dense_graph(depth_multiplier):
    nodes = []
    connections = []

    for i in range(depth_multiplier):
        nodes.append(i)
        for j in range(i):
            connections.append((j, i))
    return nodes, connections


def giraffeneck_config(min_level,
                       max_level,
                       weight_method=None,
                       depth_multiplier=5,
                       with_backslash=False,
                       with_slash=False,
                       with_skip_connect=False,
                       skip_connect_type='dense'):
    """Graph config with log2n merge and panet"""
    if skip_connect_type == 'dense':
        nodes, connections = get_dense_graph(depth_multiplier)
    elif skip_connect_type == 'log2n':
        nodes, connections = get_log2n_graph(depth_multiplier)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(connections)

    drop_node = []
    nodes, input_nodes, output_nodes = get_graph_info(graph)

    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    node_ids_per_layer = {}

    pnodes = {}

    def update_drop_node(new_id, input_offsets):
        if new_id not in drop_node:
            new_id = new_id
        else:
            while new_id in drop_node:
                if new_id in pnodes:
                    for n in pnodes[new_id]['inputs_offsets']:
                        if n not in input_offsets and n not in drop_node:
                            input_offsets.append(n)
                new_id = new_id - 1
        if new_id not in input_offsets:
            input_offsets.append(new_id)

    # top-down layer
    for i in range(max_level, min_level - 1, -1):
        node_ids_per_layer[i] = []
        for id, node in enumerate(nodes):
            input_offsets = []
            if id in input_nodes:
                input_offsets.append(node_ids[i][0])
            else:
                if with_skip_connect:
                    for input_id in node.inputs:
                        new_id = nodeid_trans(input_id, i - min_level,
                                              num_levels)
                        update_drop_node(new_id, input_offsets)

            # add top2down
            new_id = nodeid_trans(id, i - min_level, num_levels)

            # add backslash node
            def cal_backslash_node(id):
                ind = id // num_levels
                mod = id % num_levels
                if ind % 2 == 0:  # even
                    if mod == (num_levels - 1):
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (
                            num_levels - 1 - mod - 1)
                else:  # odd
                    if mod == 0:
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (
                            num_levels - 1 - mod + 1)

                return last

            # add slash node
            def cal_slash_node(id):
                ind = id // num_levels
                mod = id % num_levels
                if ind % 2 == 1:  # odd
                    if mod == (num_levels - 1):
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (
                            num_levels - 1 - mod - 1)
                else:  # even
                    if mod == 0:
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (
                            num_levels - 1 - mod + 1)

                return last

            # add last node
            last = new_id - 1
            update_drop_node(last, input_offsets)

            if with_backslash:
                backslash = cal_backslash_node(new_id)
                if backslash != -1 and backslash not in input_offsets:
                    input_offsets.append(backslash)

            if with_slash:
                slash = cal_slash_node(new_id)
                if slash != -1 and slash not in input_offsets:
                    input_offsets.append(slash)

            if new_id in drop_node:
                input_offsets = []

            pnodes[new_id] = {
                'reduction': 1 << i,
                'inputs_offsets': input_offsets,
                'weight_method': weight_method,
                'is_out': 0,
            }

        input_offsets = []
        for out_id in output_nodes:
            new_id = nodeid_trans(out_id, i - min_level, num_levels)
            input_offsets.append(new_id)

        pnodes[node_ids[i][0] + num_levels * (len(nodes) + 1)] = {
            'reduction': 1 << i,
            'inputs_offsets': input_offsets,
            'weight_method': weight_method,
            'is_out': 1,
        }

    pnodes = dict(sorted(pnodes.items(), key=lambda x: x[0]))
    return pnodes


def get_graph_config(fpn_name,
                     min_level=3,
                     max_level=7,
                     weight_method='concat',
                     depth_multiplier=5,
                     with_backslash=False,
                     with_slash=False,
                     with_skip_connect=False,
                     skip_connect_type='dense'):
    name_to_config = {
        'giraffeneck':
        giraffeneck_config(
            min_level=min_level,
            max_level=max_level,
            weight_method=weight_method,
            depth_multiplier=depth_multiplier,
            with_backslash=with_backslash,
            with_slash=with_slash,
            with_skip_connect=with_skip_connect,
            skip_connect_type=skip_connect_type),
    }
    return name_to_config[fpn_name]
