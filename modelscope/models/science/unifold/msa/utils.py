# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import os
from typing import Mapping, Sequence

import json
from absl import logging

from modelscope.models.science.unifold.data import protein


def get_chain_id_map(
    sequences: Sequence[str],
    descriptions: Sequence[str],
):
    """
    Makes a mapping from PDB-format chain ID to sequence and description,
    and parses the order of multi-chains
    """
    unique_seqs = []
    for seq in sequences:
        if seq not in unique_seqs:
            unique_seqs.append(seq)

    chain_id_map = {
        chain_id: {
            'descriptions': [],
            'sequence': seq
        }
        for chain_id, seq in zip(protein.PDB_CHAIN_IDS, unique_seqs)
    }
    chain_order = []

    for seq, des in zip(sequences, descriptions):
        chain_id = protein.PDB_CHAIN_IDS[unique_seqs.index(seq)]
        chain_id_map[chain_id]['descriptions'].append(des)
        chain_order.append(chain_id)

    return chain_id_map, chain_order


def divide_multi_chains(
    fasta_name: str,
    output_dir_base: str,
    sequences: Sequence[str],
    descriptions: Sequence[str],
):
    """
    Divides the multi-chains fasta into several single fasta files and
    records multi-chains mapping information.
    """
    if len(sequences) != len(descriptions):
        raise ValueError('sequences and descriptions must have equal length. '
                         f'Got {len(sequences)} != {len(descriptions)}.')
    if len(sequences) > protein.PDB_MAX_CHAINS:
        raise ValueError(
            'Cannot process more chains than the PDB format supports. '
            f'Got {len(sequences)} chains.')

    chain_id_map, chain_order = get_chain_id_map(sequences, descriptions)

    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chain_id_map_path = os.path.join(output_dir, 'chain_id_map.json')
    with open(chain_id_map_path, 'w') as f:
        json.dump(chain_id_map, f, indent=4, sort_keys=True)

    chain_order_path = os.path.join(output_dir, 'chains.txt')
    with open(chain_order_path, 'w') as f:
        f.write(' '.join(chain_order))

    logging.info('Mapping multi-chains fasta with chain order: %s',
                 ' '.join(chain_order))

    temp_names = []
    temp_paths = []
    for chain_id in chain_id_map.keys():
        temp_name = fasta_name + '_{}'.format(chain_id)
        temp_path = os.path.join(output_dir, temp_name + '.fasta')
        des = 'chain_{}'.format(chain_id)
        seq = chain_id_map[chain_id]['sequence']
        with open(temp_path, 'w') as f:
            f.write('>' + des + '\n' + seq)
        temp_names.append(temp_name)
        temp_paths.append(temp_path)
    return temp_names, temp_paths
