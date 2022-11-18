# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import copy
import logging
import os
# from typing import *
from typing import Dict, Iterable, List, Optional, Tuple, Union

import json
import ml_collections as mlc
import numpy as np
import torch
from unicore.data import UnicoreDataset, data_utils
from unicore.distributed import utils as distributed_utils

from .data import utils
from .data.data_ops import NumpyDict, TorchDict
from .data.process import process_features, process_labels
from .data.process_multimer import (add_assembly_features,
                                    convert_monomer_features, merge_msas,
                                    pair_and_merge, post_process)

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features
    if cfg.common.is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def process_label(all_atom_positions: np.ndarray,
                  operation: Operation) -> np.ndarray:
    if operation == 'I':
        return all_atom_positions
    rot, trans = operation
    rot = np.array(rot).reshape(3, 3)
    trans = np.array(trans).reshape(3)
    return all_atom_positions @ rot.T + trans


@utils.lru_cache(maxsize=8, copy=True)
def load_single_feature(
    sequence_id: str,
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    is_monomer: bool = False,
) -> NumpyDict:

    monomer_feature = utils.load_pickle(
        os.path.join(monomer_feature_dir, f'{sequence_id}.feature.pkl.gz'))
    monomer_feature = convert_monomer_features(monomer_feature)
    chain_feature = {**monomer_feature}

    if uniprot_msa_dir is not None:
        all_seq_feature = utils.load_pickle(
            os.path.join(uniprot_msa_dir, f'{sequence_id}.uniprot.pkl.gz'))
        if is_monomer:
            chain_feature['msa'], chain_feature[
                'deletion_matrix'] = merge_msas(
                    chain_feature['msa'],
                    chain_feature['deletion_matrix'],
                    all_seq_feature['msa'],
                    all_seq_feature['deletion_matrix'],
                )  # noqa
        else:
            all_seq_feature = utils.convert_all_seq_feature(all_seq_feature)
            for key in [
                    'msa_all_seq',
                    'msa_species_identifiers_all_seq',
                    'deletion_matrix_all_seq',
            ]:
                chain_feature[key] = all_seq_feature[key]

    return chain_feature


def load_single_label(
    label_id: str,
    label_dir: str,
    symmetry_operation: Optional[Operation] = None,
) -> NumpyDict:
    label = utils.load_pickle(
        os.path.join(label_dir, f'{label_id}.label.pkl.gz'))
    if symmetry_operation is not None:
        label['all_atom_positions'] = process_label(
            label['all_atom_positions'], symmetry_operation)
    label = {
        k: v
        for k, v in label.items() if k in
        ['aatype', 'all_atom_positions', 'all_atom_mask', 'resolution']
    }
    return label


def load(
    sequence_ids: List[str],
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
) -> NumpyExample:

    all_chain_features = [
        load_single_feature(s, monomer_feature_dir, uniprot_msa_dir,
                            is_monomer) for s in sequence_ids
    ]

    if label_ids is not None:
        # load labels
        assert len(label_ids) == len(sequence_ids)
        assert label_dir is not None
        if symmetry_operations is None:
            symmetry_operations = ['I' for _ in label_ids]
        all_chain_labels = [
            load_single_label(ll, label_dir, o)
            for ll, o in zip(label_ids, symmetry_operations)
        ]
        # update labels into features to calculate spatial cropping etc.
        [f.update(ll) for f, ll in zip(all_chain_features, all_chain_labels)]

    all_chain_features = add_assembly_features(all_chain_features)

    # get labels back from features, as add_assembly_features may alter the order of inputs.
    if label_ids is not None:
        all_chain_labels = [{
            k: f[k]
            for k in
            ['aatype', 'all_atom_positions', 'all_atom_mask', 'resolution']
        } for f in all_chain_features]
    else:
        all_chain_labels = None

    asym_len = np.array([c['seq_length'] for c in all_chain_features],
                        dtype=np.int64)
    if is_monomer:
        all_chain_features = all_chain_features[0]
    else:
        all_chain_features = pair_and_merge(all_chain_features)
        all_chain_features = post_process(all_chain_features)
    all_chain_features['asym_len'] = asym_len

    return all_chain_features, all_chain_labels


def process(
    config: mlc.ConfigDict,
    mode: str,
    features: NumpyDict,
    labels: Optional[List[NumpyDict]] = None,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
) -> TorchExample:

    if mode == 'train':
        assert batch_idx is not None
        with data_utils.numpy_seed(seed, batch_idx, key='recycling'):
            num_iters = np.random.randint(
                0, config.common.max_recycling_iters + 1)
            use_clamped_fape = np.random.rand(
            ) < config[mode].use_clamped_fape_prob
    else:
        num_iters = config.common.max_recycling_iters
        use_clamped_fape = 1

    features['num_recycling_iters'] = int(num_iters)
    features['use_clamped_fape'] = int(use_clamped_fape)
    features['is_distillation'] = int(is_distillation)
    if is_distillation and 'msa_chains' in features:
        features.pop('msa_chains')

    num_res = int(features['seq_length'])
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)

    if labels is not None:
        features['resolution'] = labels[0]['resolution'].reshape(-1)

    with data_utils.numpy_seed(seed, data_idx, key='protein_feature'):
        features['crop_and_fix_size_seed'] = np.random.randint(0, 63355)
        features = utils.filter(features, desired_keys=feature_names)
        features = {k: torch.tensor(v) for k, v in features.items()}
        with torch.no_grad():
            features = process_features(features, cfg.common, cfg[mode])

    if labels is not None:
        labels = [{k: torch.tensor(v) for k, v in ll.items()} for ll in labels]
        with torch.no_grad():
            labels = process_labels(labels)

    return features, labels


def load_and_process(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    **load_kwargs,
):
    is_monomer = (
        is_distillation
        if 'is_monomer' not in load_kwargs else load_kwargs.pop('is_monomer'))
    features, labels = load(**load_kwargs, is_monomer=is_monomer)
    features, labels = process(config, mode, features, labels, seed, batch_idx,
                               data_idx, is_distillation)
    return features, labels


class UnifoldDataset(UnicoreDataset):

    def __init__(
        self,
        args,
        seed,
        config,
        data_path,
        mode='train',
        max_step=None,
        disable_sd=False,
        json_prefix='',
    ):
        self.path = data_path

        def load_json(filename):
            return json.load(open(filename, 'r', encoding='utf-8'))

        sample_weight = load_json(
            os.path.join(self.path,
                         json_prefix + mode + '_sample_weight.json'))
        self.multi_label = load_json(
            os.path.join(self.path, json_prefix + mode + '_multi_label.json'))
        self.inverse_multi_label = self._inverse_map(self.multi_label)
        self.sample_weight = {}
        for chain in self.inverse_multi_label:
            entity = self.inverse_multi_label[chain]
            self.sample_weight[chain] = sample_weight[entity]
        self.seq_sample_weight = sample_weight
        logger.info('load {} chains (unique {} sequences)'.format(
            len(self.sample_weight), len(self.seq_sample_weight)))
        self.feature_path = os.path.join(self.path, 'pdb_features')
        self.label_path = os.path.join(self.path, 'pdb_labels')
        sd_sample_weight_path = os.path.join(
            self.path, json_prefix + 'sd_train_sample_weight.json')
        if mode == 'train' and os.path.isfile(
                sd_sample_weight_path) and not disable_sd:
            self.sd_sample_weight = load_json(sd_sample_weight_path)
            logger.info('load {} self-distillation samples.'.format(
                len(self.sd_sample_weight)))
            self.sd_feature_path = os.path.join(self.path, 'sd_features')
            self.sd_label_path = os.path.join(self.path, 'sd_labels')
        else:
            self.sd_sample_weight = None
        self.batch_size = (
            args.batch_size * distributed_utils.get_data_parallel_world_size()
            * args.update_freq[0])
        self.data_len = (
            max_step * self.batch_size
            if max_step is not None else len(self.sample_weight))
        self.mode = mode
        self.num_seq, self.seq_keys, self.seq_sample_prob = self.cal_sample_weight(
            self.seq_sample_weight)
        self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
            self.sample_weight)
        if self.sd_sample_weight is not None:
            (
                self.sd_num_chain,
                self.sd_chain_keys,
                self.sd_sample_prob,
            ) = self.cal_sample_weight(self.sd_sample_weight)
        self.config = config.data
        self.seed = seed
        self.sd_prob = args.sd_prob

    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    def sample_chain(self, idx, sample_by_seq=False):
        is_distillation = False
        if self.mode == 'train':
            with data_utils.numpy_seed(self.seed, idx, key='data_sample'):
                is_distillation = ((np.random.rand(1)[0] < self.sd_prob)
                                   if self.sd_sample_weight is not None else
                                   False)
                if is_distillation:
                    prot_idx = np.random.choice(
                        self.sd_num_chain, p=self.sd_sample_prob)
                    label_name = self.sd_chain_keys[prot_idx]
                    seq_name = label_name
                else:
                    if not sample_by_seq:
                        prot_idx = np.random.choice(
                            self.num_chain, p=self.sample_prob)
                        label_name = self.chain_keys[prot_idx]
                        seq_name = self.inverse_multi_label[label_name]
                    else:
                        seq_idx = np.random.choice(
                            self.num_seq, p=self.seq_sample_prob)
                        seq_name = self.seq_keys[seq_idx]
                        label_name = np.random.choice(
                            self.multi_label[seq_name])
        else:
            label_name = self.chain_keys[idx]
            seq_name = self.inverse_multi_label[label_name]
        return seq_name, label_name, is_distillation

    def __getitem__(self, idx):
        sequence_id, label_id, is_distillation = self.sample_chain(
            idx, sample_by_seq=True)
        feature_dir, label_dir = ((self.feature_path,
                                   self.label_path) if not is_distillation else
                                  (self.sd_feature_path, self.sd_label_path))
        features, _ = load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=[sequence_id],
            monomer_feature_dir=feature_dir,
            uniprot_msa_dir=None,
            label_ids=[label_id],
            label_dir=label_dir,
            symmetry_operations=None,
            is_monomer=True,
        )
        return features

    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        return data_utils.collate_dict(samples, dim=1)

    @staticmethod
    def _inverse_map(mapping: Dict[str, List[str]]):
        inverse_mapping = {}
        for ent, refs in mapping.items():
            for ref in refs:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    ent_2 = inverse_mapping[ref]
                    assert (
                        ent == ent_2
                    ), f'multiple entities ({ent_2}, {ent}) exist for reference {ref}.'
                inverse_mapping[ref] = ent
        return inverse_mapping


class UnifoldMultimerDataset(UnifoldDataset):

    def __init__(
        self,
        args: mlc.ConfigDict,
        seed: int,
        config: mlc.ConfigDict,
        data_path: str,
        mode: str = 'train',
        max_step: Optional[int] = None,
        disable_sd: bool = False,
        json_prefix: str = '',
        **kwargs,
    ):
        super().__init__(args, seed, config, data_path, mode, max_step,
                         disable_sd, json_prefix)
        self.data_path = data_path
        self.pdb_assembly = json.load(
            open(
                os.path.join(self.data_path,
                             json_prefix + 'pdb_assembly.json'),
                encoding='utf-8'))
        self.pdb_chains = self.get_chains(self.inverse_multi_label)
        self.monomer_feature_path = os.path.join(self.data_path,
                                                 'pdb_features')
        self.uniprot_msa_path = os.path.join(self.data_path, 'pdb_uniprots')
        self.label_path = os.path.join(self.data_path, 'pdb_labels')
        self.max_chains = args.max_chains
        if self.mode == 'train':
            self.pdb_chains, self.sample_weight = self.filter_pdb_by_max_chains(
                self.pdb_chains, self.pdb_assembly, self.sample_weight,
                self.max_chains)
            self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
                self.sample_weight)

    def __getitem__(self, idx):
        seq_id, label_id, is_distillation = self.sample_chain(idx)
        if is_distillation:
            label_ids = [label_id]
            sequence_ids = [seq_id]
            monomer_feature_path, uniprot_msa_path, label_path = (
                self.sd_feature_path,
                None,
                self.sd_label_path,
            )
            symmetry_operations = None
        else:
            pdb_id = self.get_pdb_name(label_id)
            if pdb_id in self.pdb_assembly and self.mode == 'train':
                label_ids = [
                    pdb_id + '_' + id
                    for id in self.pdb_assembly[pdb_id]['chains']
                ]
                symmetry_operations = [
                    t for t in self.pdb_assembly[pdb_id]['opers']
                ]
            else:
                label_ids = self.pdb_chains[pdb_id]
                symmetry_operations = None
            sequence_ids = [
                self.inverse_multi_label[chain_id] for chain_id in label_ids
            ]
            monomer_feature_path, uniprot_msa_path, label_path = (
                self.monomer_feature_path,
                self.uniprot_msa_path,
                self.label_path,
            )

        return load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=sequence_ids,
            monomer_feature_dir=monomer_feature_path,
            uniprot_msa_dir=uniprot_msa_path,
            label_ids=label_ids,
            label_dir=label_path,
            symmetry_operations=symmetry_operations,
            is_monomer=False,
        )

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        if len(samples) <= 0:  # tackle empty batch
            return None
        feats = [s[0] for s in samples]
        labs = [s[1] for s in samples if s[1] is not None]
        try:
            feats = data_utils.collate_dict(feats, dim=1)
        except BaseException:
            raise ValueError('cannot collate features', feats)
        if not labs:
            labs = None
        return feats, labs

    @staticmethod
    def get_pdb_name(chain):
        return chain.split('_')[0]

    @staticmethod
    def get_chains(canon_chain_map):
        pdb_chains = {}
        for chain in canon_chain_map:
            pdb = UnifoldMultimerDataset.get_pdb_name(chain)
            if pdb not in pdb_chains:
                pdb_chains[pdb] = []
            pdb_chains[pdb].append(chain)
        return pdb_chains

    @staticmethod
    def filter_pdb_by_max_chains(pdb_chains, pdb_assembly, sample_weight,
                                 max_chains):
        new_pdb_chains = {}
        for chain in pdb_chains:
            if chain in pdb_assembly:
                size = len(pdb_assembly[chain]['chains'])
                if size <= max_chains:
                    new_pdb_chains[chain] = pdb_chains[chain]
            else:
                size = len(pdb_chains[chain])
                if size == 1:
                    new_pdb_chains[chain] = pdb_chains[chain]
        new_sample_weight = {
            k: sample_weight[k]
            for k in sample_weight
            if UnifoldMultimerDataset.get_pdb_name(k) in new_pdb_chains
        }
        logger.info(
            f'filtered out {len(pdb_chains) - len(new_pdb_chains)} / {len(pdb_chains)} PDBs '
            f'({len(sample_weight) - len(new_sample_weight)} / {len(sample_weight)} chains) '
            f'by max_chains {max_chains}')
        return new_pdb_chains, new_sample_weight
