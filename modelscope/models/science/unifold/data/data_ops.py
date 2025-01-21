# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import itertools
from functools import reduce, wraps
from operator import add
from typing import List, MutableMapping, Optional

import numpy as np
import torch
from unicore.data import data_utils
from unicore.utils import batched_gather, one_hot, tensor_tree_map, tree_map

from modelscope.models.science.unifold.config import (N_EXTRA_MSA, N_MSA,
                                                      N_RES, N_TPL)
from modelscope.models.science.unifold.data import residue_constants as rc
from modelscope.models.science.unifold.modules.frame import Frame, Rotation

NumpyDict = MutableMapping[str, np.ndarray]
TorchDict = MutableMapping[str, np.ndarray]

protein: TorchDict

MSA_FEATURE_NAMES = [
    'msa',
    'deletion_matrix',
    'msa_mask',
    'msa_row_mask',
    'bert_mask',
    'true_msa',
    'msa_chains',
]


def cast_to_64bit_ints(protein):
    # We keep all ints as int64
    for k, v in protein.items():
        if k.endswith('_mask'):
            protein[k] = v.type(torch.float32)
        elif v.dtype in (torch.int32, torch.uint8, torch.int8):
            protein[k] = v.type(torch.int64)

    return protein


def make_seq_mask(protein):
    protein['seq_mask'] = torch.ones(
        protein['aatype'].shape, dtype=torch.float32)
    return protein


def make_template_mask(protein):
    protein['template_mask'] = torch.ones(
        protein['template_aatype'].shape[0], dtype=torch.float32)
    return protein


def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def correct_msa_restypes(protein):
    """Correct MSA restype to have the same order as rc."""
    protein['msa'] = protein['msa'].long()
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = (
        torch.tensor(new_order_list, dtype=torch.int8).unsqueeze(-1).expand(
            -1, protein['msa'].shape[1]))
    protein['msa'] = torch.gather(new_order, 0, protein['msa']).long()

    return protein


def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    if len(protein['aatype'].shape) == 2:
        protein['aatype'] = torch.argmax(protein['aatype'], dim=-1)
    if 'resolution' in protein and len(protein['resolution'].shape) == 1:
        # use tensor for resolution
        protein['resolution'] = protein['resolution'][0]
    for k in [
            'domain_name',
            'msa',
            'num_alignments',
            'seq_length',
            'sequence',
            'superfamily',
            'deletion_matrix',
            'between_segment_residues',
            'residue_index',
            'template_all_atom_mask',
    ]:
        if k in protein and len(protein[k].shape):
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ['seq_length', 'num_alignments']:
        if k in protein and len(protein[k].shape):
            protein[k] = protein[k][0]

    return protein


@curry1
def randomly_replace_msa_with_unknown(protein, replace_proportion):
    """Replace a portion of the MSA with 'X'."""
    if replace_proportion > 0.0:
        msa_mask = np.random.rand(protein['msa'].shape) < replace_proportion
        x_idx = 20
        gap_idx = 21
        msa_mask = torch.logical_and(msa_mask, protein['msa'] != gap_idx)
        protein['msa'] = torch.where(msa_mask,
                                     torch.ones_like(protein['msa']) * x_idx,
                                     protein['msa'])
        aatype_mask = np.random.rand(
            protein['aatype'].shape) < replace_proportion

        protein['aatype'] = torch.where(
            aatype_mask,
            torch.ones_like(protein['aatype']) * x_idx,
            protein['aatype'],
        )
    return protein


def gumbel_noise(shape):
    """Generate Gumbel Noise of given Shape.
    This generates samples from Gumbel(0, 1).
    Args:
        shape: Shape of noise to return.
    Returns:
        Gumbel noise of given shape.
    """
    epsilon = 1e-6
    uniform_noise = torch.from_numpy(np.random.uniform(0, 1, shape))
    gumbel = -torch.log(-torch.log(uniform_noise + epsilon) + epsilon)
    return gumbel


def gumbel_max_sample(logits):
    """Samples from a probability distribution given by 'logits'.
    This uses Gumbel-max trick to implement the sampling in an efficient manner.
    Args:
        logits: Logarithm of probabilities to sample from, probabilities can be
        unnormalized.
    Returns:
        Sample from logprobs in one-hot form.
    """
    z = gumbel_noise(logits.shape)
    return torch.argmax(logits + z, dim=-1)


def gumbel_argsort_sample_idx(logits):
    """Samples with replacement from a distribution given by 'logits'.
    This uses Gumbel trick to implement the sampling an efficient manner. For a
    distribution over k items this samples k times without replacement, so this
    is effectively sampling a random permutation with probabilities over the
    permutations derived from the logprobs.
    Args:
        logits: Logarithm of probabilities to sample from, probabilities can be
        unnormalized.
    Returns:
        Sample from logprobs in index
    """
    z = gumbel_noise(logits.shape)
    return torch.argsort(logits + z, dim=-1, descending=True)


def uniform_permutation(num_seq):
    shuffled = torch.from_numpy(np.random.permutation(num_seq - 1) + 1)
    return torch.cat((torch.tensor([0]), shuffled), dim=0)


def gumbel_permutation(msa_mask, msa_chains=None):
    has_msa = torch.sum(msa_mask.long(), dim=-1) > 0
    # default logits is zero
    logits = torch.zeros_like(has_msa, dtype=torch.float32)
    logits[~has_msa] = -1e6
    # one sample only
    assert len(logits.shape) == 1
    # skip first row
    logits = logits[1:]
    has_msa = has_msa[1:]
    if logits.shape[0] == 0:
        return torch.tensor([0])
    if msa_chains is not None:
        # skip first row
        msa_chains = msa_chains[1:].reshape(-1)
        msa_chains[~has_msa] = 0
        keys, counts = np.unique(msa_chains, return_counts=True)
        num_has_msa = has_msa.sum()
        num_pair = (msa_chains == 1).sum()
        num_unpair = num_has_msa - num_pair
        num_chains = (keys > 1).sum()
        logits[has_msa] = 1.0 / (num_has_msa + 1e-6)
        logits[~has_msa] = 0
        for k in keys:
            if k > 1:
                cur_mask = msa_chains == k
                cur_cnt = cur_mask.sum()
                if cur_cnt > 0:
                    logits[cur_mask] *= num_unpair / (num_chains * cur_cnt)
        logits = torch.log(logits + 1e-6)
    shuffled = gumbel_argsort_sample_idx(logits) + 1
    return torch.cat((torch.tensor([0]), shuffled), dim=0)


@curry1
def sample_msa(protein,
               max_seq,
               keep_extra,
               gumbel_sample=False,
               biased_msa_by_chain=False):
    """Sample MSA randomly, remaining sequences are stored are stored as `extra_*`."""
    num_seq = protein['msa'].shape[0]
    num_sel = min(max_seq, num_seq)
    if not gumbel_sample:
        index_order = uniform_permutation(num_seq)
    else:
        msa_chains = (
            protein['msa_chains'] if
            (biased_msa_by_chain and 'msa_chains' in protein) else None)
        index_order = gumbel_permutation(protein['msa_mask'], msa_chains)
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(index_order,
                                       [num_sel, num_seq - num_sel])

    for k in MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra:
                protein['extra_' + k] = torch.index_select(
                    protein[k], 0, not_sel_seq)
            protein[k] = torch.index_select(protein[k], 0, sel_seq)

    return protein


@curry1
def sample_msa_distillation(protein, max_seq):
    if 'is_distillation' in protein and protein['is_distillation'] == 1:
        protein = sample_msa(max_seq, keep_extra=False)(protein)
    return protein


@curry1
def random_delete_msa(protein, config):
    # to reduce the cost of msa features
    num_seq = protein['msa'].shape[0]
    seq_len = protein['msa'].shape[1]
    max_seq = config.max_msa_entry // seq_len
    if num_seq > max_seq:
        keep_index = (
            torch.from_numpy(
                np.random.choice(num_seq - 1, max_seq - 1,
                                 replace=False)).long() + 1)
        keep_index = torch.sort(keep_index)[0]
        keep_index = torch.cat((torch.tensor([0]), keep_index), dim=0)
        for k in MSA_FEATURE_NAMES:
            if k in protein:
                protein[k] = torch.index_select(protein[k], 0, keep_index)
    return protein


@curry1
def crop_extra_msa(protein, max_extra_msa):
    num_seq = protein['extra_msa'].shape[0]
    num_sel = min(max_extra_msa, num_seq)
    select_indices = torch.from_numpy(np.random.permutation(num_seq)[:num_sel])
    for k in MSA_FEATURE_NAMES:
        if 'extra_' + k in protein:
            protein['extra_' + k] = torch.index_select(protein['extra_' + k],
                                                       0, select_indices)

    return protein


def delete_extra_msa(protein):
    for k in MSA_FEATURE_NAMES:
        if 'extra_' + k in protein:
            del protein['extra_' + k]
    return protein


@curry1
def block_delete_msa(protein, config):
    if 'is_distillation' in protein and protein['is_distillation'] == 1:
        return protein
    num_seq = protein['msa'].shape[0]
    if num_seq <= config.min_num_msa:
        return protein
    block_num_seq = torch.floor(
        torch.tensor(num_seq, dtype=torch.float32)
        * config.msa_fraction_per_block).to(torch.int32)

    if config.randomize_num_blocks:
        nb = np.random.randint(0, config.num_blocks + 1)
    else:
        nb = config.num_blocks

    del_block_starts = torch.from_numpy(np.random.randint(0, num_seq, [nb]))
    del_blocks = del_block_starts[:, None] + torch.arange(0, block_num_seq)
    del_blocks = torch.clip(del_blocks, 0, num_seq - 1)
    del_indices = torch.unique(del_blocks.view(-1))
    # add zeros to ensure cnt_zero > 1
    combined = torch.hstack((torch.arange(0, num_seq)[None], del_indices[None],
                             torch.zeros(2)[None])).long()
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    keep_indices = difference.view(-1)
    keep_indices = torch.hstack(
        [torch.zeros(1).long()[None], keep_indices[None]]).view(-1)
    assert int(keep_indices[0]) == 0
    for k in MSA_FEATURE_NAMES:
        if k in protein:
            protein[k] = torch.index_select(protein[k], 0, index=keep_indices)
    return protein


@curry1
def nearest_neighbor_clusters(protein, gap_agreement_weight=0.0):
    weights = torch.cat(
        [torch.ones(21), gap_agreement_weight * torch.ones(1),
         torch.zeros(1)],
        0,
    )

    msa_one_hot = one_hot(protein['msa'], 23)
    sample_one_hot = protein['msa_mask'][:, :, None] * msa_one_hot
    extra_msa_one_hot = one_hot(protein['extra_msa'], 23)
    extra_one_hot = protein['extra_msa_mask'][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    a = extra_one_hot.view(extra_num_seq, num_res * 23)
    b = (sample_one_hot * weights).view(num_seq, num_res * 23).transpose(0, 1)
    agreement = a @ b
    # Assign each sequence in the extra sequences to the closest MSA sample
    protein['extra_cluster_assignment'] = torch.argmax(agreement, dim=1).long()

    return protein


def unsorted_segment_sum(data, segment_ids, num_segments):
    assert len(
        segment_ids.shape) == 1 and segment_ids.shape[0] == data.shape[0]
    segment_ids = segment_ids.view(segment_ids.shape[0],
                                   *((1, ) * len(data.shape[1:])))
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add_(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = protein['msa'].shape[0]

    def csum(x):
        return unsorted_segment_sum(x, protein['extra_cluster_assignment'],
                                    num_seq)

    mask = protein['extra_msa_mask']
    mask_counts = 1e-6 + protein['msa_mask'] + csum(mask)  # Include center

    # TODO: this line is very slow
    msa_sum = csum(mask[:, :, None] * one_hot(protein['extra_msa'], 23))
    msa_sum += one_hot(protein['msa'], 23)  # Original sequence
    protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * protein['extra_deletion_matrix'])
    del_sum += protein['deletion_matrix']  # Original sequence
    protein['cluster_deletion_mean'] = del_sum / mask_counts
    del del_sum

    return protein


@curry1
def nearest_neighbor_clusters_v2(batch, gap_agreement_weight=0.0):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask.

    weights = torch.tensor(
        [1.0] * 21 + [gap_agreement_weight] + [0.0], dtype=torch.float32)

    msa_mask = batch['msa_mask']
    extra_mask = batch['extra_msa_mask']
    msa_one_hot = one_hot(batch['msa'], 23)
    extra_one_hot = one_hot(batch['extra_msa'], 23)

    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

    t1 = weights * msa_one_hot_masked
    t1 = t1.view(t1.shape[0], t1.shape[1] * t1.shape[2])
    t2 = extra_one_hot_masked.view(
        extra_one_hot.shape[0],
        extra_one_hot.shape[1] * extra_one_hot.shape[2])
    agreement = t1 @ t2.T

    cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, dim=0)
    cluster_assignment *= torch.einsum('mr, nr->mn', msa_mask, extra_mask)

    cluster_count = torch.sum(cluster_assignment, dim=-1)
    cluster_count += 1.0  # We always include the sequence itself.

    msa_sum = torch.einsum('nm, mrc->nrc', cluster_assignment,
                           extra_one_hot_masked)
    msa_sum += msa_one_hot_masked

    cluster_profile = msa_sum / cluster_count[:, None, None]

    deletion_matrix = batch['deletion_matrix']
    extra_deletion_matrix = batch['extra_deletion_matrix']

    del_sum = torch.einsum('nm, mc->nc', cluster_assignment,
                           extra_mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Original sequence.
    cluster_deletion_mean = del_sum / cluster_count[:, None]
    batch['cluster_profile'] = cluster_profile
    batch['cluster_deletion_mean'] = cluster_deletion_mean

    return batch


def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    if 'msa_mask' not in protein:
        protein['msa_mask'] = torch.ones(
            protein['msa'].shape, dtype=torch.float32)
    protein['msa_row_mask'] = torch.ones((protein['msa'].shape[0]),
                                         dtype=torch.float32)
    return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    if aatype.shape[0] > 0:
        is_gly = torch.eq(aatype, rc.restype_order['G'])
        ca_idx = rc.atom_order['CA']
        cb_idx = rc.atom_order['CB']
        pseudo_beta = torch.where(
            torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
            all_atom_positions[..., ca_idx, :],
            all_atom_positions[..., cb_idx, :],
        )
    else:
        pseudo_beta = all_atom_positions.new_zeros(*aatype.shape, 3)
    if all_atom_mask is not None:
        if aatype.shape[0] > 0:
            pseudo_beta_mask = torch.where(is_gly, all_atom_mask[..., ca_idx],
                                           all_atom_mask[..., cb_idx])
        else:
            pseudo_beta_mask = torch.zeros_like(aatype).float()
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(protein, prefix=''):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ['', 'template_']
    (
        protein[prefix + 'pseudo_beta'],
        protein[prefix + 'pseudo_beta_mask'],
    ) = pseudo_beta_fn(
        protein['template_aatype' if prefix else 'aatype'],
        protein[prefix + 'all_atom_positions'],
        protein['template_all_atom_mask' if prefix else 'all_atom_mask'],
    )
    return protein


@curry1
def add_constant_field(protein, key, value):
    protein[key] = torch.tensor(value)
    return protein


def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    probs = torch.reshape(probs + epsilon, [-1, num_classes])
    gen = torch.Generator()
    gen.manual_seed(np.random.randint(65535))
    counts = torch.multinomial(probs, 1, generator=gen)
    return torch.reshape(counts, ds[:-1])


def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if 'hhblits_profile' in protein:
        return protein

    # Compute the profile for every residue (over all MSA sequences).
    msa_one_hot = one_hot(protein['msa'], 22)

    protein['hhblits_profile'] = torch.mean(msa_one_hot, dim=0)
    return protein


def make_msa_profile(batch):
    """Compute the MSA profile."""
    # Compute the profile for every residue (over all MSA sequences).
    oh = one_hot(batch['msa'], 22)
    mask = batch['msa_mask'][:, :, None]
    oh *= mask
    return oh.sum(dim=0) / (mask.sum(dim=0) + 1e-10)


def make_hhblits_profile_v2(protein):
    """Compute the HHblits MSA profile if not already present."""
    if 'hhblits_profile' in protein:
        return protein
    protein['hhblits_profile'] = make_msa_profile(protein)
    return protein


def share_mask_by_entity(mask_position, protein):  # new in unifold
    if 'num_sym' not in protein:
        return mask_position
    entity_id = protein['entity_id']
    sym_id = protein['sym_id']
    num_sym = protein['num_sym']
    unique_entity_ids = entity_id.unique()
    first_sym_mask = sym_id == 1
    for cur_entity_id in unique_entity_ids:
        cur_entity_mask = entity_id == cur_entity_id
        cur_num_sym = int(num_sym[cur_entity_mask][0])
        if cur_num_sym > 1:
            cur_sym_mask = first_sym_mask & cur_entity_mask
            cur_sym_bert_mask = mask_position[:, cur_sym_mask]
            mask_position[:, cur_entity_mask] = cur_sym_bert_mask.repeat(
                1, cur_num_sym)
    return mask_position


@curry1
def make_masked_msa(protein,
                    config,
                    replace_fraction,
                    gumbel_sample=False,
                    share_mask=False):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32)

    categorical_probs = (
        config.uniform_prob * random_aa
        + config.profile_prob * protein['hhblits_profile']
        + config.same_prob * one_hot(protein['msa'], 22))

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(
        reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
    pad_shapes[1] = 1
    mask_prob = 1.0 - config.profile_prob - config.same_prob - config.uniform_prob
    assert mask_prob >= 0.0
    categorical_probs = torch.nn.functional.pad(
        categorical_probs, pad_shapes, value=mask_prob)
    sh = protein['msa'].shape
    mask_position = torch.from_numpy(np.random.rand(*sh) < replace_fraction)
    mask_position &= protein['msa_mask'].bool()

    if 'bert_mask' in protein:
        mask_position &= protein['bert_mask'].bool()

    if share_mask:
        mask_position = share_mask_by_entity(mask_position, protein)
    if gumbel_sample:
        logits = torch.log(categorical_probs + 1e-6)
        bert_msa = gumbel_max_sample(logits)
    else:
        bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein['msa'])
    bert_msa *= protein['msa_mask'].long()

    # Mix real and masked MSA
    protein['bert_mask'] = mask_position.to(torch.float32)
    protein['true_msa'] = protein['msa']
    protein['msa'] = bert_msa

    return protein


@curry1
def make_fixed_size(
    protein,
    shape_schema,
    msa_cluster_size,
    extra_msa_size,
    num_res=0,
    num_templates=0,
):
    """Guess at the MSA and sequence dimension to make fixed size."""

    def get_pad_size(cur_size, multiplier=4):
        return max(multiplier,
                   ((cur_size + multiplier - 1) // multiplier) * multiplier)

    if num_res is not None:
        input_num_res = (
            protein['aatype'].shape[0]
            if 'aatype' in protein else protein['msa_mask'].shape[1])
        if input_num_res != num_res:
            num_res = get_pad_size(input_num_res, 4)
    if 'extra_msa_mask' in protein:
        input_extra_msa_size = protein['extra_msa_mask'].shape[0]
        if input_extra_msa_size != extra_msa_size:
            extra_msa_size = get_pad_size(input_extra_msa_size, 8)
    pad_size_map = {
        N_RES: num_res,
        N_MSA: msa_cluster_size,
        N_EXTRA_MSA: extra_msa_size,
        N_TPL: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == 'extra_cluster_assignment':
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = 'Rank mismatch between shape and shape schema for'
        assert len(shape) == len(schema), f'{msg} {k}: {shape} vs {schema}'
        pad_size = [
            pad_size_map.get(s2, None) or s1
            for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)

    return protein


def make_target_feat(protein):
    """Create and concatenate MSA features."""
    protein['aatype'] = protein['aatype'].long()

    if 'between_segment_residues' in protein:
        has_break = torch.clip(
            protein['between_segment_residues'].to(torch.float32), 0, 1)
    else:
        has_break = torch.zeros_like(protein['aatype'], dtype=torch.float32)
        if 'asym_len' in protein:
            asym_len = protein['asym_len']
            entity_ends = torch.cumsum(asym_len, dim=-1)[:-1]
            has_break[entity_ends] = 1.0
        has_break = has_break.float()
    aatype_1hot = one_hot(protein['aatype'], 21)
    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,  # Everyone gets the original sequence.
    ]
    protein['target_feat'] = torch.cat(target_feat, dim=-1)
    return protein


def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    msa_1hot = one_hot(protein['msa'], 23)
    has_deletion = torch.clip(protein['deletion_matrix'], 0.0, 1.0)
    deletion_value = torch.atan(
        protein['deletion_matrix'] / 3.0) * (2.0 / np.pi)
    msa_feat = [
        msa_1hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]
    if 'cluster_profile' in protein:
        deletion_mean_value = torch.atan(
            protein['cluster_deletion_mean'] / 3.0) * (2.0 / np.pi)
        msa_feat.extend([
            protein['cluster_profile'],
            torch.unsqueeze(deletion_mean_value, dim=-1),
        ])

    if 'extra_deletion_matrix' in protein:
        protein['extra_msa_has_deletion'] = torch.clip(
            protein['extra_deletion_matrix'], 0.0, 1.0)
        protein['extra_msa_deletion_value'] = torch.atan(
            protein['extra_deletion_matrix'] / 3.0) * (2.0 / np.pi)

    protein['msa_feat'] = torch.cat(msa_feat, dim=-1)
    return protein


def make_msa_feat_v2(batch):
    """Create and concatenate MSA features."""
    msa_1hot = one_hot(batch['msa'], 23)
    deletion_matrix = batch['deletion_matrix']
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / np.pi))[...,
                                                                         None]

    deletion_mean_value = (
        torch.arctan(batch['cluster_deletion_mean'] / 3.0) *  # noqa W504
        (2.0 / np.pi))[..., None]

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
        batch['cluster_profile'],
        deletion_mean_value,
    ]
    batch['msa_feat'] = torch.cat(msa_feat, dim=-1)
    return batch


@curry1
def make_extra_msa_feat(batch, num_extra_msa):
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    extra_msa = batch['extra_msa'][:num_extra_msa]
    deletion_matrix = batch['extra_deletion_matrix'][:num_extra_msa]
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)
    deletion_value = torch.atan(deletion_matrix / 3.0) * (2.0 / np.pi)
    extra_msa_mask = batch['extra_msa_mask'][:num_extra_msa]
    batch['extra_msa'] = extra_msa
    batch['extra_msa_mask'] = extra_msa_mask
    batch['extra_msa_has_deletion'] = has_deletion
    batch['extra_msa_deletion_value'] = deletion_value
    return batch


@curry1
def select_feat(protein, feature_list):
    return {k: v for k, v in protein.items() if k in feature_list}


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""

    if 'atom14_atom_exists' in protein:  # lazy move
        return protein

    restype_atom14_to_atom37 = torch.tensor(
        rc.restype_atom14_to_atom37,
        dtype=torch.int64,
        device=protein['aatype'].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        rc.restype_atom37_to_atom14,
        dtype=torch.int64,
        device=protein['aatype'].device,
    )
    restype_atom14_mask = torch.tensor(
        rc.restype_atom14_mask,
        dtype=torch.float32,
        device=protein['aatype'].device,
    )
    restype_atom37_mask = torch.tensor(
        rc.restype_atom37_mask,
        dtype=torch.float32,
        device=protein['aatype'].device)

    protein_aatype = protein['aatype'].long()
    protein['residx_atom14_to_atom37'] = restype_atom14_to_atom37[
        protein_aatype].long()
    protein['residx_atom37_to_atom14'] = restype_atom37_to_atom14[
        protein_aatype].long()
    protein['atom14_atom_exists'] = restype_atom14_mask[protein_aatype]
    protein['atom37_atom_exists'] = restype_atom37_mask[protein_aatype]

    return protein


def make_atom14_masks_np(batch):
    batch = tree_map(lambda n: torch.tensor(n), batch, np.ndarray)
    out = make_atom14_masks(batch)
    out = tensor_tree_map(lambda t: np.array(t), out)
    return out


def make_atom14_positions(protein):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    protein['aatype'] = protein['aatype'].long()
    protein['all_atom_mask'] = protein['all_atom_mask'].float()
    protein['all_atom_positions'] = protein['all_atom_positions'].float()
    residx_atom14_mask = protein['atom14_atom_exists']
    residx_atom14_to_atom37 = protein['residx_atom14_to_atom37']

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        protein['all_atom_mask'],
        residx_atom14_to_atom37,
        dim=-1,
        num_batch_dims=len(protein['all_atom_mask'].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            protein['all_atom_positions'],
            residx_atom14_to_atom37,
            dim=-2,
            num_batch_dims=len(protein['all_atom_positions'].shape[:-2]),
        ))

    protein['atom14_atom_exists'] = residx_atom14_mask
    protein['atom14_gt_exists'] = residx_atom14_gt_mask
    protein['atom14_gt_positions'] = residx_atom14_gt_positions

    renaming_matrices = torch.tensor(
        rc.renaming_matrices,
        dtype=protein['all_atom_mask'].dtype,
        device=protein['all_atom_mask'].device,
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[protein['aatype']]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum('...rac,...rab->...rbc',
                                            residx_atom14_gt_positions,
                                            renaming_transform)
    protein['atom14_alt_gt_positions'] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum('...ra,...rab->...rb',
                                       residx_atom14_gt_mask,
                                       renaming_transform)
    protein['atom14_alt_gt_exists'] = alternative_gt_mask

    restype_atom14_is_ambiguous = torch.tensor(
        rc.restype_atom14_is_ambiguous,
        dtype=protein['all_atom_mask'].dtype,
        device=protein['all_atom_mask'].device,
    )
    # From this create an ambiguous_mask for the given sequence.
    protein['atom14_atom_is_ambiguous'] = restype_atom14_is_ambiguous[
        protein['aatype']]

    return protein


def atom37_to_frames(protein, eps=1e-8):
    # TODO: extract common part and put them into residue constants.
    aatype = protein['aatype']
    all_atom_positions = protein['all_atom_positions']
    all_atom_mask = protein['all_atom_mask']

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], '', dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ['C', 'CA', 'N']
    restype_rigidgroup_base_atom_names[:, 3, :] = ['CA', 'C', 'O']

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype,
                                                   chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8), )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20,
                            4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    lookuptable = rc.atom_order.copy()
    lookuptable[''] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names, )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx, )
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
        *((1, ) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape)

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        num_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        num_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Frame.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        num_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        num_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1, ) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(mat=rots)

    gt_frames = gt_frames.compose(Frame(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1, ) * batch_dims), 21, 8)
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device)
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1, ) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        num_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        num_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(
        mat=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(
        Frame(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    protein['rigidgroups_gt_frames'] = gt_frames_tensor
    protein['rigidgroups_gt_exists'] = gt_exists
    protein['rigidgroups_group_exists'] = group_exists
    protein['rigidgroups_group_is_ambiguous'] = residx_rigidgroup_is_ambiguous
    protein['rigidgroups_alt_gt_frames'] = alt_gt_frames_tensor

    return protein


@curry1
def atom37_to_torsion_angles(
    protein,
    prefix='',
):
    aatype = protein[prefix + 'aatype']
    all_atom_positions = protein[prefix + 'all_atom_positions']
    all_atom_mask = protein[prefix + 'all_atom_mask']
    if aatype.shape[-1] == 0:
        base_shape = aatype.shape
        protein[prefix
                + 'torsion_angles_sin_cos'] = all_atom_positions.new_zeros(
                    *base_shape, 7, 2)
        protein[prefix
                + 'alt_torsion_angles_sin_cos'] = all_atom_positions.new_zeros(
                    *base_shape, 7, 2)
        protein[prefix + 'torsion_angles_mask'] = all_atom_positions.new_zeros(
            *base_shape, 7)
        return protein

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3)

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
            all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4])

    chi_atom_indices = torch.as_tensor(
        rc.chi_atom_indices, device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2,
                                   len(atom_indices.shape[:-2]))

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        num_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Frame.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        ) + 1e-8)
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0], )[
            ((None, ) * len(torsion_angles_sin_cos.shape[:-2]))
            + (slice(None), None)])

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic, )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None])

    if prefix == '':
        # consistent to uni-fold. use [1, 0] placeholder
        placeholder_torsions = torch.stack(
            [
                torch.ones(torsion_angles_sin_cos.shape[:-1]),
                torch.zeros(torsion_angles_sin_cos.shape[:-1]),
            ],
            dim=-1,
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
            ...,
            None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
            ...,
            None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    protein[prefix + 'torsion_angles_sin_cos'] = torsion_angles_sin_cos
    protein[prefix + 'alt_torsion_angles_sin_cos'] = alt_torsion_angles_sin_cos
    protein[prefix + 'torsion_angles_mask'] = torsion_angles_mask

    return protein


def get_backbone_frames(protein):
    protein['true_frame_tensor'] = protein['rigidgroups_gt_frames'][...,
                                                                    0, :, :]
    protein['frame_mask'] = protein['rigidgroups_gt_exists'][..., 0]

    return protein


def get_chi_angles(protein):
    dtype = protein['all_atom_mask'].dtype
    protein['chi_angles_sin_cos'] = (
        protein['torsion_angles_sin_cos'][..., 3:, :]).to(dtype)
    protein['chi_mask'] = protein['torsion_angles_mask'][..., 3:].to(dtype)

    return protein


@curry1
def crop_templates(
    protein,
    max_templates,
    subsample_templates=False,
):
    if 'template_mask' in protein:
        num_templates = protein['template_mask'].shape[-1]
    else:
        num_templates = 0

    # don't sample when there are no templates
    if num_templates > 0:
        if subsample_templates:
            # af2's sampling, min(4, uniform[0, n])
            max_templates = min(max_templates,
                                np.random.randint(0, num_templates + 1))
            template_idx = torch.tensor(
                np.random.choice(num_templates, max_templates, replace=False),
                dtype=torch.int64,
            )
        else:
            # use top templates
            template_idx = torch.arange(
                min(num_templates, max_templates), dtype=torch.int64)
        for k, v in protein.items():
            if k.startswith('template'):
                try:
                    v = v[template_idx]
                except Exception as ex:
                    print(ex.__class__, ex)
                    print('num_templates', num_templates)
                    print(k, v.shape)
                    print('protein:', protein)
                    print(
                        'protein_shape:',
                        {
                            k: v.shape
                            for k, v in protein.items() if 'shape' in dir(v)
                        },
                    )
            protein[k] = v

    return protein


@curry1
def crop_to_size_single(protein, crop_size, shape_schema, seed):
    """crop to size."""
    num_res = (
        protein['aatype'].shape[0]
        if 'aatype' in protein else protein['msa_mask'].shape[1])
    crop_idx = get_single_crop_idx(num_res, crop_size, seed)
    protein = apply_crop_idx(protein, shape_schema, crop_idx)
    return protein


@curry1
def crop_to_size_multimer(protein, crop_size, shape_schema, seed,
                          spatial_crop_prob, ca_ca_threshold):
    """crop to size."""
    with data_utils.numpy_seed(seed, key='multimer_crop'):
        use_spatial_crop = np.random.rand() < spatial_crop_prob
    is_distillation = 'is_distillation' in protein and protein[
        'is_distillation'] == 1
    if is_distillation:
        return crop_to_size_single(
            crop_size=crop_size, shape_schema=shape_schema, seed=seed)(
                protein)
    elif use_spatial_crop:
        crop_idx = get_spatial_crop_idx(protein, crop_size, seed,
                                        ca_ca_threshold)
    else:
        crop_idx = get_contiguous_crop_idx(protein, crop_size, seed)
    return apply_crop_idx(protein, shape_schema, crop_idx)


def get_single_crop_idx(num_res: NumpyDict, crop_size: int,
                        random_seed: Optional[int]) -> torch.Tensor:

    if num_res < crop_size:
        return torch.arange(num_res)
    with data_utils.numpy_seed(random_seed):
        crop_start = int(np.random.randint(0, num_res - crop_size + 1))
        return torch.arange(crop_start, crop_start + crop_size)


def get_crop_sizes_each_chain(
    asym_len: torch.Tensor,
    crop_size: int,
    random_seed: Optional[int] = None,
    use_multinomial: bool = False,
) -> torch.Tensor:
    """get crop sizes for contiguous crop"""
    if not use_multinomial:
        with data_utils.numpy_seed(
                random_seed, key='multimer_contiguous_perm'):
            shuffle_idx = np.random.permutation(len(asym_len))
        num_left = asym_len.sum()
        num_budget = torch.tensor(crop_size)
        crop_sizes = [0 for _ in asym_len]
        for j, idx in enumerate(shuffle_idx):
            this_len = asym_len[idx]
            num_left -= this_len
            # num res at most we can keep in this ent
            max_size = min(num_budget, this_len)
            # num res at least we shall keep in this ent
            min_size = min(this_len, max(0, num_budget - num_left))
            with data_utils.numpy_seed(
                    random_seed, j, key='multimer_contiguous_crop_size'):
                this_crop_size = int(
                    np.random.randint(
                        low=int(min_size), high=int(max_size) + 1))
            num_budget -= this_crop_size
            crop_sizes[idx] = this_crop_size
        crop_sizes = torch.tensor(crop_sizes)
    else:  # use multinomial
        # TODO: better multimer
        entity_probs = asym_len / torch.sum(asym_len)
        crop_sizes = torch.from_numpy(
            np.random.multinomial(crop_size, pvals=entity_probs))
        crop_sizes = torch.min(crop_sizes, asym_len)
    return crop_sizes


def get_contiguous_crop_idx(
    protein: NumpyDict,
    crop_size: int,
    random_seed: Optional[int] = None,
    use_multinomial: bool = False,
) -> torch.Tensor:

    num_res = protein['aatype'].shape[0]
    if num_res <= crop_size:
        return torch.arange(num_res)

    assert 'asym_len' in protein
    asym_len = protein['asym_len']

    crop_sizes = get_crop_sizes_each_chain(asym_len, crop_size, random_seed,
                                           use_multinomial)
    crop_idxs = []
    asym_offset = torch.tensor(0, dtype=torch.int64)
    with data_utils.numpy_seed(
            random_seed, key='multimer_contiguous_crop_start_idx'):
        for ll, csz in zip(asym_len, crop_sizes):
            this_start = np.random.randint(0, int(ll - csz) + 1)
            crop_idxs.append(
                torch.arange(asym_offset + this_start,
                             asym_offset + this_start + csz))
            asym_offset += ll

    return torch.cat(crop_idxs)


def get_spatial_crop_idx(
    protein: NumpyDict,
    crop_size: int,
    random_seed: int,
    ca_ca_threshold: float,
    inf: float = 3e4,
) -> List[int]:

    ca_idx = rc.atom_order['CA']
    ca_coords = protein['all_atom_positions'][..., ca_idx, :]
    ca_mask = protein['all_atom_mask'][..., ca_idx].bool()
    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(dim=-1) <= 1).all():
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    ca_distances = get_pairwise_distances(ca_coords)

    interface_candidates = get_interface_candidates(ca_distances,
                                                    protein['asym_id'],
                                                    pair_mask, ca_ca_threshold)

    if torch.any(interface_candidates):
        with data_utils.numpy_seed(random_seed, key='multimer_spatial_crop'):
            target_res = int(np.random.choice(interface_candidates))
    else:
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    to_target_distances = ca_distances[target_res]
    # set inf to non-position residues
    to_target_distances[~ca_mask] = inf
    break_tie = (
        torch.arange(
            0,
            to_target_distances.shape[-1],
            device=to_target_distances.device).float() * 1e-3)
    to_target_distances += break_tie
    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values


def get_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    coord_diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)
    return torch.sqrt(torch.sum(coord_diff**2, dim=-1))


def get_interface_candidates(
    ca_distances: torch.Tensor,
    asym_id: torch.Tensor,
    pair_mask: torch.Tensor,
    ca_ca_threshold,
) -> torch.Tensor:

    in_same_asym = asym_id[..., None] == asym_id[..., None, :]
    # set distance in the same entity to zero
    ca_distances = ca_distances * (1.0 - in_same_asym.float()) * pair_mask
    cnt_interfaces = torch.sum(
        (ca_distances > 0) & (ca_distances < ca_ca_threshold), dim=-1)
    interface_candidates = cnt_interfaces.nonzero(as_tuple=True)[0]
    return interface_candidates


def apply_crop_idx(protein, shape_schema, crop_idx):
    cropped_protein = {}
    for k, v in protein.items():
        if k not in shape_schema:  # skip items with unknown shape schema
            continue
        for i, dim_size in enumerate(shape_schema[k]):
            if dim_size == N_RES:
                v = torch.index_select(v, i, crop_idx)
        cropped_protein[k] = v
    return cropped_protein
