# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from typing import Optional

import numpy as np
import torch

from modelscope.models.science.unifold.data import data_ops


def nonensembled_fns(common_cfg, mode_cfg):
    """Input pipeline data transformers that are not ensembled."""
    v2_feature = common_cfg.v2_feature
    operators = []
    if mode_cfg.random_delete_msa:
        operators.append(
            data_ops.random_delete_msa(common_cfg.random_delete_msa))
    operators.extend([
        data_ops.cast_to_64bit_ints,
        data_ops.correct_msa_restypes,
        data_ops.squeeze_features,
        data_ops.randomly_replace_msa_with_unknown(0.0),
        data_ops.make_seq_mask,
        data_ops.make_msa_mask,
    ])
    operators.append(data_ops.make_hhblits_profile_v2
                     if v2_feature else data_ops.make_hhblits_profile)
    if common_cfg.use_templates:
        operators.extend([
            data_ops.make_template_mask,
            data_ops.make_pseudo_beta('template_'),
        ])
        operators.append(
            data_ops.crop_templates(
                max_templates=mode_cfg.max_templates,
                subsample_templates=mode_cfg.subsample_templates,
            ))

    if common_cfg.use_template_torsion_angles:
        operators.extend([
            data_ops.atom37_to_torsion_angles('template_'),
        ])

    operators.append(data_ops.make_atom14_masks)
    operators.append(data_ops.make_target_feat)

    return operators


def crop_and_fix_size_fns(common_cfg, mode_cfg, crop_and_fix_size_seed):
    operators = []
    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters
    crop_feats = dict(common_cfg.features)
    if mode_cfg.fixed_size:
        if mode_cfg.crop:
            if common_cfg.is_multimer:
                crop_fn = data_ops.crop_to_size_multimer(
                    crop_size=mode_cfg.crop_size,
                    shape_schema=crop_feats,
                    seed=crop_and_fix_size_seed,
                    spatial_crop_prob=mode_cfg.spatial_crop_prob,
                    ca_ca_threshold=mode_cfg.ca_ca_threshold,
                )
            else:
                crop_fn = data_ops.crop_to_size_single(
                    crop_size=mode_cfg.crop_size,
                    shape_schema=crop_feats,
                    seed=crop_and_fix_size_seed,
                )
            operators.append(crop_fn)

        operators.append(data_ops.select_feat(crop_feats))

        operators.append(
            data_ops.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                common_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            ))
    return operators


def ensembled_fns(common_cfg, mode_cfg):
    """Input pipeline data transformers that can be ensembled and averaged."""
    operators = []
    multimer_mode = common_cfg.is_multimer
    v2_feature = common_cfg.v2_feature
    # multimer don't use block delete msa
    if mode_cfg.block_delete_msa and not multimer_mode:
        operators.append(
            data_ops.block_delete_msa(common_cfg.block_delete_msa))
    if 'max_distillation_msa_clusters' in mode_cfg:
        operators.append(
            data_ops.sample_msa_distillation(
                mode_cfg.max_distillation_msa_clusters))

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa

    assert common_cfg.resample_msa_in_recycling
    gumbel_sample = common_cfg.gumbel_sample
    operators.append(
        data_ops.sample_msa(
            max_msa_clusters,
            keep_extra=True,
            gumbel_sample=gumbel_sample,
            biased_msa_by_chain=mode_cfg.biased_msa_by_chain,
        ))

    if 'masked_msa' in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        operators.append(
            data_ops.make_masked_msa(
                common_cfg.masked_msa,
                mode_cfg.masked_msa_replace_fraction,
                gumbel_sample=gumbel_sample,
                share_mask=mode_cfg.share_mask,
            ))

    if common_cfg.msa_cluster_features:
        if v2_feature:
            operators.append(data_ops.nearest_neighbor_clusters_v2())
        else:
            operators.append(data_ops.nearest_neighbor_clusters())
            operators.append(data_ops.summarize_clusters)

    if v2_feature:
        operators.append(data_ops.make_msa_feat_v2)
    else:
        operators.append(data_ops.make_msa_feat)
    # Crop after creating the cluster profiles.
    if max_extra_msa:
        if v2_feature:
            operators.append(data_ops.make_extra_msa_feat(max_extra_msa))
        else:
            operators.append(data_ops.crop_extra_msa(max_extra_msa))
    else:
        operators.append(data_ops.delete_extra_msa)
    # operators.append(data_operators.select_feat(common_cfg.recycling_features))
    return operators


def process_features(tensors, common_cfg, mode_cfg):
    """Based on the config, apply filters and transformations to the data."""
    is_distillation = bool(tensors.get('is_distillation', 0))
    multimer_mode = common_cfg.is_multimer
    crop_and_fix_size_seed = int(tensors['crop_and_fix_size_seed'])
    crop_fn = crop_and_fix_size_fns(
        common_cfg,
        mode_cfg,
        crop_and_fix_size_seed,
    )

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_fns(
            common_cfg,
            mode_cfg,
        )
        new_d = compose(fns)(d)
        if not multimer_mode or is_distillation:
            new_d = data_ops.select_feat(common_cfg.recycling_features)(new_d)
            return compose(crop_fn)(new_d)
        else:  # select after crop for spatial cropping
            d = compose(crop_fn)(d)
            d = data_ops.select_feat(common_cfg.recycling_features)(d)
            return d

    nonensembled = nonensembled_fns(common_cfg, mode_cfg)

    if mode_cfg.supervised and (not multimer_mode or is_distillation):
        nonensembled.extend(label_transform_fn())

    tensors = compose(nonensembled)(tensors)

    num_recycling = int(tensors['num_recycling_iters']) + 1
    num_ensembles = mode_cfg.num_ensembles

    ensemble_tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x),
        torch.arange(num_recycling * num_ensembles),
    )
    tensors = compose(crop_fn)(tensors)
    # add a dummy dim to align with recycling features
    tensors = {k: torch.stack([tensors[k]], dim=0) for k in tensors}
    tensors.update(ensemble_tensors)
    return tensors


@data_ops.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def pad_then_stack(values, ):
    if len(values[0].shape) >= 1:
        size = max(v.shape[0] for v in values)
        new_values = []
        for v in values:
            if v.shape[0] < size:
                res = values[0].new_zeros(size, *v.shape[1:])
                res[:v.shape[0], ...] = v
            else:
                res = v
            new_values.append(res)
    else:
        new_values = values
    return torch.stack(new_values, dim=0)


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = pad_then_stack(
            [dict_i[feat] for dict_i in ensembles])
    return ensembled_dict


def process_single_label(label: dict,
                         num_ensemble: Optional[int] = None) -> dict:
    assert 'aatype' in label
    assert 'all_atom_positions' in label
    assert 'all_atom_mask' in label
    label = compose(label_transform_fn())(label)
    if num_ensemble is not None:
        label = {
            k: torch.stack([v for _ in range(num_ensemble)])
            for k, v in label.items()
        }
    return label


def process_labels(labels_list, num_ensemble: Optional[int] = None):
    return [process_single_label(ll, num_ensemble) for ll in labels_list]


def label_transform_fn():
    return [
        data_ops.make_atom14_masks,
        data_ops.make_atom14_positions,
        data_ops.atom37_to_frames,
        data_ops.atom37_to_torsion_angles(''),
        data_ops.make_pseudo_beta(''),
        data_ops.get_backbone_frames,
        data_ops.get_chi_angles,
    ]
