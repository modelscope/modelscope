# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from typing import Dict

import torch
import torch.nn as nn
from unicore.utils import batched_gather, one_hot

from modelscope.models.science.unifold.data import residue_constants as rc
from .frame import Frame


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order['G']
    ca_idx = rc.atom_order['CA']
    cb_idx = rc.atom_order['CB']
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1, ) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch['residx_atom37_to_atom14'],
        dim=-2,
        num_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch['atom37_atom_exists'][..., None]

    return atom37_data


def build_template_angle_feat(template_feats, v2_feature=False):
    template_aatype = template_feats['template_aatype']
    torsion_angles_sin_cos = template_feats['template_torsion_angles_sin_cos']
    torsion_angles_mask = template_feats['template_torsion_angles_mask']
    if not v2_feature:
        alt_torsion_angles_sin_cos = template_feats[
            'template_alt_torsion_angles_sin_cos']
        template_angle_feat = torch.cat(
            [
                one_hot(template_aatype, 22),
                torsion_angles_sin_cos.reshape(
                    *torsion_angles_sin_cos.shape[:-2], 14),
                alt_torsion_angles_sin_cos.reshape(
                    *alt_torsion_angles_sin_cos.shape[:-2], 14),
                torsion_angles_mask,
            ],
            dim=-1,
        )
        template_angle_mask = torsion_angles_mask[..., 2]
    else:
        chi_mask = torsion_angles_mask[..., 3:]
        chi_angles_sin = torsion_angles_sin_cos[..., 3:, 0] * chi_mask
        chi_angles_cos = torsion_angles_sin_cos[..., 3:, 1] * chi_mask
        template_angle_feat = torch.cat(
            [
                one_hot(template_aatype, 22),
                chi_angles_sin,
                chi_angles_cos,
                chi_mask,
            ],
            dim=-1,
        )
        template_angle_mask = chi_mask[..., 0]
    return template_angle_feat, template_angle_mask


def build_template_pair_feat(
    batch,
    min_bin,
    max_bin,
    num_bins,
    eps=1e-20,
    inf=1e8,
):
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    tpb = batch['template_pseudo_beta']
    dgram = torch.sum(
        (tpb[..., None, :] - tpb[..., None, :, :])**2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, num_bins, device=tpb.device)**2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = nn.functional.one_hot(
        batch['template_aatype'],
        rc.restype_num + 2,
    )

    n_res = batch['template_aatype'].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(
        *aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[...,
                                    None, :].expand(*aatype_one_hot.shape[:-2],
                                                    -1, n_res, -1))

    to_concat.append(template_mask_2d.new_zeros(*template_mask_2d.shape, 3))
    to_concat.append(template_mask_2d[..., None])

    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_template_pair_feat_v2(
    batch,
    min_bin,
    max_bin,
    num_bins,
    multichain_mask_2d=None,
    eps=1e-20,
    inf=1e8,
):
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    if multichain_mask_2d is not None:
        template_mask_2d *= multichain_mask_2d

    tpb = batch['template_pseudo_beta']
    dgram = torch.sum(
        (tpb[..., None, :] - tpb[..., None, :, :])**2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, num_bins, device=tpb.device)**2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    dgram *= template_mask_2d[..., None]
    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = one_hot(
        batch['template_aatype'],
        rc.restype_num + 2,
    )

    n_res = batch['template_aatype'].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(
        *aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[...,
                                    None, :].expand(*aatype_one_hot.shape[:-2],
                                                    -1, n_res, -1))

    n, ca, c = [rc.atom_order[a] for a in ['N', 'CA', 'C']]
    rigids = Frame.make_transform_from_reference(
        n_xyz=batch['template_all_atom_positions'][..., n, :],
        ca_xyz=batch['template_all_atom_positions'][..., ca, :],
        c_xyz=batch['template_all_atom_positions'][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

    t_aa_masks = batch['template_all_atom_mask']
    backbone_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[...,
                                                                          c]
    backbone_mask_2d = backbone_mask[..., :, None] * backbone_mask[...,
                                                                   None, :]
    if multichain_mask_2d is not None:
        backbone_mask_2d *= multichain_mask_2d

    inv_distance_scalar = inv_distance_scalar * backbone_mask_2d
    unit_vector_data = rigid_vec * inv_distance_scalar[..., None]
    to_concat.extend(torch.unbind(unit_vector_data[..., None, :], dim=-1))
    to_concat.append(backbone_mask_2d[..., None])

    return to_concat


def build_extra_msa_feat(batch):
    msa_1hot = one_hot(batch['extra_msa'], 23)
    msa_feat = [
        msa_1hot,
        batch['extra_msa_has_deletion'].unsqueeze(-1),
        batch['extra_msa_deletion_value'].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)
