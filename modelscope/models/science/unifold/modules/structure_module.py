# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import math
from typing import Tuple

import torch
import torch.nn as nn
from unicore.modules import LayerNorm, softmax_dropout
from unicore.utils import dict_multimap, one_hot, permute_final_dims

from modelscope.models.science.unifold.data.residue_constants import (
    restype_atom14_mask, restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group, restype_rigid_group_default_frame)
from .attentions import gen_attn_mask
from .common import Linear, SimpleModuleList, residual
from .frame import Frame, Quaternion, Rotation


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def torsion_angles_to_frames(
    frame: Frame,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
):
    default_frame = Frame.from_tensor_4x4(default_frames[aatype, ...])

    bb_rot = alpha.new_zeros((*((1, ) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha],
                      dim=-2)

    all_rots = alpha.new_zeros(default_frame.get_rots().rot_mat.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Frame(Rotation(mat=all_rots), None)

    all_frames = default_frame.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Frame.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = frame[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    frame: Frame,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    group_mask = group_idx[aatype, ...]
    group_mask = one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    t_atoms_to_global = frame[..., None, :] * group_mask
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1))

    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


class SideChainAngleResnetIteration(nn.Module):

    def __init__(self, d_hid):
        super(SideChainAngleResnetIteration, self).__init__()

        self.d_hid = d_hid

        self.linear_1 = Linear(self.d_hid, self.d_hid, init='relu')
        self.act = nn.GELU()
        self.linear_2 = Linear(self.d_hid, self.d_hid, init='final')

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = self.act(s)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return residual(s, x, self.training)


class SidechainAngleResnet(nn.Module):

    def __init__(self, d_in, d_hid, num_blocks, num_angles):
        super(SidechainAngleResnet, self).__init__()

        self.linear_in = Linear(d_in, d_hid)
        self.act = nn.GELU()
        self.linear_initial = Linear(d_in, d_hid)

        self.layers = SimpleModuleList()
        for _ in range(num_blocks):
            self.layers.append(SideChainAngleResnetIteration(d_hid=d_hid))

        self.linear_out = Linear(d_hid, num_angles * 2)

    def forward(self, s: torch.Tensor,
                initial_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        initial_s = self.linear_initial(self.act(initial_s))
        s = self.linear_in(self.act(s))

        s = s + initial_s

        for layer in self.layers:
            s = layer(s)

        s = self.linear_out(self.act(s))

        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s.float()**2, dim=-1, keepdim=True),
                min=1e-12,
            ))
        s = s.float() / norm_denom

        return unnormalized_s, s.type(unnormalized_s.dtype)


class InvariantPointAttention(nn.Module):

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        separate_kv: bool = False,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super(InvariantPointAttention, self).__init__()

        self.d_hid = d_hid
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps

        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, bias=bias)
        self.separate_kv = separate_kv
        if self.separate_kv:
            self.linear_k = Linear(d_single, hc, bias=bias)
            self.linear_v = Linear(d_single, hc, bias=bias)
        else:
            self.linear_kv = Linear(d_single, 2 * hc, bias=bias)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(d_single, hpq)
        hpk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        if self.separate_kv:
            self.linear_k_points = Linear(d_single, hpk)
            self.linear_v_points = Linear(d_single, hpv)
        else:
            hpkv = hpk + hpv
            self.linear_kv_points = Linear(d_single, hpkv)

        self.linear_b = Linear(d_pair, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.num_heads * (
            d_pair + self.d_hid + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, d_single, init='final')

        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        f: Frame,
        square_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.linear_q(s)

        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        if self.separate_kv:
            k = self.linear_k(s)
            v = self.linear_v(s)
            k = k.view(k.shape[:-1] + (self.num_heads, -1))
            v = v.view(v.shape[:-1] + (self.num_heads, -1))
        else:
            kv = self.linear_kv(s)
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
            k, v = torch.split(kv, self.d_hid, dim=-1)

        q_pts = self.linear_q_points(s)

        def process_points(pts, no_points):
            shape = pts.shape[:-1] + (pts.shape[-1] // 3, 3)
            if self.separate_kv:
                # alphafold-multimer uses different layout
                pts = pts.view(pts.shape[:-1]
                               + (self.num_heads, no_points * 3))
            pts = torch.split(pts, pts.shape[-1] // 3, dim=-1)
            pts = torch.stack(pts, dim=-1).view(*shape)
            pts = f[..., None].apply(pts)

            pts = pts.view(pts.shape[:-2] + (self.num_heads, no_points, 3))
            return pts

        q_pts = process_points(q_pts, self.num_qk_points)

        if self.separate_kv:
            k_pts = self.linear_k_points(s)
            v_pts = self.linear_v_points(s)
            k_pts = process_points(k_pts, self.num_qk_points)
            v_pts = process_points(v_pts, self.num_v_points)
        else:
            kv_pts = self.linear_kv_points(s)

            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = f[..., None].apply(kv_pts)

            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3))

            k_pts, v_pts = torch.split(
                kv_pts, [self.num_qk_points, self.num_v_points], dim=-2)

        bias = self.linear_b(z)

        attn = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )

        if self.training:
            attn = attn * math.sqrt(1.0 / (3 * self.d_hid))
            attn = attn + (
                math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1)))
            pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att = pt_att.float()**2
        else:
            attn *= math.sqrt(1.0 / (3 * self.d_hid))
            attn += (math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1)))
            pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att *= pt_att

        pt_att = pt_att.sum(dim=-1)
        head_weights = self.softplus(self.head_weights).view(  # noqa
            *((1, ) * len(pt_att.shape[:-2]) + (-1, 1)))  # noqa
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.num_qk_points * 9.0 / 2)))
        pt_att *= head_weights * (-0.5)

        pt_att = torch.sum(pt_att, dim=-1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        attn += square_mask
        attn = softmax_dropout(
            attn, 0, self.training, bias=pt_att.type(attn.dtype))
        del pt_att, q_pts, k_pts, bias
        o = torch.matmul(attn, v.transpose(-2, -3)).transpose(-2, -3)
        o = o.contiguous().view(*o.shape[:-2], -1)
        del q, k, v
        o_pts = torch.sum(
            (attn[..., None, :, :, None]
             * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        o_pts = permute_final_dims(o_pts, (2, 0, 3, 1))
        o_pts = f[..., None, None].invert_apply(o_pts)
        if self.training:
            o_pts_norm = torch.sqrt(
                torch.sum(o_pts.float()**2, dim=-1) + self.eps).type(
                    o_pts.dtype)
        else:
            o_pts_norm = torch.sqrt(torch.sum(o_pts**2, dim=-1)
                                    + self.eps).type(o_pts.dtype)

        o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1)

        o_pts = o_pts.view(*o_pts.shape[:-3], -1, 3)

        o_pair = torch.matmul(attn.transpose(-2, -3), z)

        o_pair = o_pair.view(*o_pair.shape[:-2], -1)

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pts, dim=-1), o_pts_norm, o_pair),
                      dim=-1))

        return s


class BackboneUpdate(nn.Module):

    def __init__(self, d_single):
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(d_single, 6, init='final')

    def forward(self, s: torch.Tensor):
        return self.linear(s)


class StructureModuleTransitionLayer(nn.Module):

    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.linear_1 = Linear(c, c, init='relu')
        self.linear_2 = Linear(c, c, init='relu')
        self.act = nn.GELU()
        self.linear_3 = Linear(c, c, init='final')

    def forward(self, s):
        s_old = s
        s = self.linear_1(s)
        s = self.act(s)
        s = self.linear_2(s)
        s = self.act(s)
        s = self.linear_3(s)

        s = residual(s_old, s, self.training)

        return s


class StructureModuleTransition(nn.Module):

    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = SimpleModuleList()
        for _ in range(self.num_layers):
            self.layers.append(StructureModuleTransitionLayer(c))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(c)

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):

    def __init__(
        self,
        d_single,
        d_pair,
        d_ipa,
        d_angle,
        num_heads_ipa,
        num_qk_points,
        num_v_points,
        dropout_rate,
        num_blocks,
        no_transition_layers,
        num_resnet_blocks,
        num_angles,
        trans_scale_factor,
        separate_kv,
        ipa_bias,
        epsilon,
        inf,
        **kwargs,
    ):
        super(StructureModule, self).__init__()

        self.num_blocks = num_blocks
        self.trans_scale_factor = trans_scale_factor
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None
        self.inf = inf

        self.layer_norm_s = LayerNorm(d_single)
        self.layer_norm_z = LayerNorm(d_pair)

        self.linear_in = Linear(d_single, d_single)

        self.ipa = InvariantPointAttention(
            d_single,
            d_pair,
            d_ipa,
            num_heads_ipa,
            num_qk_points,
            num_v_points,
            separate_kv=separate_kv,
            bias=ipa_bias,
            eps=epsilon,
        )

        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(d_single)

        self.transition = StructureModuleTransition(
            d_single,
            no_transition_layers,
            dropout_rate,
        )

        self.bb_update = BackboneUpdate(d_single)

        self.angle_resnet = SidechainAngleResnet(
            d_single,
            d_angle,
            num_resnet_blocks,
            num_angles,
        )

    def forward(
        self,
        s,
        z,
        aatype,
        mask=None,
    ):
        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        # generate square mask
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = gen_attn_mask(square_mask, -self.inf).unsqueeze(-3)
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        initial_s = s
        s = self.linear_in(s)

        quat_encoder = Quaternion.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            requires_grad=False,
        )
        backb_to_global = Frame(
            Rotation(mat=quat_encoder.get_rot_mats(), ),
            quat_encoder.get_trans(),
        )
        outputs = []
        for i in range(self.num_blocks):
            s = residual(s, self.ipa(s, z, backb_to_global, square_mask),
                         self.training)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # update quaternion encoder
            # use backb_to_global to avoid quat-to-rot conversion
            quat_encoder = quat_encoder.compose_update_vec(
                self.bb_update(s), pre_rot_mat=backb_to_global.get_rots())

            # initial_s is always used to update the backbone
            unnormalized_angles, angles = self.angle_resnet(s, initial_s)

            # convert quaternion to rotation matrix
            backb_to_global = Frame(
                Rotation(mat=quat_encoder.get_rot_mats(), ),
                quat_encoder.get_trans(),
            )
            if i == self.num_blocks - 1:
                all_frames_to_global = self.torsion_angles_to_frames(
                    backb_to_global.scale_translation(self.trans_scale_factor),
                    angles,
                    aatype,
                )

                pred_positions = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype,
                )

            preds = {
                'frames':
                backb_to_global.scale_translation(
                    self.trans_scale_factor).to_tensor_4x4(),
                'unnormalized_angles':
                unnormalized_angles,
                'angles':
                angles,
            }

            outputs.append(preds)
            if i < (self.num_blocks - 1):
                # stop gradient in iteration
                quat_encoder = quat_encoder.stop_rot_gradient()
                backb_to_global = backb_to_global.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs['sidechain_frames'] = all_frames_to_global.to_tensor_4x4()
        outputs['positions'] = pred_positions
        outputs['single'] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, frame, alpha, aatype):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(frame, alpha, aatype,
                                        self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, frame, aatype):
        self._init_residue_constants(frame.get_rots().dtype,
                                     frame.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            frame,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
