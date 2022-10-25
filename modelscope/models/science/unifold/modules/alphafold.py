# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import torch
import torch.nn as nn
from unicore.utils import tensor_tree_map

from ..data import residue_constants
from .attentions import gen_msa_attn_mask, gen_tri_attn_mask
from .auxillary_heads import AuxiliaryHeads
from .common import residual
from .embedders import (ExtraMSAEmbedder, InputEmbedder, RecyclingEmbedder,
                        TemplateAngleEmbedder, TemplatePairEmbedder)
from .evoformer import EvoformerStack, ExtraMSAStack
from .featurization import (atom14_to_atom37, build_extra_msa_feat,
                            build_template_angle_feat,
                            build_template_pair_feat,
                            build_template_pair_feat_v2, pseudo_beta_fn)
from .structure_module import StructureModule
from .template import (TemplatePairStack, TemplatePointwiseAttention,
                       TemplateProjection)


class AlphaFold(nn.Module):

    def __init__(self, config):
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        config = config.model
        template_config = config.template
        extra_msa_config = config.extra_msa

        self.input_embedder = InputEmbedder(
            **config['input_embedder'],
            use_chain_relative=config.is_multimer,
        )
        self.recycling_embedder = RecyclingEmbedder(
            **config['recycling_embedder'], )
        if config.template.enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **template_config['template_angle_embedder'], )
            self.template_pair_embedder = TemplatePairEmbedder(
                **template_config['template_pair_embedder'], )
            self.template_pair_stack = TemplatePairStack(
                **template_config['template_pair_stack'], )
        else:
            self.template_pair_stack = None
        self.enable_template_pointwise_attention = template_config[
            'template_pointwise_attention'].enabled
        if self.enable_template_pointwise_attention:
            self.template_pointwise_att = TemplatePointwiseAttention(
                **template_config['template_pointwise_attention'], )
        else:
            self.template_proj = TemplateProjection(
                **template_config['template_pointwise_attention'], )
        self.extra_msa_embedder = ExtraMSAEmbedder(
            **extra_msa_config['extra_msa_embedder'], )
        self.extra_msa_stack = ExtraMSAStack(
            **extra_msa_config['extra_msa_stack'], )
        self.evoformer = EvoformerStack(**config['evoformer_stack'], )
        self.structure_module = StructureModule(**config['structure_module'], )

        self.aux_heads = AuxiliaryHeads(config['heads'], )

        self.config = config
        self.dtype = torch.float
        self.inf = self.globals.inf
        if self.globals.alphafold_original_mode:
            self.alphafold_original_mode()

    def __make_input_float__(self):
        self.input_embedder = self.input_embedder.float()
        self.recycling_embedder = self.recycling_embedder.float()

    def half(self):
        super().half()
        if (not getattr(self, 'inference', False)):
            self.__make_input_float__()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        if (not getattr(self, 'inference', False)):
            self.__make_input_float__()
        self.dtype = torch.bfloat16
        return self

    def alphafold_original_mode(self):

        def set_alphafold_original_mode(module):
            if hasattr(module, 'apply_alphafold_original_mode'):
                module.apply_alphafold_original_mode()
            if hasattr(module, 'act'):
                module.act = nn.ReLU()

        self.apply(set_alphafold_original_mode)

    def inference_mode(self):

        def set_inference_mode(module):
            setattr(module, 'inference', True)

        self.apply(set_inference_mode)

    def __convert_input_dtype__(self, batch):
        for key in batch:
            # only convert features with mask
            if batch[key].dtype != self.dtype and 'mask' in key:
                batch[key] = batch[key].type(self.dtype)
        return batch

    def embed_templates_pair_core(self, batch, z, pair_mask,
                                  tri_start_attn_mask, tri_end_attn_mask,
                                  templ_dim, multichain_mask_2d):
        if self.config.template.template_pair_embedder.v2_feature:
            t = build_template_pair_feat_v2(
                batch,
                inf=self.config.template.inf,
                eps=self.config.template.eps,
                multichain_mask_2d=multichain_mask_2d,
                **self.config.template.distogram,
            )
            num_template = t[0].shape[-4]
            single_templates = [
                self.template_pair_embedder([x[..., ti, :, :, :]
                                             for x in t], z)
                for ti in range(num_template)
            ]
        else:
            t = build_template_pair_feat(
                batch,
                inf=self.config.template.inf,
                eps=self.config.template.eps,
                **self.config.template.distogram,
            )
            single_templates = [
                self.template_pair_embedder(x, z)
                for x in torch.unbind(t, dim=templ_dim)
            ]

        t = self.template_pair_stack(
            single_templates,
            pair_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            templ_dim=templ_dim,
            chunk_size=self.globals.chunk_size,
            block_size=self.globals.block_size,
            return_mean=not self.enable_template_pointwise_attention,
        )
        return t

    def embed_templates_pair(self, batch, z, pair_mask, tri_start_attn_mask,
                             tri_end_attn_mask, templ_dim):
        if self.config.template.template_pair_embedder.v2_feature and 'asym_id' in batch:
            multichain_mask_2d = (
                batch['asym_id'][..., :, None] == batch['asym_id'][...,
                                                                   None, :])
            multichain_mask_2d = multichain_mask_2d.unsqueeze(0)
        else:
            multichain_mask_2d = None

        if self.training or self.enable_template_pointwise_attention:
            t = self.embed_templates_pair_core(batch, z, pair_mask,
                                               tri_start_attn_mask,
                                               tri_end_attn_mask, templ_dim,
                                               multichain_mask_2d)
            if self.enable_template_pointwise_attention:
                t = self.template_pointwise_att(
                    t,
                    z,
                    template_mask=batch['template_mask'],
                    chunk_size=self.globals.chunk_size,
                )
                t_mask = torch.sum(
                    batch['template_mask'], dim=-1, keepdims=True) > 0
                t_mask = t_mask[..., None, None].type(t.dtype)
                t *= t_mask
            else:
                t = self.template_proj(t, z)
        else:
            template_aatype_shape = batch['template_aatype'].shape
            # template_aatype is either [n_template, n_res] or [1, n_template_, n_res]
            batch_templ_dim = 1 if len(template_aatype_shape) == 3 else 0
            n_templ = batch['template_aatype'].shape[batch_templ_dim]

            if n_templ <= 0:
                t = None
            else:
                template_batch = {
                    k: v
                    for k, v in batch.items() if k.startswith('template_')
                }

                def embed_one_template(i):

                    def slice_template_tensor(t):
                        s = [slice(None) for _ in t.shape]
                        s[batch_templ_dim] = slice(i, i + 1)
                        return t[s]

                    template_feats = tensor_tree_map(
                        slice_template_tensor,
                        template_batch,
                    )
                    t = self.embed_templates_pair_core(
                        template_feats, z, pair_mask, tri_start_attn_mask,
                        tri_end_attn_mask, templ_dim, multichain_mask_2d)
                    return t

                t = embed_one_template(0)
                # iterate templates one by one
                for i in range(1, n_templ):
                    t += embed_one_template(i)
                t /= n_templ
            t = self.template_proj(t, z)
        return t

    def embed_templates_angle(self, batch):
        template_angle_feat, template_angle_mask = build_template_angle_feat(
            batch,
            v2_feature=self.config.template.template_pair_embedder.v2_feature)
        t = self.template_angle_embedder(template_angle_feat)
        return t, template_angle_mask

    def iteration_evoformer(self, feats, m_1_prev, z_prev, x_prev):
        batch_dims = feats['target_feat'].shape[:-2]
        n = feats['target_feat'].shape[-2]
        seq_mask = feats['seq_mask']
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats['msa_mask']

        m, z = self.input_embedder(
            feats['target_feat'],
            feats['msa_feat'],
        )

        if m_1_prev is None:
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.d_msa),
                requires_grad=False,
            )
        if z_prev is None:
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.d_pair),
                requires_grad=False,
            )
        if x_prev is None:
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )
        x_prev = pseudo_beta_fn(feats['aatype'], x_prev, None)

        z += self.recycling_embedder.recyle_pos(x_prev)

        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
        )

        m[..., 0, :, :] += m_1_prev_emb

        z += z_prev_emb

        z += self.input_embedder.relpos_emb(
            feats['residue_index'].long(),
            feats.get('sym_id', None),
            feats.get('asym_id', None),
            feats.get('entity_id', None),
            feats.get('num_sym', None),
        )

        m = m.type(self.dtype)
        z = z.type(self.dtype)
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(
            pair_mask, self.inf)

        if self.config.template.enabled:
            template_mask = feats['template_mask']
            if torch.any(template_mask):
                z = residual(
                    z,
                    self.embed_templates_pair(
                        feats,
                        z,
                        pair_mask,
                        tri_start_attn_mask,
                        tri_end_attn_mask,
                        templ_dim=-4,
                    ),
                    self.training,
                )

        if self.config.extra_msa.enabled:
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))
            extra_msa_row_mask = gen_msa_attn_mask(
                feats['extra_msa_mask'],
                inf=self.inf,
                gen_col_mask=False,
            )
            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats['extra_msa_mask'],
                chunk_size=self.globals.chunk_size,
                block_size=self.globals.block_size,
                pair_mask=pair_mask,
                msa_row_attn_mask=extra_msa_row_mask,
                msa_col_attn_mask=None,
                tri_start_attn_mask=tri_start_attn_mask,
                tri_end_attn_mask=tri_end_attn_mask,
            )

        if self.config.template.embed_angles:
            template_1d_feat, template_1d_mask = self.embed_templates_angle(
                feats)
            m = torch.cat([m, template_1d_feat], dim=-3)
            msa_mask = torch.cat([feats['msa_mask'], template_1d_mask], dim=-2)

        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )

        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            msa_row_attn_mask=msa_row_mask,
            msa_col_attn_mask=msa_col_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            chunk_size=self.globals.chunk_size,
            block_size=self.globals.block_size,
        )
        return m, z, s, msa_mask, m_1_prev_emb, z_prev_emb

    def iteration_evoformer_structure_module(self,
                                             batch,
                                             m_1_prev,
                                             z_prev,
                                             x_prev,
                                             cycle_no,
                                             num_recycling,
                                             num_ensembles=1):
        z, s = 0, 0
        n_seq = batch['msa_feat'].shape[-3]
        assert num_ensembles >= 1
        for ensemble_no in range(num_ensembles):
            idx = cycle_no * num_ensembles + ensemble_no

            # fetch_cur_batch = lambda t: t[min(t.shape[0] - 1, idx), ...]
            def fetch_cur_batch(t):
                return t[min(t.shape[0] - 1, idx), ...]

            feats = tensor_tree_map(fetch_cur_batch, batch)
            m, z0, s0, msa_mask, m_1_prev_emb, z_prev_emb = self.iteration_evoformer(
                feats, m_1_prev, z_prev, x_prev)
            z += z0
            s += s0
            del z0, s0
        if num_ensembles > 1:
            z /= float(num_ensembles)
            s /= float(num_ensembles)

        outputs = {}

        outputs['msa'] = m[..., :n_seq, :, :]
        outputs['pair'] = z
        outputs['single'] = s

        # norm loss
        if (not getattr(self, 'inference',
                        False)) and num_recycling == (cycle_no + 1):
            delta_msa = m
            delta_msa[...,
                      0, :, :] = delta_msa[...,
                                           0, :, :] - m_1_prev_emb.detach()
            delta_pair = z - z_prev_emb.detach()
            outputs['delta_msa'] = delta_msa
            outputs['delta_pair'] = delta_pair
            outputs['msa_norm_mask'] = msa_mask

        outputs['sm'] = self.structure_module(
            s,
            z,
            feats['aatype'],
            mask=feats['seq_mask'],
        )
        outputs['final_atom_positions'] = atom14_to_atom37(
            outputs['sm']['positions'], feats)
        outputs['final_atom_mask'] = feats['atom37_atom_exists']
        outputs['pred_frame_tensor'] = outputs['sm']['frames'][-1]

        # use float32 for numerical stability
        if (not getattr(self, 'inference', False)):
            m_1_prev = m[..., 0, :, :].float()
            z_prev = z.float()
            x_prev = outputs['final_atom_positions'].float()
        else:
            m_1_prev = m[..., 0, :, :]
            z_prev = z
            x_prev = outputs['final_atom_positions']

        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):

        m_1_prev = batch.get('m_1_prev', None)
        z_prev = batch.get('z_prev', None)
        x_prev = batch.get('x_prev', None)

        is_grad_enabled = torch.is_grad_enabled()

        num_iters = int(batch['num_recycling_iters']) + 1
        num_ensembles = int(batch['msa_mask'].shape[0]) // num_iters
        if self.training:
            # don't use ensemble during training
            assert num_ensembles == 1

        # convert dtypes in batch
        batch = self.__convert_input_dtype__(batch)
        for cycle_no in range(num_iters):
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                (
                    outputs,
                    m_1_prev,
                    z_prev,
                    x_prev,
                ) = self.iteration_evoformer_structure_module(
                    batch,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    cycle_no=cycle_no,
                    num_recycling=num_iters,
                    num_ensembles=num_ensembles,
                )
            if not is_final_iter:
                del outputs

        if 'asym_id' in batch:
            outputs['asym_id'] = batch['asym_id'][0, ...]
        outputs.update(self.aux_heads(outputs))
        return outputs
