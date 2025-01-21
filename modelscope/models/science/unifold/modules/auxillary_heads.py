# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from typing import Dict

import torch.nn as nn
from unicore.modules import LayerNorm

from .common import Linear
from .confidence import (predicted_aligned_error, predicted_lddt,
                         predicted_tm_score)


class AuxiliaryHeads(nn.Module):

    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PredictedLDDTHead(**config['plddt'], )

        self.distogram = DistogramHead(**config['distogram'], )

        self.masked_msa = MaskedMSAHead(**config['masked_msa'], )

        if config.experimentally_resolved.enabled:
            self.experimentally_resolved = ExperimentallyResolvedHead(
                **config['experimentally_resolved'], )

        if config.pae.enabled:
            self.pae = PredictedAlignedErrorHead(**config.pae, )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        plddt_logits = self.plddt(outputs['sm']['single'])
        aux_out['plddt_logits'] = plddt_logits

        aux_out['plddt'] = predicted_lddt(plddt_logits.detach())

        distogram_logits = self.distogram(outputs['pair'])
        aux_out['distogram_logits'] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs['msa'])
        aux_out['masked_msa_logits'] = masked_msa_logits

        if self.config.experimentally_resolved.enabled:
            exp_res_logits = self.experimentally_resolved(outputs['single'])
            aux_out['experimentally_resolved_logits'] = exp_res_logits

        if self.config.pae.enabled:
            pae_logits = self.pae(outputs['pair'])
            aux_out['pae_logits'] = pae_logits
            pae_logits = pae_logits.detach()
            aux_out.update(
                predicted_aligned_error(
                    pae_logits,
                    **self.config.pae,
                ))
            aux_out['ptm'] = predicted_tm_score(
                pae_logits, interface=False, **self.config.pae)

            iptm_weight = self.config.pae.get('iptm_weight', 0.0)
            if iptm_weight > 0.0:
                aux_out['iptm'] = predicted_tm_score(
                    pae_logits,
                    interface=True,
                    asym_id=outputs['asym_id'],
                    **self.config.pae,
                )
                aux_out['iptm+ptm'] = (
                    iptm_weight * aux_out['iptm'] +  # noqa W504
                    (1.0 - iptm_weight) * aux_out['ptm'])

        return aux_out


class PredictedLDDTHead(nn.Module):

    def __init__(self, num_bins, d_in, d_hid):
        super(PredictedLDDTHead, self).__init__()

        self.num_bins = num_bins
        self.d_in = d_in
        self.d_hid = d_hid

        self.layer_norm = LayerNorm(self.d_in)

        self.linear_1 = Linear(self.d_in, self.d_hid, init='relu')
        self.linear_2 = Linear(self.d_hid, self.d_hid, init='relu')
        self.act = nn.GELU()
        self.linear_3 = Linear(self.d_hid, self.num_bins, init='final')

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.act(s)
        s = self.linear_2(s)
        s = self.act(s)
        s = self.linear_3(s)
        return s


class EnhancedHeadBase(nn.Module):

    def __init__(self, d_in, d_out, disable_enhance_head):
        super(EnhancedHeadBase, self).__init__()
        if disable_enhance_head:
            self.layer_norm = None
            self.linear_in = None
        else:
            self.layer_norm = LayerNorm(d_in)
            self.linear_in = Linear(d_in, d_in, init='relu')
        self.act = nn.GELU()
        self.linear = Linear(d_in, d_out, init='final')

    def apply_alphafold_original_mode(self):
        self.layer_norm = None
        self.linear_in = None

    def forward(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            x = self.act(self.linear_in(x))
        logits = self.linear(x)
        return logits


class DistogramHead(EnhancedHeadBase):

    def __init__(self, d_pair, num_bins, disable_enhance_head, **kwargs):
        super(DistogramHead, self).__init__(
            d_in=d_pair,
            d_out=num_bins,
            disable_enhance_head=disable_enhance_head,
        )

    def forward(self, x):
        logits = super().forward(x)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PredictedAlignedErrorHead(EnhancedHeadBase):

    def __init__(self, d_pair, num_bins, disable_enhance_head, **kwargs):
        super(PredictedAlignedErrorHead, self).__init__(
            d_in=d_pair,
            d_out=num_bins,
            disable_enhance_head=disable_enhance_head,
        )


class MaskedMSAHead(EnhancedHeadBase):

    def __init__(self, d_msa, d_out, disable_enhance_head, **kwargs):
        super(MaskedMSAHead, self).__init__(
            d_in=d_msa,
            d_out=d_out,
            disable_enhance_head=disable_enhance_head,
        )


class ExperimentallyResolvedHead(EnhancedHeadBase):

    def __init__(self, d_single, d_out, disable_enhance_head, **kwargs):
        super(ExperimentallyResolvedHead, self).__init__(
            d_in=d_single,
            d_out=d_out,
            disable_enhance_head=disable_enhance_head,
        )
