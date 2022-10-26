# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from typing import Dict, Optional, Tuple

import torch


def predicted_lddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """Computes per-residue pLDDT from logits.
    Args:
        logits: [num_res, num_bins] output from the PredictedLDDTHead.
    Returns:
        plddt: [num_res] per-residue pLDDT.
    """
    num_bins = plddt_logits.shape[-1]
    bin_probs = torch.nn.functional.softmax(plddt_logits.float(), dim=-1)
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width,
        end=1.0,
        step=bin_width,
        device=plddt_logits.device)
    plddt = torch.sum(
        bin_probs
        * bounds.view(*((1, ) * len(bin_probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def compute_bin_values(breaks: torch.Tensor):
    """Gets the bin centers from the bin edges.
    Args:
        breaks: [num_bins - 1] the error bin edges.
    Returns:
        bin_centers: [num_bins] the error bin centers.
    """
    step = breaks[1] - breaks[0]
    bin_values = breaks + step / 2
    bin_values = torch.cat([bin_values, (bin_values[-1] + step).unsqueeze(-1)],
                           dim=0)
    return bin_values


def compute_predicted_aligned_error(
    bin_edges: torch.Tensor,
    bin_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates expected aligned distance errors for every pair of residues.
    Args:
        alignment_confidence_breaks: [num_bins - 1] the error bin edges.
        aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
        probs for each error bin, for each pair of residues.
    Returns:
        predicted_aligned_error: [num_res, num_res] the expected aligned distance
        error for each pair of residues.
        max_predicted_aligned_error: The maximum predicted error possible.
    """
    bin_values = compute_bin_values(bin_edges)
    return torch.sum(bin_probs * bin_values, dim=-1)


def predicted_aligned_error(
    pae_logits: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.
    Args:
        logits: [num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
        breaks: [num_bins - 1] the error bin edges.
    Returns:
        aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
        predicted_aligned_error: [num_res, num_res] the expected aligned distance
        error for each pair of residues.
        max_predicted_aligned_error: The maximum predicted error possible.
    """
    bin_probs = torch.nn.functional.softmax(pae_logits.float(), dim=-1)
    bin_edges = torch.linspace(
        0, max_bin, steps=(num_bins - 1), device=pae_logits.device)

    predicted_aligned_error = compute_predicted_aligned_error(
        bin_edges=bin_edges,
        bin_probs=bin_probs,
    )

    return {
        'aligned_error_probs_per_bin': bin_probs,
        'predicted_aligned_error': predicted_aligned_error,
    }


def predicted_tm_score(
    pae_logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-8,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Computes predicted TM alignment or predicted interface TM alignment score.
    Args:
        logits: [num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
        breaks: [num_bins] the error bins.
        residue_weights: [num_res] the per residue weights to use for the
        expectation.
        asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
        ipTM calculation, i.e. when interface=True.
        interface: If True, interface predicted TM score is computed.
    Returns:
        ptm_score: The predicted TM alignment or the predicted iTM score.
    """
    pae_logits = pae_logits.float()
    if residue_weights is None:
        residue_weights = pae_logits.new_ones(pae_logits.shape[:-2])

    breaks = torch.linspace(
        0, max_bin, steps=(num_bins - 1), device=pae_logits.device)

    def tm_kernal(nres):
        clipped_n = max(nres, 19)
        d0 = 1.24 * (clipped_n - 15)**(1.0 / 3.0) - 1.8
        return lambda x: 1.0 / (1.0 + (x / d0)**2)

    def rmsd_kernal(eps):  # leave for compute pRMS
        return lambda x: 1. / (x + eps)

    bin_centers = compute_bin_values(breaks)
    probs = torch.nn.functional.softmax(pae_logits, dim=-1)
    tm_per_bin = tm_kernal(nres=pae_logits.shape[-2])(bin_centers)
    # tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    # rmsd_per_bin = rmsd_kernal()(bin_centers)
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    pair_mask = predicted_tm_term.new_ones(predicted_tm_term.shape)
    if interface:
        assert asym_id is not None, 'must provide asym_id for iptm calculation.'
        pair_mask *= asym_id[..., :, None] != asym_id[..., None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (
        eps + pair_residue_weights.sum(dim=-1, keepdim=True))

    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    ret = per_alignment.gather(
        dim=-1, index=weighted.max(dim=-1,
                                   keepdim=True).indices).squeeze(dim=-1)
    return ret
