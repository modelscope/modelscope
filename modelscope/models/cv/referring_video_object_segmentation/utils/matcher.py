# The implementation is adopted from MTTR,
# made publicly available under the Apache 2.0 License at https://github.com/mttr2021/MTTR
# Modified from DETR https://github.com/facebookresearch/detr
# Module to compute the matching cost and solve the corresponding LSAP.

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .misc import interpolate, nested_tensor_from_tensor_list


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_is_referred: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_is_referred: This is the relative weight of the reference cost in the total matching cost
            cost_dice: This is the relative weight of the dice cost in the total matching cost
        """
        super().__init__()
        self.cost_is_referred = cost_is_referred
        self.cost_dice = cost_dice
        assert cost_is_referred != 0 or cost_dice != 0, 'all costs cant be 0'

    @torch.inference_mode()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: A dict that contains at least these entries:
                 "pred_is_referred": Tensor of dim [time, batch_size, num_queries, 2] with the reference logits
                 "pred_masks": Tensor of dim [time, batch_size, num_queries, H, W] with the predicted masks logits

            targets: A list of lists of targets (outer - time steps, inner - batch samples). each target is a dict
                     which contain mask and reference ground truth information for a single frame.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_masks)
        """
        t, bs, num_queries = outputs['pred_masks'].shape[:3]

        # We flatten to compute the cost matrices in a batch
        out_masks = outputs['pred_masks'].flatten(
            1, 2)  # [t, batch_size * num_queries, mask_h, mask_w]

        # preprocess and concat the target masks
        tgt_masks = [[
            m for v in t_step_batch for m in v['masks'].unsqueeze(1)
        ] for t_step_batch in targets]
        # pad the target masks to a uniform shape
        tgt_masks, valid = list(
            zip(*[
                nested_tensor_from_tensor_list(t).decompose()
                for t in tgt_masks
            ]))
        tgt_masks = torch.stack(tgt_masks).squeeze(2)

        # upsample predicted masks to target mask size
        out_masks = interpolate(
            out_masks,
            size=tgt_masks.shape[-2:],
            mode='bilinear',
            align_corners=False)

        # Compute the soft-tokens cost:
        if self.cost_is_referred > 0:
            cost_is_referred = compute_is_referred_cost(outputs, targets)
        else:
            cost_is_referred = 0

        # Compute the DICE coefficient between the masks:
        if self.cost_dice > 0:
            cost_dice = -dice_coef(out_masks, tgt_masks)
        else:
            cost_dice = 0

        # Final cost matrix
        C = self.cost_is_referred * cost_is_referred + self.cost_dice * cost_dice
        C = C.view(bs, num_queries, -1).cpu()

        num_traj_per_batch = [
            len(v['masks']) for v in targets[0]
        ]  # number of instance trajectories in each batch
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(num_traj_per_batch, -1))
        ]
        device = out_masks.device
        return [(torch.as_tensor(i, dtype=torch.int64, device=device),
                 torch.as_tensor(j, dtype=torch.int64, device=device))
                for i, j in indices]


def dice_coef(inputs, targets, smooth=1.0):
    """
    Compute the DICE coefficient, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(2).unsqueeze(2)
    targets = targets.flatten(2).unsqueeze(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    coef = (numerator + smooth) / (denominator + smooth)
    coef = coef.mean(
        0)  # average on the temporal dim to get instance trajectory scores
    return coef


def compute_is_referred_cost(outputs, targets):
    pred_is_referred = outputs['pred_is_referred'].flatten(1, 2).softmax(
        dim=-1)  # [t, b*nq, 2]
    device = pred_is_referred.device
    t = pred_is_referred.shape[0]
    # number of instance trajectories in each batch
    num_traj_per_batch = torch.tensor([len(v['masks']) for v in targets[0]],
                                      device=device)
    total_trajectories = num_traj_per_batch.sum()
    # note that ref_indices are shared across time steps:
    ref_indices = torch.tensor(
        [v['referred_instance_idx'] for v in targets[0]], device=device)
    # convert ref_indices to fit flattened batch targets:
    ref_indices += torch.cat(
        (torch.zeros(1, dtype=torch.long,
                     device=device), num_traj_per_batch.cumsum(0)[:-1]))
    # number of instance trajectories in each batch
    target_is_referred = torch.zeros((t, total_trajectories, 2), device=device)
    # 'no object' class by default (for un-referred objects)
    target_is_referred[:, :, :] = torch.tensor([0.0, 1.0], device=device)
    if 'is_ref_inst_visible' in targets[0][
            0]:  # visibility labels are available per-frame for the referred object:
        is_ref_inst_visible = torch.stack([
            torch.stack([t['is_ref_inst_visible'] for t in t_step])
            for t_step in targets
        ]).permute(1, 0)
        for ref_idx, is_visible in zip(ref_indices, is_ref_inst_visible):
            is_visible = is_visible.nonzero().squeeze()
            target_is_referred[is_visible,
                               ref_idx, :] = torch.tensor([1.0, 0.0],
                                                          device=device)
    else:  # assume that the referred object is visible in every frame:
        target_is_referred[:, ref_indices, :] = torch.tensor([1.0, 0.0],
                                                             device=device)
    cost_is_referred = -(pred_is_referred.unsqueeze(2)
                         * target_is_referred.unsqueeze(1)).sum(dim=-1).mean(
                             dim=0)
    return cost_is_referred
