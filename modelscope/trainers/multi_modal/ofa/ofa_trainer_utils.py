# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import math
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.modules.loss import _Loss


def recursive_overwrite(src, dst, ignore=None):
    if os.path.isdir(src):
        if not os.path.isdir(dst):
            os.makedirs(dst)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(
                    os.path.join(src, f), os.path.join(dst, f), ignore)
    else:
        shutil.copyfile(src, dst)


def construct_rdrop_sample(x):
    r"""
    Construct a new sample which doubles each value.

    .. note::
        This function seems to only work when the type if `x` is `Tensor`,
        other types should check the correctness.
    """
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim() - 1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    r"""
    The Kullback-Leibler divergence loss using in OFA

    step 1. Calculate the Kullback-leibler divergence for each setting, see
    more from :class:`~torch.nn.functional.kl_div` for details:
        - `p` as input, `q` as target
        - `q` as input, `p` as target
    step 2. Average the two kl divergences as final loss.

    Args:
        p (Tensor): Tensor with arbitrary shape.
        q (Tensor): Tensor with the same shape as p.

    .. note::
        :attr:`p` and :attr:`q` should be in the log space of observation and model
        prediction values.
    """
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(lprobs,
                            target,
                            epsilon,
                            update_num,
                            reduce=True,
                            drop_worst_ratio=0.0,
                            drop_worst_after=0,
                            use_rdrop=False,
                            reg_alpha=1.0,
                            constraint_masks=None,
                            constraint_start=None,
                            constraint_end=None):
    r"""
    Computing label smoothed negative log likelihood loss.

    step 1. Calculating the negative log likelihood loss as `nll_loss`.
    step 2. Calculating the smooth loss which is the sum of last dimension of
        `nll_loss` as `smooth_loss`
    step 3. Calculating the `esp_i`, which is the scale factor of `nll_loss`
        and `smooth_loss` while calculating the `loss`.
    step 4. Calculating the `loss` using :attr:`epsilon`, `eps_i`, `nll_loss`
        and `smooth_loss`.
    step 5. If `use_rdrop` is True, computing the Kullback-Leilber divergence
        loss, making the doubled samples keep close after dropout. Add the kl
        loss to the final `loss`.

    Args:
        lprobs (`Tensor` with shape `[bsz*seq_len, embed_dim]`):
            log probabilities of the model.
        target (`Tensor` with shape `[bsz*seq_len]`):
            the target tokens
        epsilon (`float`): scale factor of combine `nll_loss` and `smooth_loss`.
        update_num (`int`): the number of updating parameters.
        drop_worst_ratio (`float`, **optional**, default to `0.0`):
            the ratio of dropped tokens whose score is worse then others.
        drop_worst_after (`int`, **optional**, default to `0`):
            the number of tokens after dropped by score.
        use_rdrop (`bool`, **optional**, default to `False`):
            whether or not to add Kullback-leilber divergence loss. if true, the
            sample should be doubled in the preprocessing.
        reg_alpha (`float`, **optional**, default to `1.0`):
            the regular factor to add kl divergence loss to total loss.
        constraint_masks (`tensor`, **optional**, default to `None`):
            bool tensor with arbitrary shape which can be broadcast to the
            shape of `lporbs`
        constraint_start(`int`, **optional**, default to `None`):
            the start of the token index.
        constraint_start(`int`, **optional**, default to `None`):
            the end of the token index.

    Returns:
        A tuple of:
         - loss, scalar tensor with average total loss of total tokens.
         - nll_loss, scalar tensor with average negative log likelihood loss
         of total tokens.
         - ntokens, the number of total tokens, should be `bsz * seq_len`.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(
            dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(
            range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(
            dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(
                loss[:true_batch_size],
                k=int(true_batch_size * (1 - drop_worst_ratio)),
                largest=False)
            loss = torch.cat([loss[indices], loss[indices + true_batch_size]])
            nll_loss = torch.cat(
                [nll_loss[indices], nll_loss[indices + true_batch_size]])
            lprobs = torch.cat(
                [lprobs[indices], lprobs[indices + true_batch_size]])
        else:
            loss, indices = torch.topk(
                loss,
                k=int(loss.shape[0] * (1 - drop_worst_ratio)),
                largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum() / ntokens  # 后面在grads里面处理
    loss = loss.sum() / ntokens  # 后面在grads里面处理
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(
                range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens


class AdjustLabelSmoothedCrossEntropyCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.sentence_avg = args.get('sentence_avg', False)
        self.eps = args.get('label_smoothing', 0.1)
        self.ignore_prefix_size = args.get('ignore_prefix_size', 0)
        self.ignore_eos = args.get('ignore_eos', False)
        self.report_accuracy = args.get('report_accuracy', False)
        self.drop_worst_ratio = args.get('drop_worst_ratio', 0.0)
        self.drop_worst_after = args.get('drop_worst_after', 0)
        self.use_rdrop = args.get('use_rdrop', False)
        self.reg_alpha = args.get('reg_alpha', 1.0)
        self.sample_patch_num = args.get('sample_patch_num', 196)
        self.ctc_weight = args.get('ctc_weight', 0.0)

        self.constraint_start = None
        self.constraint_end = None
        if args.get('constraint_range', None):
            constraint_start, constraint_end = args.constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)
        self.padding_idx = args.tokenizer.pad_token_id
        self.args = args

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if 'labels' in sample:
            del sample['labels']
        if 'samples' in sample:
            del sample['samples']

        if self.use_rdrop:
            construct_rdrop_sample(sample)
        output = model.model(**sample['net_input'])
        loss, nll_loss, ntokens = self.compute_loss(
            output.logits, sample, update_num, reduce=reduce)
        if self.ctc_weight > 0:
            ctc_loss = self.compute_ctc_loss(model, output, sample)
            loss = nll_loss + ctc_loss
        sample_size = (
            sample['target'].size(0) if self.sentence_avg else ntokens)
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, logits, sample):
        r"""
        Calculating the log probabilities from model's output `logits`, and processing the
        target from `sample`.

        step 1. Get the log probabilities from model's output logits.
            - Get the scale factor `conf`, default is `1`.
            - If some constrains are available, let the logits values out of
            constraints be :obj:`-math.inf`
            - Calculate the log softmax result and multiply scale factor `conf`,
             see :class:`~torch.nn.functional.log_softmax` for details.
            - If some ignore configs are available, remove the ignore token's
            log probabilities.
        step 2. Processing the target
            - If some ignore configs are available, remove the ignore tokens
            in the target.
        step 3. Get the constraint mask
            - If some ignore configs are available, remove the ignore tokens
            in the constraint mask.

        Args:
            logits (:obj:`Tensor` with shape `[bsz, seq_len, embed_dim]`):
                Model's output logits.
            sample (`Dict[str, Tensor]`):
                A sample for model's input, the key`target` must be in the
                sample for training.

        Returns:
            A tuple of:
             - log probabilities with shape `[bsz * (seq_len - 1), embed_dim]`
             - target token index with shape `[bsz * (seq_len - 1),]`
             - constraint mask with shape `[bsz * (seq_len - 1),]`
        """
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample[
            'conf'] is not None else 1
        constraint_masks = None
        if 'constraint_masks' in sample and sample[
                'constraint_masks'] is not None:
            constraint_masks = sample['constraint_masks']
            logits.masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            logits[:, :, 4:self.constraint_start] = -math.inf
            logits[:, :, self.constraint_end:] = -math.inf
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32) * conf
        target = sample['target']
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size:, :].contiguous()  # yapf: disable
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len - 1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len - 1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(
                    bsz, seq_len - 1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1,
                                                     constraint_masks.size(-1))
        return lprobs.view(-1,
                           lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(self, logits, sample, update_num, reduce=True):
        r"""
        Computing loss for adjust label smoothed cross entropy.

        step 1. Getting log probabilities and target and constraints mask.
        step 2. Remove the padding token result.
        step 3. Computing the label smoothed negative log likelihood loss
        as the final result.

        Args:
            logits (:obj:`Tensor` with shape `[bsz, seq_len, embed_dim]`):
                Model's output logits.
            sample (`Dict[str, Tensor]`):
                A sample for model's input, the key`target` must be in the
                sample for training.
            update_num (`int`): The number of updating parameters.

        .. note::
            The parameter `reduce` is never used in this function, should be
            removed.

        Returns:
            A tuple of:
             - loss, a scalar tensor, the final loss.
             - nll_loss, a scalar tensor, the negative log likelihood loss
             - ntokens, int, the number of tokens in calculating the loss.
        """
        lprobs, target, constraint_masks = self.get_lprobs_and_target(
            logits, sample)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end)
        return loss, nll_loss, ntokens

    def compute_ctc_loss(self, model, output, sample):
        lprobs = model.model.get_encoder_normalized_probs(
            output, log_probs=True).contiguous()  # (T, B, C) from the encoder

        non_padding_mask = ~output.encoder_padding_mask
        input_lengths = non_padding_mask.long().sum(-1)

        target_lengths = sample['phone_length']
        pad_mask = torch.arange(target_lengths.max()).expand([
            target_lengths.shape[0], -1
        ]).to(target_lengths) < target_lengths.unsqueeze(1)
        targets_flat = sample['phone_target'].masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='sum',
                zero_infinity=True,
            )

            return loss / lprobs.shape[1]


def get_schedule(scheduler):
    r"""
    Get the relative scheduler class and args by different input scheduler.
    So far, we support for types of input scheduler:
        - `const`
        - `linear`
        - `cosine`
        - `polynomial_decay`
    """

    if scheduler.name == 'const':
        scheduler_class = transformers.get_constant_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(scheduler.warmup_proportion * scheduler.num_train_steps)
        }
    elif scheduler.name == 'linear':
        scheduler_class = transformers.get_linear_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(scheduler.warmup_proportion * scheduler.num_train_steps),
            'num_training_steps':
            scheduler.num_train_steps
        }
    elif scheduler.name == 'cosine':
        scheduler_class = transformers.get_cosine_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(scheduler.warmup_proportion * scheduler.num_train_steps),
            'num_training_steps':
            scheduler.num_train_steps
        }
    elif scheduler.name == 'polynomial_decay':
        scheduler_class = transformers.get_polynomial_decay_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(scheduler.warmup_proportion * scheduler.num_train_steps),
            'num_training_steps':
            scheduler.num_train_steps,
            'lr_end':
            scheduler.lr_end
        }
    else:
        raise NotImplementedError

    return scheduler_class, scheduler_args
