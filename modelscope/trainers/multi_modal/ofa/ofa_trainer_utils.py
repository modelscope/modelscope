# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import math

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.modules.loss import _Loss


def construct_rdrop_sample(x):
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
        self.sentence_avg = args.sentence_avg
        self.eps = args.label_smoothing
        self.ignore_prefix_size = args.ignore_prefix_size
        self.ignore_eos = args.ignore_eos
        self.report_accuracy = args.report_accuracy
        self.drop_worst_ratio = args.drop_worst_ratio
        self.drop_worst_after = args.drop_worst_after
        self.use_rdrop = args.use_rdrop
        self.reg_alpha = args.reg_alpha
        self.sample_patch_num = args.sample_patch_num

        self.constraint_start = None
        self.constraint_end = None
        if args.constraint_range:
            constraint_start, constraint_end = args.constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)
        self.padding_idx = args.tokenizer.pad_token_id
        self.args = args

    def forward(self, output, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.use_rdrop:
            construct_rdrop_sample(sample)

        loss, nll_loss, ntokens = self.compute_loss(
            output, sample, update_num, reduce=reduce)
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

    def get_lprobs_and_target(self, net_output, sample):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample[
            'conf'] is not None else 1
        constraint_masks = None
        if 'constraint_masks' in sample and sample[
                'constraint_masks'] is not None:
            constraint_masks = sample['constraint_masks']
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = F.log_softmax(
            net_output[0], dim=-1, dtype=torch.float32) * conf
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

    def compute_loss(self, net_output, sample, update_num, reduce=True):
        lprobs, target, constraint_masks = self.get_lprobs_and_target(
            net_output, sample)
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


def get_schedule(scheduler):

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
