# Copyright (c) 2022 Zhipu.AI

from typing import List

import torch
import torch.nn.functional as F


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length),
                   device=data.device)).view(att_mask_batch, 1, seq_length,
                                             seq_length)

    # Position ids.
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, position_ids


def get_batch(
    context_tokens,
    micro_batch_size,
    eod_token,
    reset_position_ids=False,
    reset_attention_mask=False,
):
    """Generate batch from context tokens."""
    tokens = context_tokens.view(micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        reset_position_ids,
        reset_attention_mask,
    )

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def pad_batch(batch, pad_id, seq_length):
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(
    model,
    tokenizer,
    seq_length,
    out_seq_length,
    context_tokens,
    return_scores: bool = False,
    prompt_length: int = None,
    micro_batch_size: int = None,
    bad_ids: List = None,
    temperature: float = 1.0,
    topp: float = 1.0,
    topk: int = 0.0,
    greedy: bool = False,
):
    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eos_token_id,
                                                seq_length)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(
        context_tokens_tensor,
        micro_batch_size,
        tokenizer.eos_token_id,
    )

    batch_token_iterator = sample_sequence_batch(
        model,
        tokenizer,
        context_tokens_tensor,
        context_length_tensor,
        attention_mask,
        position_ids,
        seq_length=seq_length,
        out_seq_length=out_seq_length,
        return_scores=return_scores,
        prompt_length=prompt_length,
        bad_ids=bad_ids,
        temperature=temperature,
        topp=topp,
        topk=topk,
        greedy=greedy,
    )

    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    tokenizer,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    seq_length,
    out_seq_length,
    maxlen=None,
    return_scores: bool = False,
    prompt_length: int = None,
    bad_ids: List = None,
    temperature: float = 1.0,
    topp: float = 1.0,
    topk: int = 0.0,
    recompute: bool = False,
    greedy: bool = False,
):
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.eos_token_id

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = seq_length - 1
            if maxlen > (org_context_length + out_seq_length):
                maxlen = org_context_length + out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        if return_scores:
            scores = torch.zeros([batch_size]).float().cuda()

        while context_length <= (maxlen):

            if recompute:
                logits = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    prompt_length=prompt_length,
                    context_length=context_length,
                )
                logits = logits[:, context_length - 1, :]
            else:
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                logits, layer_past = model(
                    tokens2use,
                    positions2use,
                    attention_mask,
                    layer_past=layer_past,
                    get_key_value=True,
                    prompt_length=prompt_length,
                    context_length=context_length,
                )
                logits = logits[:, -1].view(batch_size, -1).contiguous()

            if bad_ids is not None:
                for bad_id in bad_ids:
                    logits[:, bad_id] = -10000
            if greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                if return_scores:
                    orig_log_probs = torch.log_softmax(logits, dim=-1)
                logits /= temperature
                logits = top_k_logits(logits, top_k=topk, top_p=topp)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_lengths <= context_length

            new_tokens = switch(tokens[:, context_length].view(-1), prev,
                                started)

            if not greedy and return_scores:
                indices = prev.view(-1, 1)
                new_scores = orig_log_probs.gather(1, indices).view(-1)
                new_scores = new_scores * started
                new_scores = new_scores * is_done.bool().logical_not()
                scores += new_scores

            tokens[:, context_length] = new_tokens
            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token
            done = torch.all(is_done)

            if return_scores:
                yield tokens, (lengths, scores)
            else:
                yield tokens, lengths

            context_length += 1
            counter += 1
            if done:
                break
