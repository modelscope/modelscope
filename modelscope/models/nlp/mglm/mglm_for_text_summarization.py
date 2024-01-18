# Copyright (c) 2022 Zhipu.AI

import os
import random
from os import path as osp
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from megatron_util import mpu

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.megatron_utils import init_megatron_util
from .arguments import get_args
from .generation_utils import BeamSearchScorer
from .train_utils import get_model
from .utils import load_checkpoint

__all__ = ['MGLMForTextSummarization']


def setup_args(args):
    args.block_lm = True
    args.task_mask = True
    args.cloze_eval = True
    args.num_layers = 24
    args.hidden_size = 1536
    args.num_attention_heads = 16
    args.max_position_embeddings = 1024
    args.tokenizer_type = 'ChineseSPTokenizer'
    args.load_pretrained = ''
    args.DDP_impl = 'none'
    args.model_parallel_size = 1
    args.fp16 = True
    args.cache_dir = 'cache'
    args.out_seq_length = 200
    args.seq_length = 512
    args.temperature = 0.9
    args.top_k = 2
    args.top_p = 0.8
    args.frequency_penalty = 0.1
    args.presence_penalty = 0.1
    args.mem_length = args.seq_length + args.mem_length - 1
    return args


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args, model_type='generation')

    if args.load_pretrained is not None:
        args.no_load_optim = True
        args.load = args.load_pretrained
        args.no_load_rng = True
        _ = load_checkpoint(model, None, None, args)

    return model


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if mem_length:
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(
            torch.triu(attention_mask, 1 - seq_length + mem_length),
            mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones(
                (att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(
            data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if set_loss_mask:
        loss_mask[data == eod_token] = 0.0
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

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
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    if args.block_lm:
        attention_mask = torch.tensor([tokens.size(1)],
                                      device=device,
                                      dtype=torch.long)
        position_ids = torch.arange(
            tokens.size(1), device=device, dtype=torch.long)
        if not args.no_block_position:
            block_position_ids = torch.zeros(
                tokens.size(1), device=device, dtype=torch.long)
            position_ids = torch.stack((position_ids, block_position_ids),
                                       dim=0)
        position_ids = position_ids.unsqueeze(0)
    else:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            reset_position_ids=False,
            reset_attention_mask=False,
            set_loss_mask=False,
            mem_length=args.mem_length)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


def sample_sequence(model,
                    tokenizer,
                    context_tokens,
                    context_length,
                    args,
                    device,
                    mems=None,
                    end_tokens=None):
    if not args.block_lm:
        context_tokens, attention_mask, position_ids = get_batch(
            context_tokens, device, args)
        tokens = torch.empty((args.num_beams, 0),
                             device=context_tokens.device,
                             dtype=torch.long)
    else:
        tokens = context_tokens.new_full((1, 1),
                                         tokenizer.get_command('sop').Id)
    counter = 0
    if mems is None:
        mems = []
    if end_tokens is None:
        end_tokens = [args.eod_token]

    last_beam_num = 1
    output_tokens_list = []
    generated_tokens_list = []

    while counter < args.out_seq_length:
        if counter == 0 and not args.block_lm:
            next_token_logits, *mems = model(context_tokens, position_ids,
                                             attention_mask, *mems)
        else:
            if args.block_lm:
                if args.no_block_position:
                    position_ids = context_tokens.new_full(
                        (last_beam_num, 1), context_length + counter)
                else:
                    position_ids = context_tokens.new_ones(last_beam_num, 2, 1)
                    position_ids[:, 0] = context_length
                    position_ids[:, 1] = counter + 1
                attention_mask = context_tokens.new_zeros(
                    [1], device=context_tokens.device, dtype=torch.long)
            else:
                position_ids = context_tokens.new_ones((last_beam_num, 1)) * (
                    context_length + counter - 1)
                attention_mask = context_tokens.new_ones(
                    last_beam_num,
                    1,
                    1,
                    args.mem_length + 1,
                    device=context_tokens.device,
                    dtype=torch.float)
            last_token = tokens[:, -1:]
            next_token_logits, *mems = model(last_token, position_ids,
                                             attention_mask, *mems)
        next_token_logits = next_token_logits[:, -1]

        next_token_logits /= args.temperature
        frequency_count = torch.zeros(next_token_logits.shape)
        for tk in output_tokens_list:
            frequency_count[0][tk] += 1

        next_token_logits -= (args.frequency_penalty
                              * frequency_count).to(device)
        next_token_logits -= (
            args.presence_penalty *  # noqa
            (frequency_count > 0)).to(device)

        next_token_logits = top_k_logits(
            next_token_logits, top_k=args.top_k, top_p=args.top_p)
        log_probs = F.softmax(next_token_logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)[0]
        is_end = prev.item() in end_tokens
        if is_end:
            break
        decode_tokens = tokenizer.DecodeIds([prev.item()])  # noqa
        generated_tokens_list.append(prev.item())
        prev = prev.view(1, 1)
        tokens = prev if tokens is None else torch.cat((tokens, prev), dim=1)
        counter += 1
        output_tokens_list = tokens.view(-1).contiguous()
    return torch.cat((context_tokens, tokens), dim=1), mems


def read_context(tokenizer, args, context):
    terminate_runs, skip_run = 0, 0  # noqa
    if mpu.get_model_parallel_rank() == 0:
        while True:
            # raw_text = input("\nContext prompt (stop to exit) >>> ")
            raw_text = context
            if not raw_text:
                print('Prompt should not be empty!')
                break
            # if raw_text == "stop":
            #     terminate_runs = 1
            #     break
            generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
            if args.block_lm and 'MASK]' not in raw_text:
                raw_text += ' ' + generation_mask
            # output.write(raw_text)
            context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
            if args.block_lm:
                context_tokens = [tokenizer.get_command('ENC').Id
                                  ] + context_tokens
                if not raw_text.endswith('[gMASK]'):
                    context_tokens = context_tokens + [
                        tokenizer.get_command('eos').Id
                    ]
            context_length = len(context_tokens)

            if context_length >= args.seq_length:
                print('\nContext length', context_length,
                      '\nPlease give smaller context than the window length!')
                break
            break
    else:
        context_length = 0

    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(
        terminate_runs_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item()

    if terminate_runs == 1:
        return terminate_runs, None, None, None

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(
        context_length_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(
        context_tokens_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group())
    if mpu.get_model_parallel_rank() != 0:
        raw_text = tokenizer.DecodeIds(context_tokens_tensor.tolist())
    return terminate_runs, raw_text, context_tokens_tensor, context_length


@MODELS.register_module(Tasks.text_summarization, module_name=Models.mglm)
class MGLMForTextSummarization(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text summarization model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        from .configure_data import prepare_tokenizer
        # Disable CuDNN.
        torch.backends.cudnn.enabled = False
        # Arguments.
        self.args = setup_args(get_args())
        self.args.load_pretrained = model_dir

        try:
            init_megatron_util(model_dir=model_dir)
        except AssertionError:
            print('megatron initialized twice')

        # setting default batch size to 1
        self.args.batch_size = 1
        self.args.tokenizer_path = model_dir
        self.tokenizer = prepare_tokenizer(self.args)
        self.model = setup_model(self.args)
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))

    def forward(self, input: Dict[str, str]) -> Dict[str, str]:
        pass

    def generate(self, input: Dict[str, str]) -> Dict[str, str]:
        model = self.model
        tokenizer = self.tokenizer
        args = self.args
        device = torch.cuda.current_device()
        model.eval()

        context = input['text'] + self.cfg.model.prompt
        with torch.no_grad():
            terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(
                tokenizer, args, context)
            mems = []
            tokens, attention_mask, position_ids = get_batch(
                context_tokens_tensor, device, args)
            mask_tokens = ['MASK', 'sMASK', 'gMASK'
                           ] if args.task_mask else ['MASK']
            mask_tokens = [
                tokenizer.get_command(token).Id for token in mask_tokens
            ]
            end_tokens = [tokenizer.get_command('eop').Id, args.eod_token]

            mask_positions = []
            for token in mask_tokens:
                mask_positions += (context_tokens_tensor == token).nonzero(
                    as_tuple=True)[0].tolist()
            mask_positions.sort()
            if args.no_block_position:
                for mask_position in mask_positions:
                    position_ids[0, mask_position + 1:] += args.out_seq_length
            _, *mems = model(tokens, position_ids, attention_mask, *mems)
            for mask_position in mask_positions:
                if args.no_block_position:
                    position = position_ids[0, mask_position].item()
                else:
                    position = mask_position
                tokens, mems, = sample_sequence(
                    model,
                    tokenizer,
                    tokens,
                    position,
                    args,
                    device,
                    mems=mems,
                    end_tokens=end_tokens)
            output_tokens_list = tokens.view(-1).contiguous()
            trim_decode_tokens = tokenizer.DecodeIds(
                output_tokens_list.tolist())
            res = trim_decode_tokens.split('<|startofpiece|>')[-1]
            print(res)
        return {OutputKeys.TEXT: res}
