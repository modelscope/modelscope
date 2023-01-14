# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for logging and serialization"""

import os
import random
import subprocess
import time

import json
import numpy as np
import torch
from megatron_util import mpu, print_rank_0
from megatron_util.fp16 import FP16_Optimizer

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_log_dir(name, base):
    return os.path.join(base, SUMMARY_WRITER_DIR_NAME, name)


def get_hostname():
    hostname_cmd = ['hostname -I']
    result = subprocess.check_output(hostname_cmd, shell=True)
    master_addr = result.decode('utf-8').split()[0]
    return master_addr


def get_spare_port(args):
    if torch.distributed.get_rank() == 0:
        port = subprocess.check_output(['shuf -n 1 -i 10000-65535'],
                                       shell=True)
        port = int(port.strip())
        if port == args.master_port:
            port = subprocess.check_output(['shuf -n 1 -i 10000-65535'],
                                           shell=True)
            port = int(port.strip())
        port = torch.cuda.LongTensor([port])
    else:
        port = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(port, 0)
    port = port.item()
    return port


def print_and_save_args(args, verbose=True, log_dir=None):
    """Print arguments."""
    if verbose:
        print('arguments:', flush=True)
        for arg in vars(args):
            dots = '.' * (29 - len(arg))
            print(
                '  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)
    if log_dir is not None:
        json_file = os.path.join(log_dir, 'config.json')
        with open(json_file, 'w') as output:
            json.dump(vars(args), output, sort_keys=True)
        if args.deepspeed and args.deepspeed_config is not None:
            with open(args.deepspeed_config, encoding='utf-8') as file:
                deepspeed_config = json.load(file)
            deepspeed_json_file = os.path.join(log_dir,
                                               'config_gpt_large.json')
            with open(deepspeed_json_file, 'w') as output:
                json.dump(deepspeed_config, output)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    if isinstance(optimizer, FP16_Optimizer):
        optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated()
                                        / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated()
                                            / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(torch.cuda.memory_reserved()
                                         / mega_bytes)
    print_rank_0(string)


def get_checkpoint_name(checkpoints_path,
                        iteration,
                        release=False,
                        zero=False):
    if release:
        d = 'release'
    else:
        d = '{}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(
        checkpoints_path, d,
        'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {
        'iteration': iteration,
        'optimizer_state_dict': optimizer.state_dict()
    }
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    tag=None,
                    barrier=True,
                    only_changed_parameters=False,
                    no_deepspeed=False,
                    no_save_optim=False):
    """Save a model checkpoint."""
    if tag is None:
        tag = str(iteration)
    if args.deepspeed and not no_deepspeed:
        save_ds_checkpoint(iteration, model, lr_scheduler, args, tag=tag)
    else:
        # Only rank zer0 of the data parallel writes to the disk.

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, tag)
            print(
                'global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration,
                       checkpoint_name))
            sd = {'iteration': iteration}
            if args.deepspeed:
                model = model.module
            state_dict = model.state_dict()
            if only_changed_parameters:
                requires_grad_dict = {}
                for name, parameter in model.named_parameters():
                    requires_grad_dict[name] = parameter.requires_grad
                state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if requires_grad_dict[key]
                }
            sd['module'] = state_dict

            # Optimizer stuff.
            if not args.no_save_optim and not no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker(
                ).get_states()

            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    if barrier:
        torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(tag)


def save_ds_checkpoint(iteration, model, lr_scheduler, args, tag):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    model.save_checkpoint(args.save, tag, client_state=sd)


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        if os.path.isdir(load_path):
            path = os.path.normpath(load_path)
            load_dir, tag = os.path.split(path)
            print_rank_0(
                'Try to directly load the checkpoint from the directory')
            return load_dir, tag, False, True
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return load_path, 0, False, False
    with open(tracker_filename, 'r', encoding='utf-8') as f:
        metastring = f.read().strip()
        release = metastring == 'release'
        # try:
        #     iteration = int(metastring)
        # except ValueError:
        #     release = metastring == 'release'
        #     if not release:
        #         print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
        #             tracker_filename))
        #         exit()

    # assert iteration > 0 or release, 'error parsing metadata file {}'.format(
    #     tracker_filename)

    return load_path, metastring, release, True


def load_checkpoint(model,
                    optimizer,
                    lr_scheduler,
                    args,
                    no_deepspeed=False,
                    no_load_optim=False):
    """Load a model checkpoint."""

    load_dir, tag, release, success = get_checkpoint_iteration(args.load)

    if not success:
        return 0

    if args.deepspeed and not no_deepspeed:

        checkpoint_name, sd = model.load_checkpoint(
            load_dir,
            tag,
            load_optimizer_states=not args.no_load_optim and not no_load_optim,
            load_lr_scheduler_states=not args.no_load_lr_scheduler)
        if not args.no_load_lr_scheduler and 'client_lr_scheduler' in sd:
            lr_scheduler.load_state_dict(sd['client_lr_scheduler'])
            print_rank_0('Load lr scheduler state')
        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print('Unable to load checkpoint.')
            return tag

    else:

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(load_dir, tag, release)

        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        # Model.
        if args.deepspeed:
            model = model.module
        missing_keys, unexpected_keys = model.load_state_dict(
            sd['module'], strict=False)
        if missing_keys or unexpected_keys:
            print_rank_0(
                f'Missing keys {missing_keys}, unexpected keys {unexpected_keys}'
            )

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim and not no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0(
                    'Unable to load optimizer from checkpoint {}, exiting. '
                    'Specify --no-load-optim or --finetune to prevent '
                    'attempting to load the optimizer '
                    'state.'.format(checkpoint_name))

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0(
                    'A metadata file exists but Unable to load iteration '
                    ' from checkpoint {}, starting from 0 iteration'.format(
                        checkpoint_name))
                iteration = 0

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0(
                'Unable to load random state from checkpoint {}, exiting. '
                'Specify --no-load-rng or --finetune to prevent '
                'attempting to load the random '
                'state.'.format(checkpoint_name))

    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_weights(src, dst, dst2src=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt2model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)


#        dst._parameters[n].data.copy_(data)


def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)


def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)


def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)


def move_weights(our, oai, dst2src=False):
    """
    Loads weights from `oai` to `our` via in place copy.
    `oai` is a huggingface gpt2model, while `our` is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src=True is still untested
    """
    #    while isinstance(our, (torchDDP, model.distributed.DistributedDataParallel, FP16_Module)):
    #        our=our.module
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm,
                 dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)


def debug_finetune_data(local_vars, batch_id, tokenizer):
    tokens, target_ids = local_vars['tokens'], local_vars['target_ids']
    attention_mask, logit_mask, position_ids = local_vars[
        'attention_mask'], local_vars['logit_mask'], local_vars['position_ids']
    output_tokens = []
    sep = attention_mask[batch_id].item()
    for i, token in enumerate(tokens[batch_id][:sep].tolist()):
        token = tokenizer.IdToToken(token)
        if token == '[MASK]':
            token = f'[{position_ids[batch_id][0, i].item()}]'
        output_tokens.append(token)
    print(' '.join(output_tokens))
    target_positions = []
    for i in range(sep, tokens.size(-1)):
        if logit_mask[batch_id][i]:
            target_positions.append(i)
    print(target_positions)
    print(tokenizer.DecodeIds(tokens[batch_id][target_positions].tolist()))
    if len(target_ids.shape) > 2:
        print(
            tokenizer.DecodeIds(
                target_ids[batch_id][target_positions].tolist()))
    else:
        print(tokenizer.DecodeIds(target_ids[batch_id].tolist()))
    print(position_ids[batch_id][:, target_positions])
