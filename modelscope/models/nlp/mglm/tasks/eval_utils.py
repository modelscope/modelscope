# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
"""Evaluation utilities."""

import datetime
import os
import random
import time
from collections import OrderedDict
from typing import List

import torch
from finetune_glm import process_batch
from megatron_util import mpu
from sklearn.metrics import f1_score
from tasks.data_utils import InputExample, build_data_loader
from utils import debug_finetune_data, get_spare_port, print_rank_0


def accuracy_metric(predictions, labels, examples):
    count = 0
    num_predictions = max(len(predictions), 1)
    assert len(predictions) == len(labels)
    for prediction, label in zip(predictions, labels):
        count += prediction == label
    return count * 100.0 / num_predictions


def f1_metric(predictions, labels, examples):
    return f1_score(labels, predictions)


def f1_macro_metric(predictions, labels, examples):
    return f1_score(labels, predictions, average='macro')


global_tokenizer = None


def accuracy_func_provider(single_dataset_provider,
                           metric_dict,
                           args,
                           is_test=False,
                           eval_func=None,
                           output_func=None,
                           only_rank0=True,
                           tokenizer=None):
    """Provide function that calculates accuracies."""
    # Build dataloaders.
    global global_tokenizer
    global_tokenizer = tokenizer
    if only_rank0 and torch.distributed.is_initialized(
    ) and torch.distributed.get_rank() != 0:
        return None
    if is_test and not args.eval_valid:
        datapaths = args.test_data if args.test_data is not None else ['test']
    else:
        datapaths = args.valid_data if args.valid_data is not None else ['dev']
    if eval_func is None:
        eval_func = multichoice_evaluate
    dataloaders = []
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset,
            eval_batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False,
            only_rank0=only_rank0)
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model,
                     epoch,
                     output_predictions=False,
                     summary_writer=None):
        print_rank_0('calculating metrics ...')
        score_dict = OrderedDict([(key, 0.0) for key in metric_dict
                                  ]) if isinstance(metric_dict, dict) else {
                                      metric_dict: 0.0
                                  }  # noqa
        total = 0
        for name, dataloader in dataloaders:
            example_dict = None
            if hasattr(dataloader.dataset, 'examples'):
                example_dict = dataloader.dataset.examples
            start_time = time.time()
            predictions, labels, examples = eval_func(model, dataloader,
                                                      example_dict, args)
            elapsed_time = time.time() - start_time
            if output_predictions and torch.distributed.get_rank() == 0:
                filename = os.path.join(args.log_dir, name + '.jsonl')
                output_func(predictions, examples, filename)
            total_count = len(predictions)
            single_dict = {
                key: metric(predictions, labels, examples)
                for key, metric in metric_dict.items()
            }
            output_str = ' > |epoch: {}| metrics for {}: total {}'.format(
                epoch, name, total_count)
            for key, value in single_dict.items():
                output_str += ' {} = {:.4f} %'.format(key, value)
                if summary_writer is not None and epoch >= 0 and not is_test and len(
                        dataloaders) > 1:
                    summary_writer.add_scalar(f'Train/valid_{name}_{key}',
                                              value, epoch)
            output_str += ' elapsed time (sec): {:.3f}'.format(elapsed_time)
            if len(dataloaders) > 1:
                print_rank_0(output_str)
            for key in score_dict:
                score_dict[key] += single_dict[key] * total_count
            total += total_count
        score_dict = {
            key: score / float(total)
            for key, score in score_dict.items()
        }
        output_str = ' >> |epoch: {}| overall: total = {}'.format(epoch, total)
        for key, score in score_dict.items():
            output_str += ' {} = {:.4f}'.format(key, score)
            if summary_writer is not None and epoch >= 0 and not is_test:
                summary_writer.add_scalar(f'Train/valid_{key}', score, epoch)
        print_rank_0(output_str)
        return score_dict

    return metrics_func


segment_length = 10


def multichoice_evaluate(model, dataloader, example_dict, args):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""
    model.eval()
    port = get_spare_port(args)
    print_rank_0(f'Using port {port}')
    store = torch.distributed.TCPStore(args.master_ip, port,
                                       torch.distributed.get_world_size(),
                                       torch.distributed.get_rank() == 0,
                                       datetime.timedelta(seconds=30))
    # file_path = os.path.join("/cache", args.experiment_name + "_store")
    # print_rank_0(f"Using file store at {file_path}")
    # store = torch.distributed.FileStore(file_path, torch.distributed.get_world_size())
    with torch.no_grad():
        # For all the batches in the dataset.
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            data = process_batch(batch, args)
            if args.pretrained_bert:
                tokens, types, labels_, attention_mask = data['text'], data[
                    'types'], data['label'], data['padding_mask']
                inputs = [tokens, types, attention_mask]
            elif args.cloze_eval:
                tokens, labels_, position_ids = data['text'], data[
                    'label'], data['position']
                attention_mask, target_ids, logit_mask = data['mask'], data[
                    'target'], data['logit_mask']
                if not args.fast_decode:
                    inputs = [
                        tokens, position_ids, attention_mask, target_ids,
                        logit_mask
                    ]
                    if args.continuous_prompt:
                        prompt_pos = data['prompt_pos']
                        inputs.append(prompt_pos)
                else:
                    dec_input_ids, dec_position_ids, dec_attention_mask = data[
                        'dec_text'], data['dec_position'], data['dec_mask']
                    dec_target_ids, dec_logit_mask = data['dec_target'], data[
                        'dec_logit_mask']
                    inputs = [
                        tokens, position_ids, attention_mask, dec_input_ids,
                        dec_position_ids, dec_attention_mask, dec_target_ids,
                        dec_logit_mask
                    ]
            else:
                tokens, labels_, position_ids, attention_mask = data[
                    'text'], data['label'], data['position'], data['mask']
                inputs = [tokens, position_ids, attention_mask]
            if len(inputs[0].shape
                   ) == 3 and inputs[0].size(1) > segment_length:
                logit_list = []
                for i in range((inputs[0].size(1) - 1) // segment_length + 1):
                    input_batch = [
                        arg[:, i * segment_length:(i + 1) * segment_length]
                        for arg in inputs
                    ]
                    if args.pretrained_bert:
                        logits = model(*input_batch)
                    else:
                        logits, *mems = model(*input_batch)
                    logit_list.append(logits)
                logits = torch.cat(logit_list, dim=1)
            elif args.cloze_eval and args.fast_decode:
                logit_list = []
                num_choices = inputs[3].size(1)
                for i in range((num_choices - 1) // segment_length + 1):
                    input_batch = inputs[:3] + [
                        arg[:, i * segment_length:(i + 1) * segment_length]
                        for arg in inputs[3:]
                    ]
                    logits, *mems = model(*input_batch)
                    logit_list.append(logits)
                logits = torch.cat(logit_list, dim=1)
            else:
                if args.pretrained_bert:
                    logits = model(*inputs)
                else:
                    logits, *mems = model(*inputs)
            if 'segment_id' in data:
                from torch_scatter import scatter_sum
                if 'loss_mask' in data:
                    logits = logits * data['loss_mask']
                logits = scatter_sum(logits, data['segment_id'], dim=1)
            elif 'loss_mask' in data:
                loss_mask = data['loss_mask']
                logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
            uid_list = batch['uid']
            if isinstance(uid_list, torch.Tensor):
                uid_list = uid_list.cpu().numpy().tolist()
            predicted = torch.argmax(logits, dim=-1).tolist()
            labels = labels_.tolist()
            if args.task.lower() == 'wsc':
                predicted = [1 if pred == 0 else 0 for pred in predicted]
            if mpu.get_model_parallel_rank() == 0:
                for uid, prediction, label in zip(uid_list, predicted, labels):
                    store.set(uid, str((prediction, label)))
    model.train()
    torch.distributed.barrier()
    predictions, labels, examples = [], [], []
    for uid, example in example_dict.items():
        prediction, label = eval(store.get(uid))
        predictions.append(prediction)
        labels.append(label)
        examples.append(example)
    torch.distributed.barrier()
    return predictions, labels, examples
