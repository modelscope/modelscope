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
"""Race."""
import functools
from collections import OrderedDict

import mpu
import torch
from finetune_glm import finetune
from pretrain_glm import get_batch
from tasks.eval_utils import accuracy_func_provider
from tasks.seq2seq.dataset import (BlankLMDataset, ExtractionDataset,
                                   Seq2SeqDataset)
from tasks.seq2seq.evaluate import (BlankLMEvaluater, DecoderEvaluater,
                                    rouge_metric)

global_tokenizer = None


def seq2seq_forward_step(data, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    if timers is not None:
        timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data, args)
    if timers is not None:
        timers('batch generator').stop()
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    # logits, loss_mask = logits[:, args.src_seq_length:], loss_mask[:, args.src_seq_length:]
    # target_ids = target_ids[:, args.src_seq_length:]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    if args.label_smoothing > 0.0:
        epsilon = args.label_smoothing
        smooth_loss = -torch.nn.functional.log_softmax(
            logits, dim=-1).mean(dim=-1)
        losses = (1 - epsilon) * losses + epsilon * smooth_loss
    loss_mask = loss_mask.reshape(-1)
    # The loss is not normalized for fair comparison
    loss = torch.sum(losses.reshape(-1) * loss_mask) / loss_mask.sum()
    return loss, mems, 'bert'


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    if args.task.lower() == 'blank':
        train_dataset = BlankLMDataset(
            args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    elif args.task.lower() == 'extraction':
        train_dataset = ExtractionDataset(
            args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    else:
        train_dataset = Seq2SeqDataset(
            args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    global global_tokenizer
    global_tokenizer = tokenizer
    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Provide metrics callback function."""

    def single_dataset_provider(split):
        if args.task.lower() == 'blank':
            return BlankLMDataset(args, split=split, tokenizer=tokenizer)
        elif args.task.lower() == 'extraction':
            return ExtractionDataset(args, split=split, tokenizer=tokenizer)
        else:
            return Seq2SeqDataset(args, split=split, tokenizer=tokenizer)

    if args.task.lower() in ['blank', 'extraction']:
        evaluater = BlankLMEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        metric_dict = {}
    else:
        evaluater = DecoderEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        if args.tokenizer_type == 'BertWordPieceTokenizer':
            dataset = 'cnn_dm'
        elif args.task.lower() == 'gigaword':
            dataset = 'gigaword'
        else:
            dataset = 'cnn_dm_org'
        metric_dict = OrderedDict({
            'rouge-1':
            functools.partial(rouge_metric, metric='rouge-1', dataset=dataset),
            'rouge-2':
            functools.partial(rouge_metric, metric='rouge-2', dataset=dataset),
            'rouge-l':
            functools.partial(rouge_metric, metric='rouge-l', dataset=dataset)
        })

    def output_func(predictions, examples, output_file):
        with open(output_file + '.hyps', 'w', encoding='utf-8') as output:
            for prediction in predictions:
                output.write(prediction)
                output.write('\n')
        with open(output_file + '.refs', 'w', encoding='utf-8') as output:
            for example in examples:
                output.write(example.meta['ref'])
                output.write('\n')
        if args.task.lower() == 'squad_generation':
            with open(
                    output_file + '.source', 'w', encoding='utf-8') as output:
                for example in examples:
                    output.write(
                        example.text_a.replace('\n', ' ') + ' Answer: '
                        + example.meta['answer'])
                    output.write('\n')

    return accuracy_func_provider(
        single_dataset_provider,
        metric_dict,
        args,
        is_test=is_test,
        eval_func=eval_func,
        output_func=output_func,
        only_rank0=False)


def main(args):
    if args.src_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.src_seq_length
    if args.task.lower() in [
            'cnn_dm', 'cnn_dm_original', 'gigaword', 'blank',
            'squad_generation', 'xsum', 'extraction'
    ]:
        finetune(
            args,
            train_valid_datasets_provider, {},
            end_of_epoch_callback_provider=metrics_func_provider,
            forward_step=seq2seq_forward_step)
    else:
        raise NotImplementedError(args.task)
