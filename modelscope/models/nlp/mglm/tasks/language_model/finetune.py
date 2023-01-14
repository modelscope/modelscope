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
"""GPT2 zero-shot evaluation."""

import functools
import math

import torch
from finetune_glm import finetune
from megatron_util import mpu, print_rank_0
from pretrain_glm import get_batch
from tasks.data_utils import build_data_loader
from tasks.language_model.dataset import (build_lambada_dataset,
                                          build_lm_dataset,
                                          build_wikitext103_dataset)

global_tokenizer = None


def lm_forward_step(data, model, args, timers, mems, eval_metric=None):
    """Forward step."""

    # Get the batch.
    if timers is not None:
        timers('batch generator').start()
    if 'mask' in data:
        data['attention_mask'] = data.pop('mask')
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data, args)
    if timers is not None:
        timers('batch generator').stop()

    def print_masked_text(batch_id):
        block_position_ids = position_ids[:, 1]
        position_ids_ = position_ids[:, 0]
        output_tokens = []
        sep = attention_mask[batch_id].item()
        for i, token in enumerate(tokens[batch_id, :sep].tolist()):
            if global_tokenizer is not None:
                token = global_tokenizer.IdToToken(token)
                if token.startswith('[MASK'):
                    token = f'[{position_ids_[batch_id, i].item()}, {token}]'
                if token.startswith('##') and len(
                        output_tokens) > 0 and not output_tokens[-1].endswith(
                            ']'):
                    output_tokens[-1] += token[2:]
                else:
                    output_tokens.append(token)
            else:
                output_tokens.append(str(token))
        print(' '.join(output_tokens))
        last_index = None
        for i in range(sep, tokens.size(1)):
            if global_tokenizer.IdToToken(
                    tokens[batch_id, i].item()).startswith('<|startofpiece'):
                if last_index is not None:
                    print(
                        global_tokenizer.DecodeIds(
                            tokens[batch_id, last_index:i].tolist()), '|',
                        global_tokenizer.DecodeIds(
                            labels[batch_id, last_index:i].tolist())),
                    print(position_ids_[batch_id, last_index:i].tolist(),
                          block_position_ids[batch_id, last_index:i].tolist())
                last_index = i
        if last_index is not None:
            print(
                global_tokenizer.DecodeIds(tokens[batch_id,
                                                  last_index:].tolist()), '|',
                global_tokenizer.DecodeIds(labels[batch_id,
                                                  last_index:].tolist()))
            print(position_ids_[batch_id, last_index:].tolist(),
                  block_position_ids[batch_id, last_index:].tolist())

    # Forward model.
    if args.continuous_prompt:
        prompt_pos = data['prompt_pos'].long().cuda()
        logits, *mems = model(
            tokens, position_ids, attention_mask, *mems, prompt_pos=prompt_pos)
    else:
        logits, *mems = model(tokens, position_ids, attention_mask, *mems)

    if eval_metric is None or eval_metric == 'loss':
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                                  labels)
        loss_mask = loss_mask.view(-1)
        # The loss is not normalized for fair comparison
        loss = torch.sum(losses.view(-1) * loss_mask)
        if eval_metric is None:
            loss = loss / loss_mask.sum()
        return loss, mems, 'bert'
    elif eval_metric == 'accuracy' or eval_metric == 'classify':
        logits = mpu.gather_from_model_parallel_region(logits)
        outputs = torch.argmax(logits, -1)
        correct = (outputs == labels).float()
        correct[(1 - loss_mask).bool()] = 1
        correct = correct.prod(-1)
        if eval_metric == 'accuracy':
            correct = correct.sum()
        return correct, mems, 'bert'
    else:
        raise NotImplementedError(
            'Metric {} not implemented'.format(eval_metric))


def classify_evaluate(model, dataloader, example_dict, args):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()
    predictions, labels, examples = [], [], []
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(dataloader):
            # Forward evaluation.
            output, _, _ = lm_forward_step(
                batch, model, args, None, [], eval_metric='classify')
            uid_list = batch['uid']
            example_batch = [example_dict[uid] for uid in uid_list]
            predictions.extend(output.long().tolist())
            label = batch['label'].tolist()
            labels.extend(label)
            examples.extend(example_batch)
    return predictions, labels, examples


def evaluate(model, dataloader, eval_metric, args):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_output, total_count = 0.0, 0
    total_tokens = 0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(dataloader):
            if (iteration + 1) % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output, _, _ = lm_forward_step(
                batch, model, args, None, [], eval_metric=eval_metric)
            count = batch['text'].size(0)
            count = torch.cuda.LongTensor([count])
            # Reduce across processes.
            torch.distributed.all_reduce(
                output, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(
                count, group=mpu.get_data_parallel_group())

            total_output += output.item()
            total_count += count.item()
            total_tokens += batch['loss_mask'].sum().item()
    totals = torch.cuda.FloatTensor([total_output, total_tokens])
    torch.distributed.all_reduce(totals, group=mpu.get_data_parallel_group())
    total_output, total_tokens = totals.tolist()
    print(total_tokens)
    return {eval_metric: total_output}, total_count


def evaluate_and_print_results(data_loader, model, eval_metric, args):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output, _ = evaluate(model, data_loader, eval_metric, args)

    string = ''
    if eval_metric == 'loss':
        output = output['loss']
        num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
        num_original_tokens = data_loader.dataset.num_original_tokens
        val_loss = output / (num_tokenized_tokens - 1)
        ppl = math.exp(min(20, val_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
        string += 'avg loss: {:.4E} | '.format(val_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
        score_dict = {
            'avg loss': val_loss,
            'ppl': ppl,
            'adjusted ppl': adjusted_ppl
        }

    elif eval_metric == 'accuracy':
        output = output['accuracy']
        num_examples = len(data_loader.dataset)
        acc = output / num_examples * 100
        string += 'number correct: {} | '.format(output)
        string += 'total examples: {} | '.format(num_examples)
        string += 'avg accuracy: {:.2f}'.format(acc)
        score_dict = {'accuracy': acc}
    else:
        raise NotImplementedError('evaluation method for {} metric is not '
                                  'implemented yet.'.format(eval_metric))

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    return score_dict


def metrics_func_provider(args, tokenizer, is_test):
    """Privde metrics callback function."""

    if args.task.lower() == 'lambda':
        eval_metric = 'accuracy'
        dataset = build_lambada_dataset(tokenizer, args)
    elif args.task == 'wikitext':
        eval_metric = 'loss'
        dataset = build_wikitext103_dataset(tokenizer, args)
    elif args.task == 'language_model':
        eval_metric = 'loss'
        dataset = build_lm_dataset(tokenizer, args)
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))
    # Data stuff
    dataloader = build_data_loader(
        dataset,
        args.eval_batch_size,
        args.num_workers,
        drop_last=False,
        shuffle=False)

    def metrics_func(model,
                     epoch,
                     output_predictions=False,
                     summary_writer=None):
        return evaluate_and_print_results(
            dataloader, model, eval_metric=eval_metric, args=args)

    global global_tokenizer
    global_tokenizer = tokenizer
    return metrics_func


def main(args):
    """Main program."""
    finetune(
        args,
        None, {},
        end_of_epoch_callback_provider=metrics_func_provider,
        forward_step=lm_forward_step)
