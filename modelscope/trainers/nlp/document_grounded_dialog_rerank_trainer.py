import os
import random
import time
from typing import Iterable

import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.preprocessors import DocumentGroundedDialogRerankPreprocessor
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(
    module_name=Trainers.document_grounded_dialog_rerank_trainer)
class DocumentGroundedDialogRerankTrainer(EpochBasedTrainer):

    def __init__(self, model, dataset, **args):
        args = args['args']
        set_seed(args['seed'])
        self.positive_pids = ''
        self.instances_size = 1
        # load id to positive pid map
        self.inst_id2pos_pids = dict()
        self.inst_id2pos_passages = dict()
        self.dataset = dataset
        self.model = Model.from_pretrained(model, revision='v1.0.0')
        self.preprocessor = DocumentGroundedDialogRerankPreprocessor(
            self.model.model_dir, **args)
        self.tokenizer = self.preprocessor.tokenizer
        if args['model_resize']:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = self.preprocessor.device
        self.model.to(self.device)
        for jobj in self.dataset:
            self.inst_id2pos_pids[jobj['id']] = eval(jobj['positive_pids'])
            assert isinstance(eval(jobj['positive_pids']), list)
        logger.info(
            f'gathered positive pids for {len(self.inst_id2pos_pids)} instances'
        )

        # remove out-of-recall
        instance_count = 0
        for jobj in self.dataset:
            inst_id = jobj['id']
            if inst_id not in self.inst_id2pos_pids:
                continue
            passages = eval(jobj['passages'])
            positive_pids = self.inst_id2pos_pids[inst_id]
            target_mask = [p['pid'] in positive_pids for p in passages]
            if not any(target_mask) or all(target_mask):
                del self.inst_id2pos_pids[inst_id]
            else:
                instance_count += 1
        if instance_count != len(self.inst_id2pos_pids):
            logger.error(
                f'!!! Mismatch between --positive_pids and --initial_retrieval! '
                f'{len(self.inst_id2pos_pids)} vs {instance_count}')

        # transformer_optimize
        if args['train_instances'] <= 0:
            args['train_instances'] = instance_count
        # MARK
        instances_to_train_over = args['train_instances'] * args[
            'num_train_epochs'] // args['instances_size']
        self.optimizer = TransformerOptimize(args, instances_to_train_over,
                                             self.model)
        logger.info('  Num Epochs = %d', args['num_train_epochs'])
        self.optimizer.model.zero_grad()
        # MARK
        train_batch_size = \
            args['full_train_batch_size'] // args['gradient_accumulation_steps']
        self.loss_history = \
            LossHistory(
                args['train_instances'] // train_batch_size // args['instances_size']
            )
        self.args = args
        self.max_length_count = 0

    def one_instance(self, query, passages):
        model = self.optimizer.model
        input_dict = {'query': query, 'passages': passages}
        inputs = self.preprocessor(input_dict)
        logits = F.log_softmax(
            model(inputs).logits,
            dim=-1)[:, 1]  # log_softmax over the binary classification
        logprobs = F.log_softmax(
            logits, dim=0)  # log_softmax over the passages
        # we want the logits rather than the logprobs as the teacher labels
        return logprobs

    def limit_gpu_sequences_binary(self, passages, target_mask, rand):
        if len(passages) > self.args['max_num_seq_pairs_per_device']:
            num_pos = min(
                sum(target_mask),
                self.args['max_num_seq_pairs_per_device'] // 2)
            num_neg = self.args['max_num_seq_pairs_per_device'] - num_pos
            passage_and_pos = list(zip(passages, target_mask))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            target_mask = []
            for passage, mask in passage_and_pos:
                if mask and pos_count < num_pos:
                    passages.append(passage)
                    target_mask.append(mask)
                    pos_count += 1
                elif not mask and neg_count < num_neg:
                    passages.append(passage)
                    target_mask.append(mask)
                    neg_count += 1
        return passages, target_mask

    def limit_gpu_sequences(self, passages, correctness, rand):
        if len(passages) > self.args['max_num_seq_pairs_per_device']:
            num_pos = min(
                sum([c > 0 for c in correctness]),
                self.args['max_num_seq_pairs_per_device'] // 2)
            num_neg = self.args['max_num_seq_pairs_per_device'] - num_pos
            passage_and_pos = list(zip(passages, correctness))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            correctness = []
            for passage, pos in passage_and_pos:
                if pos > 0 and pos_count < num_pos:
                    passages.append(passage)
                    correctness.append(pos)
                    pos_count += 1
                elif pos == 0 and neg_count < num_neg:
                    passages.append(passage)
                    correctness.append(pos)
                    neg_count += 1
        return passages, correctness

    def passage_correctness(self, pid, positive_pids, positive_dids):
        if pid in positive_pids:
            return 1.0
        elif positive_dids and pid[:pid.index('::')] in positive_dids:
            return self.args['doc_match_weight']
        else:
            return 0

    def train(self):
        rand = random.Random()
        while self.optimizer.should_continue():
            self.optimizer.model.train()
            dataset = block_shuffle(self.dataset, block_size=100000, rand=rand)
            for line_ndx, jobj in enumerate(dataset):
                inst_id = jobj['id']
                if inst_id not in self.inst_id2pos_pids:
                    continue
                if line_ndx % self.args['world_size'] != \
                        self.args['global_rank']:
                    continue
                query = jobj['input'] if 'input' in jobj else jobj['query']
                passages = eval(jobj['passages'])
                positive_pids = self.inst_id2pos_pids[inst_id]
                if self.args['doc_match_weight'] > 0:
                    positive_dids = [
                        pid[:pid.index('::')] for pid in positive_pids
                    ]
                else:
                    positive_dids = None
                correctness = [
                    self.passage_correctness(p['pid'], positive_pids,
                                             positive_dids) for p in passages
                ]
                passages, correctness = self.limit_gpu_sequences(
                    passages, correctness, rand)
                logits = self.one_instance(query, passages)
                # nll = -(logits[target_mask].sum())  # TODO: instead take the weighted sum
                nll = -(
                    logits.dot(torch.tensor(correctness).to(logits.device)))
                loss_val = self.optimizer.step_loss(nll)
                self.loss_history.note_loss(loss_val)
                if not self.optimizer.should_continue():
                    break
        get_length = self.args['max_seq_length']
        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(
            f'truncated to max length ({get_length}) {self.max_length_count} times'
        )
        save_transformer(self.args, self.optimizer.model, self.tokenizer)


class Reporting:

    def __init__(self,
                 *,
                 recency_weight=0.001,
                 report_interval_secs=300,
                 check_every=1,
                 gather_samples: Iterable = (),
                 num_samples=10000):
        """The Reporting to print parameter status

        Args:
            recency_weight: when computing the moving average, how much weight to give to the current sample.
            report_interval_secs: how many seconds between returning true for is_time.
            check_every: how often to check the time, when calling is_time.
            gather_samples: keep the last num_samples of the listed names (gathered from moving_averages).
            num_samples: how many samples to keep.
        """
        self.check_count = 0
        self.check_every = check_every
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_interval_secs = report_interval_secs
        # For tracking moving averages of various values
        self.names = None
        self.averages = None
        self.counts = None
        self.recency_weight = recency_weight
        self.per_value_recency_weight = dict()
        self.report_count = 0
        self._prev_check_count = 0
        self.sample_names = list(gather_samples)
        if len(self.sample_names) > 0:
            self.sample_values = np.zeros(
                (len(self.sample_names), num_samples), dtype=np.float32)
            self.sample_ndxs = np.zeros(len(self.sample_names), dtype=np.int32)
        else:
            self.sample_values = None
            self.sample_ndxs = None

    def reset(self):
        self.check_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_count = 0
        self._prev_check_count = 0
        if len(self.sample_names) > 0:
            self.sample_values[:, :] = 0
            self.sample_ndxs[:] = 0
        if self.counts is not None:
            self.counts[:] = 0
            self.averages[:] = 0

    def is_time(self):
        self.check_count += 1
        if self.check_count % self.check_every == 0:
            elapsed = time.time() - self.last_time
            if elapsed >= self.report_interval_secs:
                # check the time more or less often
                if self.check_every > 1 and self.check_count - self._prev_check_count < 5 * self.check_every:
                    self.check_every //= 2
                elif self.check_count - self._prev_check_count > 50 * self.check_every:
                    self.check_every *= 2
                self.last_time = time.time()
                self.report_count += 1
                self._prev_check_count = self.check_count
                return True
        return False

    def moving_averages(self, **values):
        # create entries in avgs and counts when needed
        # update the avgs and counts
        if self.names is None:
            self.names = list(values.keys())
            self.averages = np.zeros(len(self.names))
            self.counts = np.zeros(len(self.names))
        for name in values.keys():
            if name not in self.names:
                self.names.append(name)
        if self.averages.shape[0] < len(self.names):
            old_len = self.averages.shape[0]
            self.averages = np.resize(self.averages, len(self.names))
            self.averages[old_len:] = 0
            self.counts = np.resize(self.counts, len(self.names))
            self.counts[old_len:] = 0
        for ndx, name in enumerate(self.names):
            if name in values:
                self.counts[ndx] += 1
                # support per-name recency_weight
                if name in self.per_value_recency_weight:
                    rweight = max(self.per_value_recency_weight[name],
                                  1.0 / self.counts[ndx])
                else:
                    rweight = max(self.recency_weight, 1.0 / self.counts[ndx])
                self.averages[ndx] = \
                    rweight * values[name] + (1.0 - rweight) * self.averages[ndx]
        for ndx, name in enumerate(self.sample_names):
            if name in values:
                self.sample_values[self.sample_ndxs[ndx]] = values[name]
                self.sample_ndxs[ndx] = (self.sample_ndxs[ndx]
                                         + 1) % self.sample_values.shape[1]

    def get_samples(self, name):
        for ndx, n in enumerate(self.sample_names):
            if n == name:
                count = self.get_count(name)
                if count is None:
                    count = 0
                return self.sample_values[ndx, 0:count]  # NOTE: not in order
        return None

    def get_moving_average(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.averages[ndx]
        return None

    def get_count(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.counts[ndx]
        return None

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def elapsed_time_str(self) -> str:
        return time_str(self.elapsed_seconds())

    def progress_str(self, instance_name='instance'):
        return f'On {instance_name} {self.check_count}, ' \
               f'{self.check_count / self.elapsed_seconds()} {instance_name}s per second.'

    def display(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.info(f'{prefix}{n} = {v}')

    def display_warn(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.warning(f'{prefix}{n} = {v}')


class LossHistory:

    def __init__(self,
                 one_epoch_batch_count,
                 *,
                 loss_points_per_epoch=10,
                 recency_weight=0.001):
        self.avg_loss = 0
        self.batch_count = 0
        self.recency_weight = recency_weight
        self.loss_history = []
        self.record_loss_every = max(
            1, one_epoch_batch_count // loss_points_per_epoch)

    def note_loss(self, loss_val):
        self.batch_count += 1
        rweight = max(self.recency_weight, 1.0 / self.batch_count)
        self.avg_loss = (1.0 - rweight) * self.avg_loss + rweight * loss_val
        if self.batch_count % self.record_loss_every == 0:
            self.loss_history.append(self.avg_loss)
            logger.info(
                f'loss point {self.batch_count // self.record_loss_every} = {self.avg_loss}'
            )
            if self.avg_loss == min(
                    self.loss_history) and len(self.loss_history) > 10:
                return 2
            return True
        return False


class TransformerOptimize:
    """
    Collects standard steps to train transformer
    call step_loss after computing each loss
    """

    def __init__(self, hypers, num_instances_to_train_over: int, model):
        self.step = 0
        self.global_step = 0
        self.hypers = hypers
        self.model = model
        instances_per_step = hypers['full_train_batch_size'] // hypers[
            'gradient_accumulation_steps']
        self.reporting = Reporting(recency_weight=0.0001 * instances_per_step)
        args = self.hypers

        self.t_total = num_instances_to_train_over // args[
            'full_train_batch_size']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                args['weight_decay'],
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
        ]

        warmup_instances = args['warmup_instances']
        if hasattr(
                args, 'warmup_fraction'
        ) and args['warmup_fraction'] > 0 >= args['warmup_instances']:
            warmup_instances = \
                args['warmup_fraction'] * num_instances_to_train_over
        if warmup_instances < 0:
            warmup_instances = 0

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args['learning_rate'],
            eps=args['adam_epsilon'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_instances // args['full_train_batch_size'],
            num_training_steps=self.t_total)

        # Check if saved optimizer or scheduler states exist
        if args['resume_from'] and os.path.isfile(os.path.join(args['resume_from'], 'optimizer.pt')) and \
                os.path.isfile(os.path.join(args['resume_from'], 'scheduler.pt')):
            resume_from = args['resume_from']
        # elif os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and \
        #         os.path.isfile(os.path.join(args['model_name_or_path'], "scheduler.pt")):
        #     resume_from = args['model_name_or_path']
        else:
            resume_from = None
        if resume_from is not None:
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(resume_from, 'optimizer.pt'),
                    map_location='cpu'))
            self.scheduler.load_state_dict(
                torch.load(
                    os.path.join(resume_from, 'scheduler.pt'),
                    map_location='cpu'))
            logger.info(f'loaded optimizer and scheduler from {resume_from}')

        if args['fp16']:
            self.model, optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=args['fp16_opt_level'])

        # multi-gpu training (should be after apex fp16 initialization)
        if args['n_gpu'] > 1:
            # NOTE: won't work at O2, only O1
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(args['n_gpu'])))

        # Distributed training (should be after apex fp16 initialization)
        # if args.local_rank != -1:
        #     self.model = torch.nn.parallel.DistributedDataParallel(
        #         self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        #     )
        # set_seed(args)
        # assert args.per_gpu_train_batch_size * (args.n_gpu if args.n_gpu > 0 else 1) * \
        #        args.world_size * args.gradient_accumulation_steps == args.full_train_batch_size
        logger.info('***** Running training *****')
        logger.info('  Instantaneous batch size per GPU = %d',
                    args['per_gpu_train_batch_size'])
        logger.info(
            '  Total train batch size (w. parallel, distributed & accumulation) = %d',
            args['full_train_batch_size'])
        logger.info('  Gradient Accumulation steps = %d',
                    args['gradient_accumulation_steps'])
        logger.info('  Total optimization steps = %d', self.t_total)

    def should_continue(self):
        return self.global_step < self.t_total

    def backward_on_loss(self, loss, **moving_averages):
        if self.hypers['n_gpu'] > 1:
            loss = loss.mean(
            )  # mean() to average on multi-gpu parallel training
        loss_val = loss.item()
        if self.hypers['gradient_accumulation_steps'] > 1:
            loss = loss / self.hypers['gradient_accumulation_steps']
        if self.hypers['fp16']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.reporting.moving_averages(loss=loss_val, **moving_averages)
        return loss_val

    def optimizer_step(self):
        if self.global_step >= self.t_total:
            logger.warning(
                f'Warning, exceeded total steps! {self.global_step} step of {self.t_total}'
            )
            return False
        if (self.step + 1) % self.hypers['gradient_accumulation_steps'] == 0:
            if self.hypers['max_grad_norm'] > 0:
                if self.hypers['fp16']:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer),
                        self.hypers['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.hypers['max_grad_norm'])

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        self.step += 1

        if self.reporting.is_time():
            self.reporting.display()
            inst_count = \
                self.hypers['world_size'] * self.hypers['n_gpu'] * self.hypers[
                    'per_gpu_train_batch_size'] * self.reporting.check_count
            learning_rate_scalar = self.scheduler.get_lr()[0]
            logger.info(
                f'{inst_count / self.reporting.elapsed_seconds()} instances per second; '
                f'{inst_count} total ({learning_rate_scalar} learn rate)')
        return True

    def step_loss(self, loss, **moving_averages):
        loss_val = self.backward_on_loss(loss, **moving_averages)
        if self.optimizer_step():
            return loss_val
        else:
            return None


def block_shuffle(iter, *, block_size=20000, rand=random):
    """
    shuffle the possibly endless iterator by blocks
    Good shuffling over multiple files:
    block_shuffle(read_lines(files, shuffled_files=rand), rand=rand, block_size=100000)
    :param iter: the iterator we will yield shuffled items from
    :param block_size: size of memory to use for block shuffling
    :param rand: rand.shuffle will be used on the list block
    :return:
    """
    assert block_size >= 4
    block = []
    for item in iter:
        block.append(item)
        if len(block) >= block_size:
            rand.shuffle(block)
            for _ in range(block_size // 2):
                yield block.pop(-1)
    rand.shuffle(block)
    for bi in block:
        yield bi


def save_transformer(hypers, model, tokenizer, *, save_dir=None):
    if hypers['global_rank'] == 0:
        if save_dir is None:
            save_dir = hypers['output_dir']
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info('Saving model checkpoint to %s', save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, 'module') else model
                         )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, 'training_args.bin'))
        model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def kofn(kofn: str):
    """
    ''     -> 0, 1
    '1of2' -> 0, 2
    '2of2' -> 1, 2
    :param kofn:
    :return:
    """
    if not kofn:
        return 0, 1
    k, n = [int(i) for i in kofn.lower().split('of')]
    assert 1 <= k <= n
    return k - 1, n


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
