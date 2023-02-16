# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import string
from collections import Counter

import json
import sacrebleu
import torch
import tqdm
from rouge import Rouge
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.preprocessors import DocumentGroundedDialogGeneratePreprocessor
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger

logger = get_logger()


def collate(batch):
    query = [item['query'] for item in batch]
    context = [json.loads(item['rerank']) for item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def measure_result(result_dict):
    meters = dict()

    hypothesis_list = [
        x.split('<response>')[-1].strip() for x in result_dict['outputs']
    ]
    hypothesis_list = [x if x else '@' for x in hypothesis_list]
    reference_list = [
        x.split('<response>')[-1].strip() for x in result_dict['targets']
    ]
    instance_num = len(reference_list)

    # F1
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [
        x['rouge-l']['f']
        for x in rouge_func.get_scores(hypothesis_list, reference_list)
    ]
    rouge_score = (sum(rouge_score) / instance_num) * 100
    meters['rouge'] = rouge_score

    return meters


@TRAINERS.register_module(
    module_name=Trainers.document_grounded_dialog_generate_trainer)
class DocumentGroundedDialogGenerateTrainer(EpochBasedTrainer):

    def __init__(self, model: str, revision='v1.0.0', *args, **kwargs):
        self.model = Model.from_pretrained(model, revision=revision)
        self.preprocessor = DocumentGroundedDialogGeneratePreprocessor(
            model_dir=self.model.model_dir)
        self.device = self.preprocessor.device
        self.model.model.to(self.device)
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']

    def train(self,
              total_epoches=10,
              batch_size=16,
              accumulation_steps=1,
              learning_rate=1e-4,
              warmup_ratio=0.1,
              weight_decay=0.1,
              eps=1e-06,
              loss_log_freq=40):
        """
        Fine-tuning trainsets
        """
        # obtain train loader
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate)

        optimizer = prepare_optimizer(self.model.model, learning_rate,
                                      weight_decay, eps)
        steps_per_epoch = len(train_loader) // accumulation_steps
        scheduler = prepare_scheduler(optimizer, total_epoches,
                                      steps_per_epoch, warmup_ratio)
        scaler = GradScaler()
        best_score = 0.0
        for epoch in range(total_epoches):
            self.model.model.train()
            losses = []
            for index, payload in enumerate(tqdm.tqdm(train_loader)):
                query, context, label = payload
                processed = self.preprocessor(
                    {
                        'query': query,
                        'context': context,
                        'label': label
                    },
                    invoke_mode=ModeKeys.TRAIN)
                with autocast():
                    outputs = self.model.forward(processed)
                    loss = outputs.loss.mean()

                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (index + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
                if (index + 1) % loss_log_freq == 0:
                    logger.info(
                        f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                    )
                    losses = []
            if losses:
                logger.info(
                    f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
                )

            meters = self.evaluate(batch_size=batch_size)
            total_score = sum([x for x in meters.values()])
            if total_score >= best_score:
                best_score = total_score
                model_path = os.path.join(self.model.model_dir,
                                          'finetuned_model.bin')
                state_dict = self.model.model.state_dict()
                torch.save(state_dict, model_path)
                logger.info(
                    'epoch %d obtain max score: %.4f, saving model to %s' %
                    (epoch, total_score, model_path))

    def evaluate(self, batch_size=16, checkpoint_path=None):
        """
        Evaluate testsets
        """
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.model.model.load_state_dict(state_dict)

        valid_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=batch_size,
            collate_fn=collate)
        self.model.model.eval()
        with torch.no_grad():
            results = {'outputs': [], 'targets': []}
            for index, payload in enumerate(tqdm.tqdm(valid_loader)):
                query, context, label = payload
                processed = self.preprocessor(
                    {
                        'query': query,
                        'context': context,
                    },
                    invoke_mode=ModeKeys.INFERENCE)
                outputs = self.model.generate(processed)
                predictions = self.preprocessor.generation_tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                label = self.preprocessor.generation_tokenizer.batch_decode(
                    self.preprocessor.generation_tokenizer.batch_encode_plus(
                        label, add_special_tokens=False).input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)

                results['outputs'] += predictions
                results['targets'] += label
            meters = measure_result(results)
        logger.info(meters)
        return meters
