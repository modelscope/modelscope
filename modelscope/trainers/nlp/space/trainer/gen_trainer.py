# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from collections import OrderedDict

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from modelscope.trainers.nlp.space.metrics.metrics_tracker import \
    MetricsTracker
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp.space import ontology


class Trainer(object):

    def __init__(self,
                 model,
                 to_tensor,
                 config,
                 logger=None,
                 lr_scheduler=None,
                 optimizer=None,
                 reader=None,
                 evaluator=None):
        self.to_tensor = to_tensor

        self.do_train = config.do_train
        self.do_infer = config.do_infer
        if self.do_train:
            self.is_decreased_valid_metric = config.Trainer.valid_metric_name[
                0] == '-'
            self.valid_metric_name = config.Trainer.valid_metric_name[1:]
            self.num_epochs = config.Trainer.num_epochs
            self.save_dir = config.Trainer.save_dir
            self.log_steps = config.Trainer.log_steps
            self.valid_steps = config.Trainer.valid_steps
            self.save_checkpoint = config.Trainer.save_checkpoint
            self.save_summary = config.Trainer.save_summary
        self.lr = config.Model.lr
        self.weight_decay = config.Model.weight_decay
        self.batch_size = config.Trainer.batch_size
        self.gradient_accumulation_steps = config.Model.gradient_accumulation_steps
        self.warmup_steps = config.Model.warmup_steps
        self.gpu = config.Trainer.gpu

        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

        self.model = model
        self.func_model = self.model.module if self.gpu > 1 and config.use_gpu else self.model
        self.reader = reader
        self.evaluator = evaluator
        self.tokenizer = reader.tokenizer

        self.logger = logger or get_logger()

        self.batch_metrics_tracker = MetricsTracker()
        self.token_metrics_tracker = MetricsTracker()

        if self.do_train:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.best_valid_metric = float(
                'inf' if self.is_decreased_valid_metric else '-inf')
        self.epoch = 0

    def decode_generated_bspn_resp(self, generated):
        """
        decode generated
        return decoded ('bspn', 'resp')
        """
        decoded = {}
        eos_r_id = self.reader.eos_r_id
        eos_b_id = self.reader.eos_b_id

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated) - 1
            # self.logger.info('eos_r not in generated: ' + self.tokenizer.decode(generated))

        # predicted bspn, resp
        eos_b_idx = generated.index(eos_b_id)
        decoded['bspn'] = generated[:eos_b_idx + 1]
        decoded['resp'] = generated[eos_b_idx + 1:eos_r_idx + 1]
        return decoded

    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.reader.eos_a_id
        eos_r_id = self.reader.eos_r_id
        # eos_b_id = self.reader.eos_b_id

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated) - 1
            msg = 'eos_r not in generated: ' + self.tokenizer.decode(generated)
            self.logger.info(msg)

        if self.reader.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[:eos_r_idx + 1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[:eos_a_idx + 1]
            decoded['resp'] = generated[eos_a_idx + 1:eos_r_idx + 1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.reader.eos_b_id
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[:eos_b_idx + 1]

    def set_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                self.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        num_training_steps = \
            self.reader.set_stats['train']['num_training_steps_per_epoch'] \
            * self.num_epochs \
            // self.gradient_accumulation_steps
        num_warmup_steps = self.warmup_steps if self.warmup_steps >= 0 else int(
            num_training_steps * 0.1)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, train_data, dev_data):
        # log info
        set_stats = self.reader.set_stats['train']
        self.logger.info('***** Running training *****')
        self.logger.info(
            '  Num Training steps(one turn in a batch of dialogs) per epoch = %d',
            set_stats['num_training_steps_per_epoch'])
        self.logger.info('  Num Turns = %d', set_stats['num_turns'])
        self.logger.info('  Num Dialogs = %d', set_stats['num_dials'])
        self.logger.info('  Num Epochs = %d', self.num_epochs)
        self.logger.info('  Batch size  = %d', self.batch_size)
        self.logger.info('  Gradient Accumulation steps = %d',
                         self.gradient_accumulation_steps)
        steps = set_stats[
            'num_training_steps_per_epoch'] * self.num_epochs // self.gradient_accumulation_steps
        msg = '  Total optimization steps = %d' % steps
        self.logger.info(msg)

        # begin training
        num_epochs = self.num_epochs - self.epoch
        for epoch in range(num_epochs):
            self.train_epoch(train_data=train_data, dev_data=dev_data)

    def train_epoch(self, train_data, dev_data):
        """
        Train an epoch.
        """
        raise NotImplementedError

    def infer(self, data_type):
        """
        Inference interface.
        """
        raise NotImplementedError

    def save(self, is_best=False):
        """ save """
        train_state = {
            'epoch': self.epoch,
            'best_valid_metric': self.best_valid_metric,
            'optimizer': self.optimizer.state_dict()
        }
        if self.lr_scheduler is not None:
            train_state['lr_scheduler'] = self.lr_scheduler.state_dict()

        # Save checkpoint
        if self.save_checkpoint:
            model_file = os.path.join(self.save_dir,
                                      f'state_epoch_{self.epoch}.model')
            torch.save(self.model.state_dict(), model_file)
            self.logger.info(f"Saved model state to '{model_file}'")

            train_file = os.path.join(self.save_dir,
                                      f'state_epoch_{self.epoch}.train')
            torch.save(train_state, train_file)
            self.logger.info(f"Saved train state to '{train_file}'")

        # Save current best model
        if is_best:
            best_model_file = os.path.join(self.save_dir,
                                           ModelFile.TORCH_MODEL_BIN_FILE)
            torch.save(self.model.state_dict(), best_model_file)
            best_train_file = os.path.join(
                self.save_dir,
                '{}.train'.format(ModelFile.TORCH_MODEL_BIN_FILE))
            torch.save(train_state, best_train_file)
            self.logger.info(
                f"Saved best model state to '{best_model_file}' with new best valid metric "
                f'{self.valid_metric_name.upper()}={self.best_valid_metric:.3f}'
            )

    def load(self):
        """ load """

        def _load_model_state():
            model_state_dict = torch.load(
                f'{self.func_model.init_checkpoint}',
                map_location=lambda storage, loc: storage)

            if 'module.' in list(model_state_dict.keys())[0]:
                new_model_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    assert k[:7] == 'module.'
                    new_model_state_dict[k[7:]] = v
                model_state_dict = new_model_state_dict

            new_model_state_dict = OrderedDict()
            parameters = {
                name: param
                for name, param in self.func_model.named_parameters()
            }
            for name, param in model_state_dict.items():
                if name in parameters:
                    if param.shape != parameters[name].shape:
                        assert hasattr(param, 'numpy')
                        arr = param.numpy()
                        z = np.random.normal(
                            scale=self.func_model.initializer_range,
                            size=parameters[name].shape).astype('float32')
                        if name == 'embedder.token_embedding.weight':
                            z[-param.shape[0]:] = arr
                            print(
                                f'part of parameter({name}) random normlize initialize'
                            )
                        else:
                            if z.shape[0] < param.shape[0]:
                                z = arr[:z.shape[0]]
                                print(f'part of parameter({name}) are dropped')
                            else:
                                z[:param.shape[0]] = arr
                                print(
                                    f'part of parameter({name}) random normlize initialize'
                                )
                        dtype, device = param.dtype, param.device
                        z = torch.tensor(z, dtype=dtype, device=device)
                        new_model_state_dict[name] = z
                    else:
                        new_model_state_dict[name] = param
                else:
                    print(f'parameter({name}) are dropped')
            model_state_dict = new_model_state_dict

            for name in parameters:
                if name not in model_state_dict:
                    if parameters[name].requires_grad:
                        print(f'parameter({name}) random normlize initialize')
                        z = np.random.normal(
                            scale=self.func_model.initializer_range,
                            size=parameters[name].shape).astype('float32')
                        dtype, device = parameters[name].dtype, parameters[
                            name].device
                        model_state_dict[name] = torch.tensor(
                            z, dtype=dtype, device=device)
                    else:
                        model_state_dict[name] = parameters[name]

            self.func_model.load_state_dict(model_state_dict)
            self.logger.info(
                f"Loaded model state from '{self.func_model.init_checkpoint}'")

        def _load_train_state():
            train_file = f'{self.func_model.init_checkpoint}.train'
            if os.path.exists(train_file):
                train_state_dict = torch.load(
                    train_file, map_location=lambda storage, loc: storage)
                self.epoch = train_state_dict['epoch']
                self.best_valid_metric = train_state_dict['best_valid_metric']
                if self.optimizer is not None and 'optimizer' in train_state_dict:
                    self.optimizer.load_state_dict(
                        train_state_dict['optimizer'])
                if self.lr_scheduler is not None and 'lr_scheduler' in train_state_dict:
                    self.lr_scheduler.load_state_dict(
                        train_state_dict['lr_scheduler'])
                self.logger.info(
                    f"Loaded train state from '{train_file}' with (epoch-{self.epoch} "
                    f'best_valid_metric={self.best_valid_metric:.3f})')
            else:
                self.logger.info('Loaded no train state')

        if self.func_model.init_checkpoint is None:
            self.logger.info('Loaded no model !!!')
            return

        if self.do_train:
            _load_model_state()
            return

        if self.do_infer:
            _load_model_state()
            _load_train_state()


class MultiWOZTrainer(Trainer):

    def __init__(self,
                 model,
                 to_tensor,
                 config,
                 logger=None,
                 lr_scheduler=None,
                 optimizer=None,
                 reader=None,
                 evaluator=None):
        super(MultiWOZTrainer,
              self).__init__(model, to_tensor, config, logger, lr_scheduler,
                             optimizer, reader, evaluator)

    def train_epoch(self, train_data, dev_data):
        """
        Train an epoch.
        """
        times = []
        epoch_step = 0
        global_step = 0
        tr_batch_loss = 0.0
        tr_token_loss = 0.0
        self.epoch += 1
        self.batch_metrics_tracker.clear()
        self.token_metrics_tracker.clear()
        num_training_steps = \
            self.reader.set_stats['train']['num_training_steps_per_epoch'] // \
            self.gradient_accumulation_steps  # similar to the original num_batches

        self.model.zero_grad()
        data_iterator = self.reader.get_data_iterator(all_batches=train_data)

        for batch_idx, dial_batch in enumerate(data_iterator):
            pv_batch = []
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num == 0)
                samples, pv_batch = self.reader.convert_batch_turn(
                    turn_batch, pv_batch, first_turn)
                batch, batch_size = self.reader.collate_fn_multi_turn(
                    samples=samples)
                batch = type(batch)(
                    map(lambda kv: (kv[0], self.to_tensor(kv[1])),
                        batch.items()))

                # Do a training iteration
                start_time = time.time()
                metrics = self.model(batch, is_training=True)
                if self.gpu > 1:
                    for metric in metrics:
                        if metric is not None:
                            assert len(metric) == self.gpu
                    nll, token_nll, token_num = metrics
                    metrics = {}

                    token_num = torch.sum(token_num)
                    token_nll = \
                        torch.sum(nll) * (batch_size / self.gpu) / \
                        token_num
                    nll = torch.mean(nll)
                    metrics['token_num'] = token_num
                    metrics['token_nll'] = token_nll
                    metrics['nll'] = nll
                    loss = token_nll if self.func_model.token_loss else nll

                    metrics['loss'] = loss
                else:
                    loss = metrics['loss']
                self.func_model._optimize(
                    loss, do_update=False, optimizer=self.optimizer)
                metrics = {
                    k: v.cpu().detach().numpy()
                    if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items()
                }
                token_num = metrics.pop('token_num', None)
                # bow_num = metrics.pop("bow_num", None)
                elapsed = time.time() - start_time
                times.append(elapsed)
                epoch_step += 1

                tr_batch_loss += metrics['nll']
                tr_token_loss += metrics['token_nll']
                batch_metrics = {
                    k: v
                    for k, v in metrics.items() if 'token' not in k
                }
                token_metrics = {
                    k: v
                    for k, v in metrics.items() if 'token' in k
                }
                self.batch_metrics_tracker.update(batch_metrics, batch_size)
                self.token_metrics_tracker.update(token_metrics, token_num)

                if (epoch_step % self.gradient_accumulation_steps == 0) or \
                        (epoch_step == self.reader.set_stats['train']['num_training_steps_per_epoch']):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if self.log_steps > 0 and global_step % self.log_steps == 0:
                        batch_metrics_message = self.batch_metrics_tracker.value(
                        )
                        token_metrics_message = self.token_metrics_tracker.value(
                        )
                        message_prefix = f'[Train][{self.epoch}][{global_step}/{num_training_steps}]'
                        avg_time = f'AVG_Time-{sum(times[-self.log_steps:]) / self.log_steps:.3f}'
                        message = '   '.join([
                            message_prefix, batch_metrics_message,
                            token_metrics_message, avg_time
                        ])
                        self.logger.info(message)

        self.logger.info('-' * 150)
        avg_batch_loss = tr_batch_loss / epoch_step
        avg_token_loss = tr_token_loss / epoch_step
        batch_metrics_message = self.batch_metrics_tracker.summary()
        token_metrics_message = self.token_metrics_tracker.summary()
        message_prefix = f'[Valid][{self.epoch}]'
        message = '   '.join([
            message_prefix, batch_metrics_message, token_metrics_message,
            str(avg_batch_loss),
            str(avg_token_loss)
        ])
        self.logger.info(message)

        cur_valid_metric = self.batch_metrics_tracker.get(
            self.valid_metric_name)
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best:
            self.best_valid_metric = cur_valid_metric
        self.save(is_best)
        self.logger.info('-' * 150)

        return

    def infer(self, data_type='test'):
        """
        Inference interface.
        """
        self.logger.info('Generation starts ...')
        infer_save_file = os.path.join(self.save_dir,
                                       f'infer_{self.epoch}.result.json')
        infer_samples_save_file = os.path.join(
            self.save_dir, f'infer_samples_{self.epoch}.result.json')

        # Inference
        result_collection = {}
        begin_time = time.time()

        eval_data = self.reader.get_eval_data(data_type)
        set_stats = self.reader.set_stats[data_type]
        self.logger.info('***** Running Evaluation *****')
        self.logger.info('  Num Turns = %d', set_stats['num_turns'])

        with torch.no_grad():
            pbar = tqdm(eval_data)
            for dial_idx, dialog in enumerate(pbar):
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs, prompt_id = self.reader.convert_turn_eval(
                        turn, pv_turn, first_turn)
                    batch, batch_size = self.reader.collate_fn_multi_turn(
                        samples=[inputs])
                    batch = type(batch)(
                        map(lambda kv: (kv[0], self.to_tensor(kv[1])),
                            batch.items()))
                    if self.reader.use_true_curr_bspn:  # generate act, response
                        max_len = 60
                        if not self.reader.use_true_curr_aspn:
                            max_len = 80
                        outputs = self.func_model.infer(
                            inputs=batch,
                            start_id=prompt_id,
                            eos_id=self.reader.eos_r_id,
                            max_gen_len=max_len)
                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            self.logger.info(str(exception))
                            self.logger.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                    else:  # predict bspn, access db, then generate act and resp
                        outputs = self.func_model.infer(
                            inputs=batch,
                            start_id=prompt_id,
                            eos_id=self.reader.eos_b_id,
                            max_gen_len=60)
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        bspn_gen = self.decode_generated_bspn(generated_bs)
                        # check DB result
                        if self.reader.use_true_db_pointer:
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(
                                self.tokenizer.decode(bspn_gen),
                                turn['turn_domain'])
                            assert len(turn['db']) == 3
                            assert isinstance(db_result, str)
                            db = \
                                [self.reader.sos_db_id] + \
                                self.tokenizer.convert_tokens_to_ids([db_result]) + \
                                [self.reader.eos_db_id]
                            prompt_id = self.reader.sos_a_id

                        prev_input = torch.tensor(bspn_gen + db)
                        if self.func_model.use_gpu:
                            prev_input = prev_input.cuda()
                        outputs_db = self.func_model.infer(
                            inputs=batch,
                            start_id=prompt_id,
                            eos_id=self.reader.eos_r_id,
                            max_gen_len=80,
                            prev_input=prev_input)
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        try:
                            decoded = self.decode_generated_act_resp(
                                generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            self.logger.info(str(exception))
                            self.logger.info(
                                self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn[
                        'bspn'] if self.reader.use_true_curr_bspn else decoded[
                            'bspn']
                    turn['aspn_gen'] = turn[
                        'aspn'] if self.reader.use_true_curr_aspn else decoded[
                            'aspn']
                    turn['dspn_gen'] = turn['dspn']

                    pv_turn['labels'] = inputs[
                        'labels']  # all true previous context
                    pv_turn['resp'] = turn[
                        'resp'] if self.reader.use_true_prev_resp else decoded[
                            'resp']
                    if not self.reader.use_true_curr_bspn:
                        pv_turn['bspn'] = turn[
                            'bspn'] if self.reader.use_true_prev_bspn else decoded[
                                'bspn']
                        pv_turn['db'] = turn[
                            'db'] if self.reader.use_true_prev_bspn else db
                    pv_turn['aspn'] = turn[
                        'aspn'] if self.reader.use_true_prev_aspn else decoded[
                            'aspn']

                tmp_dialog_result = self.reader.inverse_transpose_turn(dialog)
                result_collection.update(tmp_dialog_result)

                # compute tmp scores
                results, _ = self.reader.wrap_result_lm(tmp_dialog_result)
                bleu, success, match = self.evaluator.validation_metric(
                    results)
                score = 0.5 * (success + match) + bleu
                pbar.set_description(
                    'match: %2.2f  success: %2.2f  bleu: %2.2f  score: %.2f' %
                    (match, success, bleu, score))

        # compute scores
        results, _ = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu

        # log results
        metrics_message = 'match: %2.2f  success: %2.2f  bleu: %2.2f  score: %.2f' % \
                          (match, success, bleu, score)
        message_prefix = f'[Infer][{self.epoch}]'
        time_cost = f'TIME-{time.time() - begin_time:.3f}'
        message = '   '.join([message_prefix, metrics_message, time_cost])
        self.logger.info(message)

        # save results
        eval_results = {
            'bleu': bleu,
            'success': success,
            'match': match,
            'score': score,
            'result': message
        }
        with open(infer_save_file, 'w') as fp:
            json.dump(eval_results, fp, indent=2)
        self.logger.info(f'Saved inference results to {infer_save_file}')
        with open(infer_samples_save_file, 'w') as fp:
            for sample in results:
                line = json.dumps(sample)
                fp.write(line)
                fp.write('\n')
        self.logger.info(
            f'Saved inference samples to {infer_samples_save_file}')

        return

    def _get_turn_domain(self, old_pv_turn, bspn_gen_ids, first_turn):

        def _get_slots(constraint):
            domain_name = ''
            slots = {}
            for item in constraint:
                if item in ontology.placeholder_tokens:
                    continue
                if item in ontology.all_domains_with_bracket:
                    domain_name = item
                    slots[domain_name] = set()
                else:
                    assert domain_name in ontology.all_domains_with_bracket
                    slots[domain_name].add(item)
            return slots

        turn_domain = []
        if first_turn and len(bspn_gen_ids) == 0:
            turn_domain = ['[general]']
            return turn_domain

        bspn_token = self.tokenizer.convert_ids_to_tokens(bspn_gen_ids)
        turn_slots = _get_slots(bspn_token)
        if first_turn:
            return list(turn_slots.keys())

        assert 'bspn' in old_pv_turn
        pv_bspn_token = self.tokenizer.convert_ids_to_tokens(
            old_pv_turn['bspn'].cpu().numpy().tolist())
        pv_turn_slots = _get_slots(pv_bspn_token)
        for domain, value in turn_slots.items():
            pv_value = pv_turn_slots[
                domain] if domain in pv_turn_slots else set()
            if len(value - pv_value) > 0 or len(pv_value - value):
                turn_domain.append(domain)
        if len(turn_domain) == 0:
            turn_domain = list(turn_slots.keys())

        return turn_domain

    def forward(self, first_turn, batch, prompt_id, labels, old_pv_turn):
        with torch.no_grad():
            batch = type(batch)(
                map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            pv_turn = {}

            outputs = self.func_model.infer(
                inputs=batch,
                start_id=prompt_id,
                eos_id=self.reader.eos_b_id,
                max_gen_len=60)
            generated_bs = outputs[0].cpu().numpy().tolist()
            bspn_gen = self.decode_generated_bspn(generated_bs)

            turn_domain = self._get_turn_domain(old_pv_turn, bspn_gen,
                                                first_turn)

            db_result = self.reader.bspan_to_DBpointer(
                self.tokenizer.decode(bspn_gen), turn_domain)
            assert isinstance(db_result, str)
            db = \
                [self.reader.sos_db_id] + \
                self.tokenizer.convert_tokens_to_ids([db_result]) + \
                [self.reader.eos_db_id]
            prompt_id = self.reader.sos_a_id
            prev_input = torch.tensor(bspn_gen + db)
            if self.func_model.use_gpu:
                prev_input = prev_input.cuda()
            outputs_db = self.func_model.infer(
                inputs=batch,
                start_id=prompt_id,
                eos_id=self.reader.eos_r_id,
                max_gen_len=80,
                prev_input=prev_input)
            generated_ar = outputs_db[0].cpu().numpy().tolist()
            decoded = self.decode_generated_act_resp(generated_ar)
            decoded['bspn'] = bspn_gen

            pv_turn['labels'] = [
                label.cpu().numpy().tolist() for label in labels
            ]
            pv_turn['resp'] = decoded['resp']
            pv_turn['bspn'] = decoded['bspn']
            pv_turn['db'] = db
            pv_turn['aspn'] = decoded['aspn']

        return pv_turn
