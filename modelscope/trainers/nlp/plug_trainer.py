import os
from typing import Union

import torch
from deepspeed import DeepSpeedEngine
from megatron_util import mpu
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
from modelscope.models.nlp.plug import DistributedPlug
from modelscope.models.nlp.plug.backbone import BertLayerNorm
from modelscope.models.nlp.plug.generator import TextGenerator
from modelscope.utils.constant import ModeKeys
from ..base import TRAINERS
from ..nlp_trainer import NlpEpochBasedTrainer


@TRAINERS.register_module(module_name=Trainers.nlp_plug_trainer)
class PlugTrainer(NlpEpochBasedTrainer):

    def build_model(self) -> Union[nn.Module, TorchModel]:
        rank = int(os.environ.get('LOCAL_RANK', -1))
        master_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '29500')
        model = DistributedPlug(
            self.model_dir,
            rank,
            master_ip=master_ip,
            master_port=master_port,
            **self.cfg.model)
        self.unwrap_module(model.model).model_dir = self.model_dir
        return model.model

    def to_parallel(self, model) -> Union[nn.Module, TorchModel]:
        from modelscope.utils.nlp.distributed import DistributedDataParallel as DDP
        return DDP(model)

    def _get_params_for_weight_decay_optimization(self, module):

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        for module_ in module.modules():
            if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
                no_weight_decay_params['params'].extend([
                    p for p in list(module_._parameters.values())
                    if p is not None
                ])
            else:
                weight_decay_params['params'].extend([
                    p for n, p in list(module_._parameters.items())
                    if p is not None and 'mask_score' not in n
                    and 'mask' not in n and n != 'bias'
                ])
                no_weight_decay_params['params'].extend([
                    p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'
                ])

        return weight_decay_params, no_weight_decay_params

    def create_optimizer_and_scheduler(self):
        optimizer, lr_scheduler = self.optimizers
        optimizer_cfg = self.cfg.train.get('optimizer', None)
        # optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        model = self.model

        embeddings = model.module.model.bert.embeddings
        layers = model.module.model.bert.encoder.layer
        dec_layers = model.module.model.decoder.decoder
        param_groups = []
        param_groups += list(
            self._get_params_for_weight_decay_optimization(layers))
        param_groups += list(
            self._get_params_for_weight_decay_optimization(embeddings))
        param_groups += list(
            self._get_params_for_weight_decay_optimization(dec_layers))

        for param_group in param_groups:
            for param in param_group['params']:
                if not hasattr(param, 'model_parallel'):
                    param.model_parallel = False
        optimizer = DeepSpeedCPUAdam(
            param_groups,
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay)

        lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)

        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
        from modelscope.models.nlp.plug.AnnealingLR import AnnealingLR
        num_iters = self.max_iters
        lr_scheduler = AnnealingLR(
            optimizer,
            start_lr=optimizer_cfg.lr,
            warmup_iter=lr_scheduler_cfg.warmup * num_iters,
            num_iters=num_iters,
            decay_style=lr_scheduler_cfg.decay_style,
            last_iter=-1)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    def _get_masks_and_position_ids(self, data, eod_token):
        # Extract batch size and sequence length.
        batch_size, seq_length = data.size()

        # Attention mask (lower triangular).
        att_mask_batch = 1
        attention_mask = torch.tril(
            torch.ones((att_mask_batch, seq_length, seq_length),
                       device=data.device)).view(att_mask_batch, 1, seq_length,
                                                 seq_length)

        # Loss mask.
        loss_mask = torch.ones(
            data.size(), dtype=torch.float, device=data.device)
        loss_mask[data == eod_token] = 0.0

        # Position ids.
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
        return attention_mask, loss_mask, position_ids

    def train_step(self, model, inputs):
        self._mode = ModeKeys.TRAIN
        # format inputs
        checkpoint_activations = getattr(self.cfg.train,
                                         'checkpoint_activations', True)
        tgt_tokens = inputs['labels'][:, :-1].contiguous()
        tgt_labels = inputs['labels'][:, 1:].contiguous()
        tgt_attention_mask, dec_loss_mask, position_ids = self._get_masks_and_position_ids(
            tgt_tokens, 0)
        if getattr(self.cfg.train, 'fp16', None):
            tgt_attention_mask = tgt_attention_mask.half()

        # forward step
        _, output = model(
            inputs['input_ids'],
            None,
            inputs['attention_mask'],
            tgt_tokens,
            position_ids,
            tgt_attention_mask,
            checkpoint_activations=checkpoint_activations)

        losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(),
                                                  tgt_labels)
        dec_loss_mask = dec_loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * dec_loss_mask) / dec_loss_mask.sum()

        # add model output info to log
        self.train_outputs = {'loss': loss}
        self.log_buffer.update(self.train_outputs)

    def evaluation_step(self, data):
        # wapper 1: DeepspeedEngine, wapper 2: DDP
        # model = self.model.module
        if isinstance(self.model, DeepSpeedEngine):
            model = self.model.module
        else:
            model = self.model

        model.eval()

        # model: fp16 wapper; model.module : distributedPlug
        vocab_size = self.unwrap_module(self.model).config.original_vocab_size
        batch_size = data['input_ids'].shape[0]
        beam_generator = TextGenerator(model,
                                       self.eval_preprocessor.nlp_tokenizer,
                                       None)

        with torch.no_grad():
            tokens = data['input_ids'].long()
            padding_mask = data['attention_mask'].byte()
            target_ids = data['labels'].long()
            target_labels = target_ids[:, 1:].contiguous()
            encoder_inputs = [tokens, None, padding_mask]
            result = beam_generator.translate_batch(encoder_inputs)
            pred_list = result['predictions']
            target_list = target_labels.cpu().numpy().tolist()
            result['preds'] = []
            data['tgts'] = []
            for i in range(batch_size):
                pred_ids = pred_list[i][0]
                pred_ids[pred_ids > vocab_size - 1] = 100
                pred_ids = pred_ids.cpu().numpy().tolist()

                gold_string = self.eval_preprocessor.decode(
                    target_list[i], skip_special_tokens=True)
                pred_string = self.eval_preprocessor.decode(
                    pred_ids, skip_special_tokens=True)
                result['preds'].append(pred_string)
                data['tgts'].append(gold_string)
        return result
