# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

from .unified_transformer import UnifiedTransformer


class GenUnifiedTransformer(UnifiedTransformer):
    """
    Implement generation unified transformer.
    """

    def __init__(self, model_dir, config, reader, generator):
        super(GenUnifiedTransformer, self).__init__(model_dir, config, reader,
                                                    generator)
        self.understand = config.BPETextField.understand
        if torch.cuda.is_available():
            self.use_gpu = True
        if self.use_gpu:
            self.cuda()
        return

    def _forward(self, inputs, is_training, with_label):
        """ Real forward process of model in different mode(train/test). """

        def cat(x, y, dim=1):
            return torch.cat([x, y], dim=dim)

        outputs = {}

        if self.understand or self.policy:
            if self.understand:
                prompt_token = inputs['understand_token']
                prompt_mask = inputs['understand_mask']
                if self.policy:
                    prompt_token = cat(prompt_token, inputs['policy_token'])
                    prompt_mask = cat(prompt_mask, inputs['policy_mask'])
            else:
                prompt_token = inputs['policy_token']
                prompt_mask = inputs['policy_mask']

            enc_embed, dec_embed, prompt_embed = self._encoder_prompt_decoder_network(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                tgt_token=inputs['tgt_token'][:, :-1],
                tgt_mask=inputs['tgt_mask'][:, :-1],
                prompt_token=prompt_token,
                prompt_mask=prompt_mask,
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'],
                tgt_pos=inputs['tgt_pos'][:, :-1],
                tgt_type=inputs['tgt_type'][:, :-1],
                tgt_turn=inputs['tgt_turn'][:, :-1])
        else:
            enc_embed, dec_embed = self._encoder_decoder_network(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                tgt_token=inputs['tgt_token'][:, :-1],
                tgt_mask=inputs['tgt_mask'][:, :-1],
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'],
                tgt_pos=inputs['tgt_pos'][:, :-1],
                tgt_type=inputs['tgt_type'][:, :-1],
                tgt_turn=inputs['tgt_turn'][:, :-1])

        outputs['dec_probs'] = self._dec_head(dec_embed=dec_embed)
        return outputs

    def _collect_metrics(self, inputs, outputs, with_label, data_file):

        metrics = {}
        loss = 0.

        label = inputs['tgt_token'][:, 1:]
        token_num = torch.sum(torch.sum(inputs['tgt_mask'], dim=1) - 1)
        nll = self.nll_loss(
            torch.log(outputs['dec_probs'] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        token_nll = torch.sum(nll) / token_num
        nll = torch.mean(nll)
        metrics['nll'] = nll
        metrics['token_nll'] = token_nll
        metrics['token_num'] = token_num
        loss = loss + (token_nll if self.token_loss else nll)

        metrics['loss'] = loss
        if self.gpu > 1:
            return nll, token_nll, token_num
        else:
            return metrics

    def _optimize(self, loss, do_update=False, optimizer=None):
        """ Optimize loss function and update model. """
        assert optimizer is not None

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.parameters(), max_norm=self.grad_clip)

        if do_update:
            optimizer.step()
            optimizer.zero_grad()

        return

    def _init_state(self,
                    src_token,
                    src_mask,
                    src_pos=None,
                    src_type=None,
                    src_turn=None):
        """ Initialize decode state. """
        state = {}
        batch_size = src_token.shape[0]

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        src_embed = self.embed_layer_norm(src_embed)

        mask = self._create_mask(src_mask, append_head=False)

        enc_out = src_embed

        cache = {}
        for _l, layer in enumerate(self.layers):
            cache[f'layer_{_l}'] = {}
            enc_out = layer(enc_out, mask, cache[f'layer_{_l}'])

        state['cache'] = cache
        state['mask'] = mask[:, :1]
        state['batch_size'] = batch_size
        shape = [batch_size, 1, 1]
        state['pred_mask'] = torch.ones(shape, dtype=torch.float32)
        state['pred_pos'] = torch.zeros(shape, dtype=torch.int64)
        state['pred_type'] = torch.zeros(shape, dtype=torch.int64)
        state['pred_turn'] = torch.zeros(shape, dtype=torch.int64)
        if self.use_gpu:
            state['pred_mask'] = state['pred_mask'].cuda()
            state['pred_pos'] = state['pred_pos'].cuda()
            state['pred_type'] = state['pred_type'].cuda()
            state['pred_turn'] = state['pred_turn'].cuda()

        return state

    def _init_prompt_state(self,
                           src_token,
                           src_mask,
                           prompt_token,
                           prompt_mask,
                           src_pos=None,
                           src_type=None,
                           src_turn=None,
                           prompt_pos=None,
                           prompt_type=None,
                           prompt_turn=None):
        """ Initialize decode state. """
        state = {}
        batch_size = src_token.shape[0]

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        prompt_embed = self.embedder(prompt_token, prompt_pos, prompt_type,
                                     prompt_turn)
        embed = torch.cat([src_embed, prompt_embed], dim=1)
        embed = self.embed_layer_norm(embed)
        enc_out = embed

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(prompt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        cache = {}
        for _l, layer in enumerate(self.layers):
            cache[f'layer_{_l}'] = {}
            enc_out = layer(enc_out, mask, cache[f'layer_{_l}'])

        state['cache'] = cache
        state['mask'] = mask[:, -1:]  # state["mask"] = mask[:, :1]
        state['batch_size'] = batch_size
        shape = [batch_size, 1, 1]
        state['pred_mask'] = torch.ones(shape, dtype=torch.float32)
        state['pred_pos'] = torch.zeros(shape, dtype=torch.int64)
        state['pred_type'] = torch.zeros(shape, dtype=torch.int64)
        state['pred_turn'] = torch.zeros(shape, dtype=torch.int64)
        if self.use_gpu:
            state['pred_mask'] = state['pred_mask'].cuda()
            state['pred_pos'] = state['pred_pos'].cuda()
            state['pred_type'] = state['pred_type'].cuda()
            state['pred_turn'] = state['pred_turn'].cuda()

        return state

    def _decode(self, state):
        """ Decoding one time stamp. """

        # shape: [batch_size, 1, seq_len]
        mask = state['mask']

        # shape: [batch_size, 1, 1]
        if self.use_gpu:
            pred_token = state['pred_token'].cuda()
            pred_mask = state['pred_mask'].cuda()
            pred_pos = state['pred_pos'].cuda()
            pred_type = state['pred_type'].cuda()
            pred_turn = state['pred_turn'].cuda()
        else:
            pred_token = state['pred_token']
            pred_mask = state['pred_mask']
            pred_pos = state['pred_pos']
            pred_type = state['pred_type']
            pred_turn = state['pred_turn']

        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state['cache']
        pred_embed = self.embedder(pred_token, pred_pos, pred_type,
                                   pred_turn).squeeze(-2)
        pred_embed = self.embed_layer_norm(pred_embed)

        # shape: [batch_size, 1, seq_len + 1]
        mask = torch.cat([mask, 1 - pred_mask], dim=2)

        # shape: [batch_size, 1, hidden_dim]
        for _l, layer in enumerate(self.layers):
            pred_embed = layer(pred_embed, mask, cache[f'layer_{_l}'])

        # shape: [batch_size, vocab_size]
        pred_probs = self._dec_head(dec_embed=pred_embed[:, 0])
        pred_logits = torch.log(pred_probs)

        state['mask'] = mask
        return pred_logits, state

    def _infer(self,
               inputs,
               start_id=None,
               eos_id=None,
               max_gen_len=None,
               prev_input=None):
        """ Real inference process of model. """

        def cat(x, y, dim=1):
            return torch.cat([x, y], dim=dim)

        # Initial decode state.
        if self.understand or self.policy:
            if self.understand:
                prompt_token = inputs['understand_token']
                prompt_mask = inputs['understand_mask']
                if self.policy:
                    prompt_token = cat(prompt_token, inputs['policy_token'])
                    prompt_mask = cat(prompt_mask, inputs['policy_mask'])
            else:
                prompt_token = inputs['policy_token']
                prompt_mask = inputs['policy_mask']

            state = self._init_prompt_state(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                prompt_token=prompt_token,
                prompt_mask=prompt_mask,
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'])
        else:
            state = self._init_state(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'])

        # Generation process.
        gen_results = self.generator(
            step_fn=self._decode,
            state=state,
            start_id=start_id,
            eos_id=eos_id,
            max_gen_len=max_gen_len,
            prev_input=prev_input)

        outputs = gen_results['preds']
        return outputs


GenUnifiedTransformer.register('GenUnifiedTransformer')
