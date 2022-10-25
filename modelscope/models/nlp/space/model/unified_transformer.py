# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.nlp.space.model.model_base import SpaceModelBase
from modelscope.models.nlp.space.modules.embedder import Embedder
from modelscope.models.nlp.space.modules.transformer_block import \
    TransformerBlock


class UnifiedTransformer(SpaceModelBase):
    """
    Implement unified transformer.
    """

    def __init__(self, model_dir, config, reader, generator, dtype='float32'):
        super(UnifiedTransformer, self).__init__(model_dir, config)
        self.reader = reader
        self.generator = generator
        self.policy = config.BPETextField.policy
        self.generation = config.BPETextField.generation
        self.num_token_embeddings = config.Model.num_token_embeddings
        self.num_pos_embeddings = config.Model.num_pos_embeddings
        self.num_type_embeddings = config.Model.num_type_embeddings
        self.num_turn_embeddings = config.Model.num_turn_embeddings
        self.temperature = config.Model.temperature
        self.hidden_dim = config.Model.hidden_dim
        self.num_heads = config.Model.num_heads
        self.num_layers = config.Model.num_layers
        self.padding_idx = config.Model.padding_idx
        self.dropout = config.Model.dropout
        self.embed_dropout = config.Model.embed_dropout
        self.attn_dropout = config.Model.attn_dropout
        self.ff_dropout = config.Model.ff_dropout
        self.mlm_ratio = config.Model.mlm_ratio
        self.mmd_ratio = config.Model.mmd_ratio
        self.pos_trainable = config.Model.pos_trainable
        self.label_smooth = config.Model.label_smooth
        self.initializer_range = config.Model.initializer_range
        self.gradient_accumulation_steps = config.Model.gradient_accumulation_steps
        self.token_loss = config.Trainer.token_loss
        self.learning_method = config.Dataset.learning_method
        self.with_contrastive = config.Dataset.with_contrastive
        self.with_query_bow = config.BPETextField.with_query_bow
        self.with_resp_bow = config.BPETextField.with_resp_bow
        self.with_pool = config.Model.with_pool
        self.with_mlm = config.Dataset.with_mlm
        self._dtype = dtype

        self.embedder = Embedder(
            self.hidden_dim,
            self.num_token_embeddings,
            self.num_pos_embeddings,
            self.num_type_embeddings,
            self.num_turn_embeddings,
            padding_idx=self.padding_idx,
            dropout=self.embed_dropout,
            pos_trainable=self.pos_trainable)
        self.embed_layer_norm = nn.LayerNorm(
            normalized_shape=self.hidden_dim,
            eps=1e-12,
            elementwise_affine=True)

        self.layers = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.num_heads, self.dropout,
                             self.attn_dropout, self.ff_dropout)
            for _ in range(config.Model.num_layers)
        ])

        if self.with_mlm:
            self.mlm_transform = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(),
                nn.LayerNorm(
                    normalized_shape=self.hidden_dim,
                    eps=1e-12,
                    elementwise_affine=True))
            self.mlm_bias = nn.Parameter(
                torch.zeros(self.num_token_embeddings))

        self.pooler = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())

        if self.with_query_bow or self.with_resp_bow:
            self.bow_predictor = nn.Linear(
                self.hidden_dim, self.num_token_embeddings, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.BCELoss(reduction='none')
        self.nll_loss = nn.NLLLoss(
            ignore_index=self.padding_idx, reduction='none')
        self._create_parameters()

        self.max_grad_norm = config.Model.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = config.Model.weight_decay

        if self.use_gpu:
            self.cuda()

        return

    def _create_parameters(self):
        """ Create model's paramters. """
        sequence_mask = np.tri(
            self.num_pos_embeddings,
            self.num_pos_embeddings,
            dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        return

    def _create_mask(self,
                     input_mask,
                     append_head=False,
                     auto_regressive=False):
        """Create attention mask.
        from sequence to matrix：[batch_size, max_seq_len， 1] -> [batch_size, max_seq_len, max_seq_len]

        Args:
            input_mask (Variable(shape: [batch_size, max_seq_len]))
            auto_regressive(bool)
        """
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            seq_mask = seq_mask.to(mask.device)
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        """Merge source attention mask and target attention mask.
        There are four parts：left upper (lu) / right upper (ru) / left below (lb) / right below (rb)

        Args:
            mask1(Variable(shape: [batch_size, max_src_len, max_src_len])) : source attention mask
            mask2(Variable(shape: [batch_size, max_tgt_len, max_tgt_len])) : target attention mask
        """
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]
        # seq_len = seq_len1 + seq_len2

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2).to(mask_lu.device)
        if self.use_gpu:
            mask_ru = mask_ru.cuda()
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _mlm_head(self, mlm_embed):
        mlm_embed = self.mlm_transform(mlm_embed)
        mlm_logits = torch.matmul(
            mlm_embed, self.embedder.token_embedding.weight.T) + self.mlm_bias
        mlm_probs = self.softmax(mlm_logits)
        return mlm_probs

    def _dec_head(self, dec_embed):
        dec_logits = torch.matmul(dec_embed,
                                  self.embedder.token_embedding.weight.T)
        dec_probs = self.softmax(dec_logits)
        return dec_probs

    def _refactor_feature(self, features):
        features = self.pooler(features) if self.with_pool else features
        batch_size = features.size(0) // 2
        features = \
            torch.cat(
                [features[:batch_size].unsqueeze(1), features[batch_size:].unsqueeze(1)],
                dim=1
            )
        features = F.normalize(features, dim=-1, p=2)
        return features

    def _encoder_network(self,
                         input_token,
                         input_mask,
                         input_pos=None,
                         input_type=None,
                         input_turn=None):
        embed = self.embedder(input_token, input_pos, input_type, input_turn)
        embed = self.embed_layer_norm(embed)
        mask = self._create_mask(input_mask, auto_regressive=False)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        return embed

    def _encoder_decoder_network(self,
                                 src_token,
                                 src_mask,
                                 tgt_token,
                                 tgt_mask,
                                 src_pos=None,
                                 src_type=None,
                                 src_turn=None,
                                 tgt_pos=None,
                                 tgt_type=None,
                                 tgt_turn=None):
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        enc_embed = embed[:, :-tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return enc_embed, dec_embed

    def _encoder_prompt_decoder_network(self,
                                        src_token,
                                        src_mask,
                                        tgt_token,
                                        tgt_mask,
                                        prompt_token,
                                        prompt_mask,
                                        src_pos=None,
                                        src_type=None,
                                        src_turn=None,
                                        tgt_pos=None,
                                        tgt_type=None,
                                        tgt_turn=None,
                                        prompt_pos=None,
                                        prompt_type=None,
                                        prompt_turn=None):
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        prompt_embed = self.embedder(prompt_token, prompt_pos, prompt_type,
                                     prompt_turn)

        embed = torch.cat([src_embed, prompt_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(
            torch.cat([prompt_mask, tgt_mask], dim=1), auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        src_len = src_token.shape[1]
        tgt_len = tgt_token.shape[1]
        enc_embed = embed[:, :src_len]
        dec_embed = embed[:, -tgt_len:]
        prompt_embed = embed[:, src_len:-tgt_len]

        return enc_embed, dec_embed, prompt_embed

    def _optimize(self, loss, optimizer=None, lr_scheduler=None):
        """ Optimize loss function and update model. """
        assert optimizer is not None
        optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.parameters(), max_norm=self.grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return

    def _infer(self,
               inputs,
               start_id=None,
               eos_id=None,
               max_gen_len=None,
               prev_input=None):
        """ Real inference process of model. """
        results = {}
        return results


UnifiedTransformer.register('UnifiedTransformer')
