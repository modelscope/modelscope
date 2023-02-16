# Part of the implementation is borrowed and modified from THUMT,
# publicly available at https://github.com/THUNLP-MT/THUMT
# Copyright 2017-2022 The Alibaba MT Team Authors. All rights reserved.
import math
from collections import namedtuple
from typing import Dict

import tensorflow as tf

from modelscope.metainfo import Models
from modelscope.models.base import Model, Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['CsanmtForTranslation']


@MODELS.register_module(Tasks.translation, module_name=Models.translation)
class CsanmtForTranslation(Model):

    def __init__(self, model_dir, *args, **kwargs):
        """
        Args:
            params (dict): the model configuration.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.params = kwargs
        print(self.params)

    def __call__(self,
                 input: Dict[str, Tensor],
                 label: Dict[str, Tensor] = None,
                 prefix: Dict[str, Tensor] = None,
                 prefix_hit: Dict[bool, Tensor] = None) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input: the preprocessed input source sequence
            label: the ground truth target data for model training
            prefix: the preprocessed input target prefix sequence for interactive translation
            prefix_hit: the preprocessed target prefix subword vector for interactive translation

        Returns:
            output_seqs: output sequence of target ids
        """
        if label is None:
            with tf.compat.v1.variable_scope('NmtModel'):
                output_seqs, output_scores = self.beam_search(
                    {
                        'input_wids': input,
                        'prefix_wids': prefix,
                        'prefix_hit': prefix_hit
                    }, self.params)
            return {
                'output_seqs': output_seqs,
                'output_scores': output_scores,
            }
        else:
            train_op, loss = self.transformer_model_train_fn(input, label)
            return {
                'train_op': train_op,
                'loss': loss,
            }

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run the forward pass for a model.

        Args:
            input (Dict[str, Tensor]): the dict of the model inputs for the forward method

        Returns:
            Dict[str, Tensor]: output from the model forward pass
        """
        ...

    def encoding_graph(self, features, params):
        src_vocab_size = params['src_vocab_size']
        hidden_size = params['hidden_size']

        initializer = tf.compat.v1.random_normal_initializer(
            0.0, hidden_size**-0.5, dtype=tf.float32)

        if params['shared_source_target_embedding']:
            with tf.compat.v1.variable_scope(
                    'Shared_Embedding', reuse=tf.compat.v1.AUTO_REUSE):
                src_embedding = tf.compat.v1.get_variable(
                    'Weights', [src_vocab_size, hidden_size],
                    initializer=initializer)
        else:
            with tf.compat.v1.variable_scope('Source_Embedding'):
                src_embedding = tf.compat.v1.get_variable(
                    'Weights', [src_vocab_size, hidden_size],
                    initializer=initializer)
        src_bias = tf.compat.v1.get_variable('encoder_input_bias',
                                             [hidden_size])

        eos_padding = tf.zeros_like(features, dtype=tf.int64)[:, :1]
        src_seq = tf.concat([features, eos_padding], 1)
        src_mask = tf.cast(tf.not_equal(src_seq, 0), dtype=tf.float32)
        shift_src_mask = src_mask[:, :-1]
        shift_src_mask = tf.pad(
            tensor=shift_src_mask,
            paddings=[[0, 0], [1, 0]],
            constant_values=1)

        encoder_input = tf.gather(src_embedding, tf.cast(src_seq, tf.int32))
        encoder_input = encoder_input * (hidden_size**0.5)
        if params['position_info_type'] == 'absolute':
            encoder_input = add_timing_signal(encoder_input)
        encoder_input = tf.multiply(encoder_input,
                                    tf.expand_dims(shift_src_mask, 2))

        encoder_input = tf.nn.bias_add(encoder_input, src_bias)
        encoder_self_attention_bias = attention_bias(shift_src_mask, 'masking')

        if params['residual_dropout'] > 0.0:
            encoder_input = tf.nn.dropout(
                encoder_input, rate=params['residual_dropout'])

        # encode
        encoder_output = transformer_encoder(encoder_input,
                                             encoder_self_attention_bias,
                                             shift_src_mask, params)
        return encoder_output, encoder_self_attention_bias

    def semantic_encoding_graph(self, features, params, name=None):
        hidden_size = params['hidden_size']
        initializer = tf.compat.v1.random_normal_initializer(
            0.0, hidden_size**-0.5, dtype=tf.float32)
        scope = None
        if params['shared_source_target_embedding']:
            vocab_size = params['src_vocab_size']
            scope = 'Shared_Semantic_Embedding'
        elif name == 'source':
            vocab_size = params['src_vocab_size']
            scope = 'Source_Semantic_Embedding'
        elif name == 'target':
            vocab_size = params['trg_vocab_size']
            scope = 'Target_Semantic_Embedding'
        else:
            raise ValueError('error: no right name specified.')

        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            embedding_mat = tf.compat.v1.get_variable(
                'Weights', [vocab_size, hidden_size], initializer=initializer)

        eos_padding = tf.zeros_like(features, dtype=tf.int64)[:, :1]
        input_seq = tf.concat([features, eos_padding], 1)
        input_mask = tf.cast(tf.not_equal(input_seq, 0), dtype=tf.float32)
        shift_input_mask = input_mask[:, :-1]
        shift_input_mask = tf.pad(
            tensor=shift_input_mask,
            paddings=[[0, 0], [1, 0]],
            constant_values=1)

        encoder_input = tf.gather(embedding_mat, tf.cast(input_seq, tf.int32))
        encoder_input = encoder_input * (hidden_size**0.5)
        encoder_input = tf.multiply(encoder_input,
                                    tf.expand_dims(shift_input_mask, 2))

        encoder_self_attention_bias = attention_bias(shift_input_mask,
                                                     'masking')

        if params['residual_dropout'] > 0.0:
            encoder_input = tf.nn.dropout(
                encoder_input, rate=params['residual_dropout'])

        # encode
        encoder_output = transformer_semantic_encoder(
            encoder_input, encoder_self_attention_bias, shift_input_mask,
            params)
        return encoder_output

    def build_contrastive_training_graph(self, features, labels, params):
        # representations
        source_name = 'source'
        target_name = 'target'
        if params['shared_source_target_embedding']:
            source_name = None
            target_name = None
        feature_output = self.semantic_encoding_graph(
            features, params, name=source_name)
        label_output = self.semantic_encoding_graph(
            labels, params, name=target_name)

        return feature_output, label_output

    def MGMC_sampling(self, x_embedding, y_embedding, params, epsilon=1e-12):
        K = params['num_of_samples']
        eta = params['eta']
        assert K % 2 == 0

        def get_samples(x_vector, y_vector):
            bias_vector = y_vector - x_vector
            w_r = tf.math.divide(
                tf.abs(bias_vector) - tf.reduce_min(
                    input_tensor=tf.abs(bias_vector), axis=2, keepdims=True)
                + epsilon,
                tf.reduce_max(
                    input_tensor=tf.abs(bias_vector), axis=2, keepdims=True)
                - tf.reduce_min(
                    input_tensor=tf.abs(bias_vector), axis=2, keepdims=True)
                + 2 * epsilon)

            R = []
            for i in range(K // 2):
                omega = eta * tf.random.normal(tf.shape(input=bias_vector), 0.0, w_r) + \
                    (1.0 - eta) * tf.random.normal(tf.shape(input=bias_vector), 0.0, 1.0)
                sample = x_vector + omega * bias_vector
                R.append(sample)
            return R

        ALL_SAMPLES = []
        ALL_SAMPLES = get_samples(x_embedding, y_embedding)
        ALL_SAMPLES.extend(get_samples(y_embedding, x_embedding))

        assert len(ALL_SAMPLES) == K

        return tf.concat(ALL_SAMPLES, axis=0)

    def decoding_graph(self,
                       encoder_output,
                       encoder_self_attention_bias,
                       labels,
                       params={},
                       embedding_augmentation=None):
        trg_vocab_size = params['trg_vocab_size']
        hidden_size = params['hidden_size']

        initializer = tf.compat.v1.random_normal_initializer(
            0.0, hidden_size**-0.5, dtype=tf.float32)

        if params['shared_source_target_embedding']:
            with tf.compat.v1.variable_scope(
                    'Shared_Embedding', reuse=tf.compat.v1.AUTO_REUSE):
                trg_embedding = tf.compat.v1.get_variable(
                    'Weights', [trg_vocab_size, hidden_size],
                    initializer=initializer)
        else:
            with tf.compat.v1.variable_scope('Target_Embedding'):
                trg_embedding = tf.compat.v1.get_variable(
                    'Weights', [trg_vocab_size, hidden_size],
                    initializer=initializer)

        eos_padding = tf.zeros_like(labels, dtype=tf.int64)[:, :1]
        trg_seq = tf.concat([labels, eos_padding], 1)
        trg_mask = tf.cast(tf.not_equal(trg_seq, 0), dtype=tf.float32)
        shift_trg_mask = trg_mask[:, :-1]
        shift_trg_mask = tf.pad(
            tensor=shift_trg_mask,
            paddings=[[0, 0], [1, 0]],
            constant_values=1)

        decoder_input = tf.gather(trg_embedding, tf.cast(trg_seq, tf.int32))

        decoder_input *= hidden_size**0.5
        decoder_self_attention_bias = attention_bias(
            tf.shape(input=decoder_input)[1], 'causal')
        decoder_input = tf.pad(
            tensor=decoder_input, paddings=[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        if params['position_info_type'] == 'absolute':
            decoder_input = add_timing_signal(decoder_input)

        decoder_input = tf.nn.dropout(
            decoder_input, rate=1 - (1.0 - params['residual_dropout']))

        # training
        decoder_output, attention_weights = transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_self_attention_bias,
            states_key=None,
            states_val=None,
            embedding_augmentation=embedding_augmentation,
            params=params)

        logits = self.prediction(decoder_output, params)

        on_value = params['confidence']
        off_value = (1.0 - params['confidence']) / tf.cast(
            trg_vocab_size - 1, dtype=tf.float32)
        soft_targets = tf.one_hot(
            tf.cast(trg_seq, tf.int32),
            depth=trg_vocab_size,
            on_value=on_value,
            off_value=off_value)
        mask = tf.cast(shift_trg_mask, logits.dtype)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.stop_gradient(soft_targets)) * mask
        loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(
            input_tensor=mask)

        return loss

    def build_training_graph(self,
                             features,
                             labels,
                             params,
                             feature_embedding=None,
                             label_embedding=None):
        # encode
        encoder_output, encoder_self_attention_bias = self.encoding_graph(
            features, params)
        embedding_augmentation = None
        if feature_embedding is not None and label_embedding is not None:
            embedding_augmentation = self.MGMC_sampling(
                feature_embedding, label_embedding, params)

            encoder_output = tf.tile(encoder_output,
                                     [params['num_of_samples'], 1, 1])
            encoder_self_attention_bias = tf.tile(
                encoder_self_attention_bias,
                [params['num_of_samples'], 1, 1, 1])
            labels = tf.tile(labels, [params['num_of_samples'], 1])

        # decode
        loss = self.decoding_graph(
            encoder_output,
            encoder_self_attention_bias,
            labels,
            params,
            embedding_augmentation=embedding_augmentation)

        return loss

    def transformer_model_train_fn(self, features, labels):
        initializer = get_initializer(self.params)
        with tf.compat.v1.variable_scope('NmtModel', initializer=initializer):
            num_gpus = self.params['num_gpus']
            gradient_clip_norm = self.params['gradient_clip_norm']
            global_step = tf.compat.v1.train.get_global_step()
            print(global_step)

            # learning rate
            learning_rate = get_learning_rate_decay(
                self.params['learning_rate'], global_step, self.params)
            learning_rate = tf.convert_to_tensor(
                value=learning_rate, dtype=tf.float32)

            # optimizer
            if self.params['optimizer'] == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                    learning_rate)
            elif self.params['optimizer'] == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate,
                    beta1=self.params['adam_beta1'],
                    beta2=self.params['adam_beta2'],
                    epsilon=self.params['adam_epsilon'])
            else:
                tf.compat.v1.logging.info('optimizer not supported')
                sys.exit()
            opt = MultiStepOptimizer(optimizer, self.params['update_cycle'])

            def fill_gpus(inputs, num_gpus):
                outputs = inputs
                for i in range(num_gpus):
                    outputs = tf.concat([outputs, inputs], axis=0)
                outputs = outputs[:num_gpus, ]
                return outputs

            features = tf.cond(
                pred=tf.shape(input=features)[0] < num_gpus,
                true_fn=lambda: fill_gpus(features, num_gpus),
                false_fn=lambda: features)
            labels = tf.cond(
                pred=tf.shape(input=labels)[0] < num_gpus,
                true_fn=lambda: fill_gpus(labels, num_gpus),
                false_fn=lambda: labels)

            if num_gpus > 0:
                feature_shards = shard_features(features, num_gpus)
                label_shards = shard_features(labels, num_gpus)
            else:
                feature_shards = [features]
                label_shards = [labels]

            if num_gpus > 0:
                devices = ['gpu:%d' % d for d in range(num_gpus)]
            else:
                devices = ['cpu:0']
            multi_grads = []
            sharded_losses = []

            for i, device in enumerate(devices):
                with tf.device(device), tf.compat.v1.variable_scope(
                        tf.compat.v1.get_variable_scope(),
                        reuse=True if i > 0 else None):
                    with tf.name_scope('%s_%d' % ('GPU', i)):
                        feature_output, label_output = self.build_contrastive_training_graph(
                            feature_shards[i], label_shards[i], self.params)
                        mle_loss = self.build_training_graph(
                            feature_shards[i], label_shards[i], self.params,
                            feature_output, label_output)
                        sharded_losses.append(mle_loss)
                        tf.compat.v1.summary.scalar('mle_loss_{}'.format(i),
                                                    mle_loss)

                        # Optimization
                        trainable_vars_list = [
                            v for v in tf.compat.v1.trainable_variables()
                            if 'Semantic_Embedding' not in v.name
                            and 'mini_xlm_encoder' not in v.name
                        ]
                        grads_and_vars = opt.compute_gradients(
                            mle_loss,
                            var_list=trainable_vars_list,
                            colocate_gradients_with_ops=True)
                        multi_grads.append(grads_and_vars)

            total_loss = tf.add_n(sharded_losses) / len(sharded_losses)

            # Average gradients
            grads_and_vars = average_gradients(multi_grads)

            if gradient_clip_norm > 0.0:
                grads, var_list = list(zip(*grads_and_vars))
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip_norm)
                grads_and_vars = zip(grads, var_list)

            train_op = opt.apply_gradients(
                grads_and_vars,
                global_step=tf.compat.v1.train.get_global_step())

            return train_op, total_loss

    def prediction(self, decoder_output, params):
        hidden_size = params['hidden_size']
        trg_vocab_size = params['trg_vocab_size']

        if params['shared_embedding_and_softmax_weights']:
            embedding_scope = 'Shared_Embedding' if params[
                'shared_source_target_embedding'] else 'Target_Embedding'
            with tf.compat.v1.variable_scope(embedding_scope, reuse=True):
                weights = tf.compat.v1.get_variable('Weights')
        else:
            weights = tf.compat.v1.get_variable('Softmax',
                                                [tgt_vocab_size, hidden_size])
        shape = tf.shape(input=decoder_output)[:-1]
        decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
        logits = tf.matmul(decoder_output, weights, transpose_b=True)
        logits = tf.reshape(logits, tf.concat([shape, [trg_vocab_size]], 0))
        return logits

    def inference_func(self,
                       encoder_output,
                       feature_output,
                       encoder_self_attention_bias,
                       trg_seq,
                       states_key,
                       states_val,
                       params={},
                       is_prefix=False):
        trg_vocab_size = params['trg_vocab_size']
        hidden_size = params['hidden_size']

        initializer = tf.compat.v1.random_normal_initializer(
            0.0, hidden_size**-0.5, dtype=tf.float32)

        if params['shared_source_target_embedding']:
            with tf.compat.v1.variable_scope(
                    'Shared_Embedding', reuse=tf.compat.v1.AUTO_REUSE):
                trg_embedding = tf.compat.v1.get_variable(
                    'Weights', [trg_vocab_size, hidden_size],
                    initializer=initializer)
        else:
            with tf.compat.v1.variable_scope('Target_Embedding'):
                trg_embedding = tf.compat.v1.get_variable(
                    'Weights', [trg_vocab_size, hidden_size],
                    initializer=initializer)

        decoder_input = tf.gather(trg_embedding, tf.cast(trg_seq, tf.int32))
        decoder_input *= hidden_size**0.5
        decoder_self_attention_bias = attention_bias(
            tf.shape(input=decoder_input)[1], 'causal')
        decoder_input = tf.pad(
            tensor=decoder_input, paddings=[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        if params['position_info_type'] == 'absolute':
            decoder_input = add_timing_signal(decoder_input)
        if not is_prefix:
            decoder_input = decoder_input[:, -1:, :]
            decoder_self_attention_bias = decoder_self_attention_bias[:, :,
                                                                      -1:, :]
        decoder_output, attention_weights = transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_self_attention_bias,
            states_key=states_key,
            states_val=states_val,
            embedding_augmentation=feature_output,
            params=params)
        if not is_prefix:
            decoder_output_last = decoder_output[:, -1, :]
            attention_weights_last = attention_weights[:, -1, :]
        else:
            decoder_output_last = decoder_output
            attention_weights_last = attention_weights

        if params['shared_embedding_and_softmax_weights']:
            embedding_scope = \
                'Shared_Embedding' if params['shared_source_target_embedding'] else 'Target_Embedding'
            with tf.compat.v1.variable_scope(embedding_scope, reuse=True):
                weights = tf.compat.v1.get_variable('Weights')
        else:
            weights = tf.compat.v1.get_variable('Softmax',
                                                [trg_vocab_size, hidden_size])
        logits = tf.matmul(decoder_output_last, weights, transpose_b=True)
        log_prob = tf.nn.log_softmax(logits)
        return log_prob, attention_weights_last, states_key, states_val

    def beam_search(self, features, params):
        beam_size = params['beam_size']
        trg_vocab_size = params['trg_vocab_size']
        hidden_size = params['hidden_size']
        num_decoder_layers = params['num_decoder_layers']
        lp_rate = params['lp_rate']
        max_decoded_trg_len = params['max_decoded_trg_len']
        src_input = features['input_wids']
        if 'prefix_wids' in features:
            prefix = features['prefix_wids']
            prefix_hit = features['prefix_hit']
        else:
            prefix = None
            prefix_hit = None
        batch_size = tf.shape(src_input)[0]

        src_input = tile_to_beam_size(src_input, beam_size)
        src_input = merge_first_two_dims(src_input)
        if prefix is not None:
            prefix = tf.cast(tile_to_beam_size(prefix, beam_size), tf.int32)
            prefix_hit = tile_to_beam_size(prefix_hit, beam_size)

        encoder_output, encoder_self_attention_bias = self.encoding_graph(
            src_input, params)
        source_name = 'source'
        if params['shared_source_target_embedding']:
            source_name = None
        feature_output = self.semantic_encoding_graph(
            src_input, params, name=source_name)

        states_key = [
            tf.fill([batch_size, 0, hidden_size], 0.0)
            for layer in range(num_decoder_layers)
        ]
        states_val = [
            tf.fill([batch_size, 0, hidden_size], 0.0)
            for layer in range(num_decoder_layers)
        ]
        for layer in range(num_decoder_layers):
            states_key[layer].set_shape(
                tf.TensorShape([None, None, hidden_size]))
            states_val[layer].set_shape(
                tf.TensorShape([None, None, hidden_size]))
        states_key = [
            tile_to_beam_size(states_key[layer], beam_size)
            for layer in range(num_decoder_layers)
        ]
        states_val = [
            tile_to_beam_size(states_val[layer], beam_size)
            for layer in range(num_decoder_layers)
        ]
        fixed_length = 1
        if prefix is not None:
            init_seqs = tf.concat(
                [prefix, tf.fill([batch_size, beam_size, 1], 0)], axis=2)
            fixed_length = tf.shape(init_seqs)[-1]
            flat_seqs = merge_first_two_dims(init_seqs)
            flat_states_key = [
                merge_first_two_dims(states_key[layer])
                for layer in range(num_decoder_layers)
            ]
            flat_states_val = [
                merge_first_two_dims(states_val[layer])
                for layer in range(num_decoder_layers)
            ]

            step_log_probs, step_attn_weights, step_states_key, step_states_val = self.inference_func(
                encoder_output,
                feature_output,
                encoder_self_attention_bias,
                flat_seqs,
                flat_states_key,
                flat_states_val,
                params=params,
                is_prefix=True)

            states_key = [
                split_first_two_dims(step_states_key[layer], batch_size,
                                     beam_size)
                for layer in range(num_decoder_layers)
            ]
            states_val = [
                split_first_two_dims(step_states_val[layer], batch_size,
                                     beam_size)
                for layer in range(num_decoder_layers)
            ]

            prefix_hit = merge_first_two_dims(prefix_hit)
            log_probs = tf.where(
                prefix_hit, step_log_probs[:, -1, :],
                tf.ones_like(step_log_probs[:, -1, :]) * tf.float32.min)

            init_seqs = tf.concat([
                flat_seqs[:, :-1],
                tf.expand_dims(
                    tf.cast(tf.argmax(log_probs, -1), tf.int32), -1)
            ], -1)

            init_seqs = split_first_two_dims(init_seqs, batch_size, beam_size)
            init_seqs = tf.concat(
                [init_seqs, tf.fill([batch_size, beam_size, 1], 0)], axis=2)
        else:
            init_seqs = tf.fill([batch_size, beam_size, 1], 0)

        init_log_probs = \
            tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])
        init_log_probs = tf.tile(init_log_probs, [batch_size, 1])
        init_scores = tf.zeros_like(init_log_probs)
        fin_seqs = init_seqs
        fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)
        fin_flags = tf.cast(tf.fill([batch_size, beam_size], 0), tf.bool)

        state = BeamSearchState(
            inputs=(init_seqs, init_log_probs, init_scores),
            state=(states_key, states_val),
            finish=(fin_flags, fin_seqs, fin_scores),
        )

        def _beam_search_step(time, state):
            seqs, log_probs = state.inputs[:2]
            states_key, states_val = state.state

            flat_seqs = merge_first_two_dims(seqs)
            flat_states_key = [
                merge_first_two_dims(states_key[layer])
                for layer in range(num_decoder_layers)
            ]
            flat_states_val = [
                merge_first_two_dims(states_val[layer])
                for layer in range(num_decoder_layers)
            ]

            step_log_probs, step_attn_weights, step_states_key, step_states_val = self.inference_func(
                encoder_output,
                feature_output,
                encoder_self_attention_bias,
                flat_seqs,
                flat_states_key,
                flat_states_val,
                params=params,
                is_prefix=False)

            step_log_probs = split_first_two_dims(step_log_probs, batch_size,
                                                  beam_size)
            curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs

            next_states_key = [
                split_first_two_dims(step_states_key[layer], batch_size,
                                     beam_size)
                for layer in range(num_decoder_layers)
            ]
            next_states_val = [
                split_first_two_dims(step_states_val[layer], batch_size,
                                     beam_size)
                for layer in range(num_decoder_layers)
            ]

            # Apply length penalty
            length_penalty = tf.pow(
                (5.0 + tf.cast(time + 1, dtype=tf.float32)) / 6.0, lp_rate)
            curr_scores = curr_log_probs / length_penalty

            # Select top-k candidates
            # [batch_size, beam_size * vocab_size]
            curr_scores = tf.reshape(curr_scores,
                                     [-1, beam_size * trg_vocab_size])
            # [batch_size, 2 * beam_size]
            top_scores, top_indices = tf.nn.top_k(curr_scores, k=2 * beam_size)
            # Shape: [batch_size, 2 * beam_size]
            beam_indices = top_indices // trg_vocab_size
            symbol_indices = top_indices % trg_vocab_size
            # Expand sequences
            # [batch_size, 2 * beam_size, time]
            candidate_seqs = gather_2d(seqs, beam_indices)
            candidate_seqs = tf.concat(
                [candidate_seqs[:, :, :-1],
                 tf.expand_dims(symbol_indices, 2)],
                axis=2)
            pad_seqs = tf.fill([batch_size, 2 * beam_size, 1],
                               tf.constant(0, tf.int32))
            candidate_seqs = tf.concat([candidate_seqs, pad_seqs], axis=2)

            # Expand sequences
            # Suppress finished sequences
            flags = tf.equal(symbol_indices, 0)
            # [batch, 2 * beam_size]
            alive_scores = top_scores + tf.cast(
                flags, dtype=tf.float32) * tf.float32.min
            # [batch, beam_size]
            alive_scores, alive_indices = tf.nn.top_k(alive_scores, beam_size)
            alive_symbols = gather_2d(symbol_indices, alive_indices)
            alive_indices = gather_2d(beam_indices, alive_indices)
            alive_seqs = gather_2d(seqs, alive_indices)
            alive_seqs = tf.concat(
                [alive_seqs[:, :, :-1],
                 tf.expand_dims(alive_symbols, 2)],
                axis=2)
            pad_seqs = tf.fill([batch_size, beam_size, 1],
                               tf.constant(0, tf.int32))
            alive_seqs = tf.concat([alive_seqs, pad_seqs], axis=2)
            alive_states_key = [
                gather_2d(next_states_key[layer], alive_indices)
                for layer in range(num_decoder_layers)
            ]
            alive_states_val = [
                gather_2d(next_states_val[layer], alive_indices)
                for layer in range(num_decoder_layers)
            ]
            alive_log_probs = alive_scores * length_penalty

            # Select finished sequences
            prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
            # [batch, 2 * beam_size]
            step_fin_scores = top_scores + (
                1.0 - tf.cast(flags, dtype=tf.float32)) * tf.float32.min
            # [batch, 3 * beam_size]
            fin_flags = tf.concat([prev_fin_flags, flags], axis=1)
            fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)
            # [batch, beam_size]
            fin_scores, fin_indices = tf.nn.top_k(fin_scores, beam_size)
            fin_flags = gather_2d(fin_flags, fin_indices)
            pad_seqs = tf.fill([batch_size, beam_size, 1],
                               tf.constant(0, tf.int32))
            prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)
            fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)
            fin_seqs = gather_2d(fin_seqs, fin_indices)

            new_state = BeamSearchState(
                inputs=(alive_seqs, alive_log_probs, alive_scores),
                state=(alive_states_key, alive_states_val),
                finish=(fin_flags, fin_seqs, fin_scores),
            )

            return time + 1, new_state

        def _is_finished(t, s):
            log_probs = s.inputs[1]
            finished_flags = s.finish[0]
            finished_scores = s.finish[2]
            max_lp = tf.pow(
                ((5.0 + tf.cast(max_decoded_trg_len, dtype=tf.float32)) / 6.0),
                lp_rate)
            best_alive_score = log_probs[:, 0] / max_lp
            worst_finished_score = tf.reduce_min(
                input_tensor=finished_scores
                * tf.cast(finished_flags, dtype=tf.float32),
                axis=1)
            add_mask = 1.0 - tf.cast(
                tf.reduce_any(input_tensor=finished_flags, axis=1),
                dtype=tf.float32)
            worst_finished_score += tf.float32.min * add_mask
            bound_is_met = tf.reduce_all(
                input_tensor=tf.greater(worst_finished_score,
                                        best_alive_score))

            cond = tf.logical_and(
                tf.less(t, max_decoded_trg_len), tf.logical_not(bound_is_met))

            return cond

        def _loop_fn(t, s):
            outs = _beam_search_step(t, s)
            return outs

        time = tf.constant(0, name='time')
        shape_invariants = BeamSearchState(
            inputs=(tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None]), tf.TensorShape([None,
                                                                  None])),
            state=([
                tf.TensorShape([None, None, None, hidden_size])
                for layer in range(num_decoder_layers)
            ], [
                tf.TensorShape([None, None, None, hidden_size])
                for layer in range(num_decoder_layers)
            ]),
            finish=(tf.TensorShape([None,
                                    None]), tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None])))
        outputs = tf.while_loop(
            cond=_is_finished,
            body=_loop_fn,
            loop_vars=[time, state],
            shape_invariants=[tf.TensorShape([]), shape_invariants],
            parallel_iterations=1,
            back_prop=False)

        final_state = outputs[1]
        alive_seqs = final_state.inputs[0]
        alive_scores = final_state.inputs[2]
        final_flags = final_state.finish[0]
        final_seqs = final_state.finish[1]
        final_scores = final_state.finish[2]

        alive_seqs.set_shape([None, beam_size, None])
        final_seqs.set_shape([None, beam_size, None])

        final_seqs = tf.compat.v1.where(
            tf.reduce_any(input_tensor=final_flags, axis=1), final_seqs,
            alive_seqs)
        final_scores = tf.compat.v1.where(
            tf.reduce_any(input_tensor=final_flags, axis=1), final_scores,
            alive_scores)

        final_seqs = final_seqs[:, :, fixed_length - 1:-1]
        return final_seqs, final_scores


class BeamSearchState(
        namedtuple('BeamSearchState', ('inputs', 'state', 'finish'))):
    pass


def tile_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def infer_shape(x):
    x = tf.convert_to_tensor(x)

    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


def split_first_two_dims(tensor, dim_0, dim_1):
    shape = infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)


def merge_first_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)


def gather_2d(params, indices, name=None):
    """ Gather the 2nd dimension given indices
    :param params: A tensor with shape [batch_size, M, ...]
    :param indices: A tensor with shape [batch_size, N]
    :param name: An optional string
    :return: A tensor with shape [batch_size, N, ...]
    """
    batch_size = tf.shape(params)[0]
    range_size = tf.shape(indices)[1]
    batch_pos = tf.range(batch_size * range_size) // range_size
    batch_pos = tf.reshape(batch_pos, [batch_size, range_size])
    indices = tf.stack([batch_pos, indices], axis=-1)
    output = tf.gather_nd(params, indices, name=name)

    return output


def linear(inputs, output_size, bias, concat=True, dtype=None, scope=None):
    with tf.compat.v1.variable_scope(
            scope, default_name='linear', values=[inputs], dtype=dtype):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1] for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError('inputs and input_size unmatched!')

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1]]) for inp in inputs]

        results = []
        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.compat.v1.get_variable('matrix', shape)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = 'matrix_%d' % i
                matrix = tf.compat.v1.get_variable(name, shape)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.compat.v1.get_variable('bias', shape)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output


def layer_norm(inputs, epsilon=1e-6, name=None, reuse=None):
    with tf.compat.v1.variable_scope(
            name, default_name='layer_norm', values=[inputs], reuse=reuse):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.compat.v1.get_variable(
            'layer_norm_scale', [channel_size],
            initializer=tf.ones_initializer())

        offset = tf.compat.v1.get_variable(
            'layer_norm_offset', [channel_size],
            initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, -1, True)
        variance = tf.reduce_mean(tf.square(inputs - mean), -1, True)

        norm_inputs = (inputs - mean) * tf.compat.v1.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def _layer_process(x, mode):
    if not mode or mode == 'none':
        return x
    elif mode == 'layer_norm':
        return layer_norm(x)
    else:
        raise ValueError('Unknown mode %s' % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, rate=1 - (keep_prob))
    return x + y


def embedding_augmentation_layer(x, embedding_augmentation, params, name=None):
    hidden_size = params['hidden_size']
    keep_prob = 1.0 - params['relu_dropout']
    with tf.compat.v1.variable_scope(
            name,
            default_name='embedding_augmentation_layer',
            values=[x, embedding_augmentation]):
        with tf.compat.v1.variable_scope('input_layer'):
            hidden = linear(embedding_augmentation, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, rate=1 - (keep_prob))

        with tf.compat.v1.variable_scope('output_layer'):
            output = linear(hidden, hidden_size, True, True)

        return x + output


def transformer_ffn_layer(x, params, name=None):
    filter_size = params['filter_size']
    hidden_size = params['hidden_size']
    keep_prob = 1.0 - params['relu_dropout']
    with tf.compat.v1.variable_scope(
            name, default_name='ffn_layer', values=[x]):
        with tf.compat.v1.variable_scope('input_layer'):
            hidden = linear(x, filter_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, rate=1 - (keep_prob))

        with tf.compat.v1.variable_scope('output_layer'):
            output = linear(hidden, hidden_size, True, True)

        return output


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        mask,
                        params={},
                        name='encoder'):
    num_encoder_layers = params['num_encoder_layers']
    hidden_size = params['hidden_size']
    num_heads = params['num_heads']
    residual_dropout = params['residual_dropout']
    attention_dropout = params['attention_dropout']
    layer_preproc = params['layer_preproc']
    layer_postproc = params['layer_postproc']
    x = encoder_input
    mask = tf.expand_dims(mask, 2)
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for layer in range(num_encoder_layers):
            with tf.compat.v1.variable_scope('layer_%d' % layer):
                max_relative_dis = params['max_relative_dis'] \
                    if params['position_info_type'] == 'relative' else None
                o, w = multihead_attention(
                    _layer_process(x, layer_preproc),
                    None,
                    encoder_self_attention_bias,
                    hidden_size,
                    hidden_size,
                    hidden_size,
                    num_heads,
                    attention_dropout,
                    max_relative_dis=max_relative_dis,
                    name='encoder_self_attention')
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                o = transformer_ffn_layer(
                    _layer_process(x, layer_preproc), params)
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                x = tf.multiply(x, mask)

        return _layer_process(x, layer_preproc)


def transformer_semantic_encoder(encoder_input,
                                 encoder_self_attention_bias,
                                 mask,
                                 params={},
                                 name='mini_xlm_encoder'):
    num_encoder_layers = params['num_semantic_encoder_layers']
    hidden_size = params['hidden_size']
    num_heads = params['num_heads']
    residual_dropout = params['residual_dropout']
    attention_dropout = params['attention_dropout']
    layer_preproc = params['layer_preproc']
    layer_postproc = params['layer_postproc']
    x = encoder_input
    mask = tf.expand_dims(mask, 2)
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for layer in range(num_encoder_layers):
            with tf.compat.v1.variable_scope('layer_%d' % layer):
                max_relative_dis = params['max_relative_dis']
                o, w = multihead_attention(
                    _layer_process(x, layer_preproc),
                    None,
                    encoder_self_attention_bias,
                    hidden_size,
                    hidden_size,
                    hidden_size,
                    num_heads,
                    attention_dropout,
                    max_relative_dis=max_relative_dis,
                    name='encoder_self_attention')
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                o = transformer_ffn_layer(
                    _layer_process(x, layer_preproc), params)
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                x = tf.multiply(x, mask)

        with tf.compat.v1.variable_scope(
                'pooling_layer', reuse=tf.compat.v1.AUTO_REUSE):
            output = tf.reduce_sum(
                input_tensor=x, axis=1) / tf.reduce_sum(
                    input_tensor=mask, axis=1)
            output = linear(
                tf.expand_dims(output, axis=1), hidden_size, True, True)

        return _layer_process(output, layer_preproc)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        states_key=None,
                        states_val=None,
                        embedding_augmentation=None,
                        params={},
                        name='decoder'):
    num_decoder_layers = params['num_decoder_layers']
    hidden_size = params['hidden_size']
    num_heads = params['num_heads']
    residual_dropout = params['residual_dropout']
    attention_dropout = params['attention_dropout']
    layer_preproc = params['layer_preproc']
    layer_postproc = params['layer_postproc']
    x = decoder_input
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for layer in range(num_decoder_layers):
            with tf.compat.v1.variable_scope('layer_%d' % layer):
                max_relative_dis = params['max_relative_dis'] \
                    if params['position_info_type'] == 'relative' else None
                # continuous semantic augmentation
                if embedding_augmentation is not None:
                    x = embedding_augmentation_layer(
                        x, _layer_process(embedding_augmentation,
                                          layer_preproc), params)
                    x = _layer_process(x, layer_postproc)
                o, w = multihead_attention(
                    _layer_process(x, layer_preproc),
                    None,
                    decoder_self_attention_bias,
                    hidden_size,
                    hidden_size,
                    hidden_size,
                    num_heads,
                    attention_dropout,
                    states_key=states_key,
                    states_val=states_val,
                    layer=layer,
                    max_relative_dis=max_relative_dis,
                    name='decoder_self_attention')
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                o, w = multihead_attention(
                    _layer_process(x, layer_preproc),
                    encoder_output,
                    encoder_decoder_attention_bias,
                    hidden_size,
                    hidden_size,
                    hidden_size,
                    num_heads,
                    attention_dropout,
                    max_relative_dis=max_relative_dis,
                    name='encdec_attention')
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

                o = transformer_ffn_layer(
                    _layer_process(x, layer_preproc), params)
                x = _residual_fn(x, o, 1.0 - residual_dropout)
                x = _layer_process(x, layer_postproc)

        return _layer_process(x, layer_preproc), w


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = channels // 2

    log_timescale_increment = \
        (math.log(float(max_timescale) / float(min_timescale)) / (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32)
        * -log_timescale_increment)

    scaled_time = \
        tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.compat.v1.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])

    return x + tf.cast(signal, x.dtype)


def attention_bias(inputs, mode, inf=-1e9, dtype=None):
    if dtype is None:
        dtype = tf.float32

    if dtype != tf.float32:
        inf = dtype.min

    if mode == 'masking':
        mask = inputs
        ret = (1.0 - mask) * inf
        ret = tf.expand_dims(tf.expand_dims(ret, 1), 1)

    elif mode == 'causal':
        length = inputs
        lower_triangle = tf.linalg.band_part(
            tf.fill([length, length], 1.0), -1, 0)
        ret = inf * (1.0 - lower_triangle)
        ret = tf.reshape(ret, [1, 1, length, length])
    else:
        raise ValueError('Unknown mode %s' % mode)

    return tf.cast(ret, dtype)


def split_heads(x, num_heads):
    n = num_heads
    old_shape = x.get_shape().dims
    ndims = x.shape.ndims

    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    perm = [0, ndims - 1] + [i for i in range(1, ndims - 1)] + [ndims]
    return tf.transpose(ret, perm)


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name=None,
                          rpr=None):
    with tf.compat.v1.variable_scope(
            name, default_name='dot_product_attention', values=[q, k, v]):
        q_shape = tf.shape(q)
        bs, hd, lq, dk = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
        lk = tf.shape(k)[2]
        dv = tf.shape(v)[3]

        if rpr is not None:
            rpr_k, rpr_v = rpr['rpr_k'], rpr[
                'rpr_v']  # (lq, lk, dk), (lq, lk, dv)

        if rpr is None:
            logits = tf.matmul(q, k, transpose_b=True)
        else:  # self-attention with relative position representaion
            logits_part1 = tf.matmul(q, k, transpose_b=True)  # bs, hd, lq, lk

            q = tf.reshape(tf.transpose(q, [2, 0, 1, 3]),
                           [lq, bs * hd, dk])  # lq, bs*hd, dk
            logits_part2 = tf.matmul(q,
                                     tf.transpose(rpr_k,
                                                  [0, 2, 1]))  # lq, bs*hd, lk
            logits_part2 = tf.reshape(
                tf.transpose(logits_part2, [1, 0, 2]), [bs, hd, lq, lk])

            logits = logits_part1 + logits_part2  # bs, hd, lq, lk

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name='attention_weights')

        if dropout_rate > 0.0:
            weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

        if rpr is None:
            return tf.matmul(weights, v), weights
        else:
            outputs_part1 = tf.matmul(weights, v)  # bs, hd, lq, dv

            weights = tf.reshape(
                tf.transpose(weights, [2, 0, 1, 3]),
                [lq, bs * hd, lk])  # lq, bs*hd, lk
            outputs_part2 = tf.matmul(weights, rpr_v)  # lq, bs*hd, dv
            outputs_part2 = tf.reshape(
                tf.transpose(outputs_part2, [1, 0, 2]), [bs, hd, lq, dv])

            outputs = outputs_part1 + outputs_part2  # bs, hd, lq, dv
            weights = tf.reshape(
                tf.transpose(weights, [1, 0, 2]),
                [bs, hd, lq, lk])  # bs, hd, lq, lk

            return outputs, weights


def combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    x.set_shape(new_shape)

    return x


def create_rpr(orginal_var,
               length_q,
               length_kv,
               max_relative_dis,
               name='create_rpr'):
    with tf.name_scope(name):
        idxs = tf.reshape(tf.range(length_kv), [-1, 1])  # only self-attention
        idys = tf.reshape(tf.range(length_kv), [1, -1])
        ids = idxs - idys
        ids = ids + max_relative_dis
        ids = tf.maximum(ids, 0)
        ids = tf.minimum(ids, 2 * max_relative_dis)
        ids = ids[-length_q:, :]
        rpr = tf.gather(orginal_var, ids)
        return rpr


def multihead_attention(queries,
                        memories,
                        bias,
                        key_depth,
                        value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        states_key=None,
                        states_val=None,
                        layer=0,
                        max_relative_dis=None,
                        name=None):
    if key_depth % num_heads != 0:
        raise ValueError(
            'Key size (%d) must be divisible by the number of attention heads (%d).'
            % (key_size, num_heads))

    if value_depth % num_heads != 0:
        raise ValueError(
            'Value size (%d) must be divisible by the number of attention heads (%d).'
            % (value_size, num_heads))

    with tf.compat.v1.variable_scope(
            name, default_name='multihead_attention',
            values=[queries, memories]):
        if memories is None:
            # self attention
            combined = linear(
                queries,
                key_depth * 2 + value_depth,
                True,
                True,
                scope='qkv_transform')
            q, k, v = tf.split(
                combined, [key_depth, key_depth, value_depth], axis=2)
        else:
            q = linear(queries, key_depth, True, True, scope='q_transform')
            combined = linear(
                memories,
                key_depth + value_depth,
                True,
                True,
                scope='kv_transform')
            k, v = tf.split(combined, [key_depth, value_depth], axis=2)

        if states_key is not None:
            k = states_key[layer] = tf.concat([states_key[layer], k], axis=1)
        if states_val is not None:
            v = states_val[layer] = tf.concat([states_val[layer], v], axis=1)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        key_depth_per_head = key_depth // num_heads
        q *= key_depth_per_head**-0.5

        length_q = tf.shape(q)[2]
        length_kv = tf.shape(k)[2]

        # relative position representation (only in self-attention)
        if memories is None and max_relative_dis is not None:
            rpr_k = tf.compat.v1.get_variable(
                'rpr_k', [2 * max_relative_dis + 1, key_depth // num_heads])
            rpr_v = tf.compat.v1.get_variable(
                'rpr_v', [2 * max_relative_dis + 1, value_depth // num_heads])
            rpr_k = create_rpr(rpr_k, length_q, length_kv, max_relative_dis)
            rpr_v = create_rpr(rpr_v, length_q, length_kv, max_relative_dis)
            rpr = {'rpr_k': rpr_k, 'rpr_v': rpr_v}
            x, w = dot_product_attention(q, k, v, bias, dropout_rate, rpr=rpr)
        else:
            x, w = dot_product_attention(q, k, v, bias, dropout_rate)
        x = combine_heads(x)
        w = tf.reduce_mean(w, 1)
        x = linear(x, output_depth, True, True, scope='output_transform')
        return x, w


def get_initializer(params):
    if params['initializer'] == 'uniform':
        max_val = params['initializer_scale']
        return tf.compat.v1.random_uniform_initializer(-max_val, max_val)
    elif params['initializer'] == 'normal':
        return tf.compat.v1.random_normal_initializer(
            0.0, params['initializer_scale'])
    elif params['initializer'] == 'normal_unit_scaling':
        return tf.compat.v1.variance_scaling_initializer(
            params['initializer_scale'], mode='fan_avg', distribution='normal')
    elif params['initializer'] == 'uniform_unit_scaling':
        return tf.compat.v1.variance_scaling_initializer(
            params['initializer_scale'],
            mode='fan_avg',
            distribution='uniform')
    else:
        raise ValueError('Unrecognized initializer: %s'
                         % params['initializer'])


def get_learning_rate_decay(learning_rate, global_step, params):
    if params['learning_rate_decay'] in ['linear_warmup_rsqrt_decay', 'noam']:
        step = tf.cast(global_step, dtype=tf.float32)
        warmup_steps = tf.cast(params['warmup_steps'], dtype=tf.float32)
        multiplier = params['hidden_size']**-0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps**-1.5),
                                        (step + 1)**-0.5)
        return learning_rate * decay
    elif params['learning_rate_decay'] == 'piecewise_constant':
        return tf.compat.v1.train.piecewise_constant(
            tf.cast(global_step, dtype=tf.int32),
            params['learning_rate_boundaries'], params['learning_rate_values'])
    elif params['learning_rate_decay'] == 'none':
        return learning_rate
    else:
        raise ValueError('Unknown learning_rate_decay')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


_ENGINE = None


def all_reduce(tensor):
    if _ENGINE is None:
        return tensor

    return _ENGINE.allreduce(tensor, compression=_ENGINE.Compression.fp16)


class MultiStepOptimizer(tf.compat.v1.train.Optimizer):

    def __init__(self,
                 optimizer,
                 step=1,
                 use_locking=False,
                 name='MultiStepOptimizer'):
        super(MultiStepOptimizer, self).__init__(use_locking, name)
        self._optimizer = optimizer
        self._step = step
        self._step_t = tf.convert_to_tensor(step, name='step')

    def _all_reduce(self, tensor):
        with tf.name_scope(self._name + '_Allreduce'):
            if tensor is None:
                return tensor

            if isinstance(tensor, tf.IndexedSlices):
                tensor = tf.convert_to_tensor(tensor)

            return all_reduce(tensor)

    def compute_gradients(self,
                          loss,
                          var_list=None,
                          gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        grads_and_vars = self._optimizer.compute_gradients(
            loss, var_list, gate_gradients, aggregation_method,
            colocate_gradients_with_ops, grad_loss)

        grads, var_list = list(zip(*grads_and_vars))

        # Do not create extra variables when step is 1
        if self._step == 1:
            grads = [self._all_reduce(t) for t in grads]
            return list(zip(grads, var_list))

        first_var = min(var_list, key=lambda x: x.name)
        iter_var = self._create_non_slot_variable(
            initial_value=0 if self._step == 1 else 1,
            name='iter',
            colocate_with=first_var)

        new_grads = []

        for grad, var in zip(grads, var_list):
            grad_acc = self._zeros_slot(var, 'grad_acc', self._name)

            if isinstance(grad, tf.IndexedSlices):
                grad_acc = tf.scatter_add(
                    grad_acc,
                    grad.indices,
                    grad.values,
                    use_locking=self._use_locking)
            else:
                grad_acc = tf.assign_add(
                    grad_acc, grad, use_locking=self._use_locking)

            def _acc_grad():
                return grad_acc

            def _avg_grad():
                return self._all_reduce(grad_acc / self._step)

            grad = tf.cond(tf.equal(iter_var, 0), _avg_grad, _acc_grad)
            new_grads.append(grad)

        return list(zip(new_grads, var_list))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if self._step == 1:
            return self._optimizer.apply_gradients(
                grads_and_vars, global_step, name=name)

        grads, var_list = list(zip(*grads_and_vars))

        def _pass_gradients():
            return tf.group(*grads)

        def _apply_gradients():
            op = self._optimizer.apply_gradients(
                zip(grads, var_list), global_step, name)
            with tf.control_dependencies([op]):
                zero_ops = []
                for var in var_list:
                    grad_acc = self.get_slot(var, 'grad_acc')
                    zero_ops.append(
                        grad_acc.assign(
                            tf.zeros_like(grad_acc),
                            use_locking=self._use_locking))
                zero_op = tf.group(*zero_ops)
            return tf.group(*[op, zero_op])

        iter_var = self._get_non_slot_variable('iter', tf.get_default_graph())
        update_op = tf.cond(
            tf.equal(iter_var, 0), _apply_gradients, _pass_gradients)

        with tf.control_dependencies([update_op]):
            iter_op = iter_var.assign(
                tf.mod(iter_var + 1, self._step_t),
                use_locking=self._use_locking)

        return tf.group(*[update_op, iter_op])


def shard_features(x, num_datashards):
    x = tf.convert_to_tensor(x)
    batch_size = tf.shape(x)[0]
    size_splits = []

    with tf.device('/cpu:0'):
        for i in range(num_datashards):
            size_splits.append(
                tf.cond(
                    tf.greater(
                        tf.compat.v1.mod(batch_size, num_datashards),
                        i), lambda: batch_size // num_datashards + 1,
                    lambda: batch_size // num_datashards))

    return tf.split(x, size_splits, axis=0)
