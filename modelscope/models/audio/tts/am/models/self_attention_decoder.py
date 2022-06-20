"""Define self-attention decoder."""

import sys

import tensorflow as tf

from . import compat, transformer
from .modules import decoder_prenet
from .position import SinusoidalPositionEncoder


class SelfAttentionDecoder():
    """Decoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 prenet_units=256,
                 dense_units=128,
                 num_mels=80,
                 outputs_per_step=3,
                 X_band_width=None,
                 H_band_width=None,
                 position_encoder=SinusoidalPositionEncoder(),
                 self_attention_type='scaled_dot'):
        """Initializes the parameters of the decoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
          self_attention_type: Type of self attention, "scaled_dot" or "average" (case
            insensitive).

        Raises:
          ValueError: if :obj:`self_attention_type` is invalid.
        """
        super(SelfAttentionDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder
        self.self_attention_type = self_attention_type.lower()
        if self.self_attention_type not in ('scaled_dot', 'average'):
            raise ValueError('invalid attention type %s'
                             % self.self_attention_type)
        if self.self_attention_type == 'average':
            tf.logging.warning(
                'Support for average attention network is experimental '
                'and may change in future versions.')
        self.prenet_units = prenet_units
        self.dense_units = dense_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.X_band_width = X_band_width
        self.H_band_width = H_band_width

    @property
    def output_size(self):
        """Returns the decoder output size."""
        return self.num_units

    @property
    def support_alignment_history(self):
        return True

    @property
    def support_multi_source(self):
        return True

    def _init_cache(self, batch_size, dtype=tf.float32, num_sources=1):
        cache = {}

        for layer in range(self.num_layers):
            proj_cache_shape = [
                batch_size, self.num_heads, 0, self.num_units // self.num_heads
            ]
            layer_cache = {}
            layer_cache['memory'] = [{
                'memory_keys':
                tf.zeros(proj_cache_shape, dtype=dtype),
                'memory_values':
                tf.zeros(proj_cache_shape, dtype=dtype)
            } for _ in range(num_sources)]
            if self.self_attention_type == 'scaled_dot':
                layer_cache['self_keys'] = tf.zeros(
                    proj_cache_shape, dtype=dtype)
                layer_cache['self_values'] = tf.zeros(
                    proj_cache_shape, dtype=dtype)
            elif self.self_attention_type == 'average':
                layer_cache['prev_g'] = tf.zeros(
                    [batch_size, 1, self.num_units], dtype=dtype)
            cache['layer_{}'.format(layer)] = layer_cache

        return cache

    def _init_attn(self, dtype=tf.float32):
        attn = []
        for layer in range(self.num_layers):
            attn.append(tf.TensorArray(tf.float32, size=0, dynamic_size=True))
        return attn

    def _self_attention_stack(self,
                              inputs,
                              sequence_length=None,
                              mode=True,
                              cache=None,
                              memory=None,
                              memory_sequence_length=None,
                              step=None):

        # [N, T_out, self.dense_units] or [N, 1, self.dense_units]
        prenet_outputs = decoder_prenet(inputs, self.prenet_units,
                                        self.dense_units, mode)
        if step is None:
            decoder_inputs = tf.concat(
                [memory, prenet_outputs],
                axis=-1)  # [N, T_out, memory_size + self.dense_units]
        else:
            decoder_inputs = tf.concat(
                [memory[:, step:step + 1, :], prenet_outputs],
                axis=-1)  # [N, 1, memory_size + self.dense_units]
        decoder_inputs = tf.layers.dense(
            decoder_inputs, units=self.dense_units)

        inputs = decoder_inputs
        inputs *= self.num_units**0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(
                inputs, position=step + 1 if step is not None else None)

        inputs = tf.layers.dropout(inputs, rate=self.dropout, training=mode)

        decoder_mask = None
        memory_mask = None
        # last_attention = None

        X_band_width_tmp = -1
        H_band_width_tmp = -1
        if self.X_band_width is not None:
            X_band_width_tmp = tf.cast(
                tf.cond(
                    tf.less(tf.shape(memory)[1], self.X_band_width),
                    lambda: -1, lambda: self.X_band_width),
                dtype=tf.int64)
        if self.H_band_width is not None:
            H_band_width_tmp = tf.cast(
                tf.cond(
                    tf.less(tf.shape(memory)[1], self.H_band_width),
                    lambda: -1, lambda: self.H_band_width),
                dtype=tf.int64)

        if self.self_attention_type == 'scaled_dot':
            if sequence_length is not None:
                decoder_mask = transformer.build_future_mask(
                    sequence_length,
                    num_heads=self.num_heads,
                    maximum_length=tf.shape(inputs)[1],
                    band=X_band_width_tmp)  # [N, 1, T_out, T_out]
        elif self.self_attention_type == 'average':
            if cache is None:
                if sequence_length is None:
                    sequence_length = tf.fill([tf.shape(inputs)[0]],
                                              tf.shape(inputs)[1])
                decoder_mask = transformer.cumulative_average_mask(
                    sequence_length,
                    maximum_length=tf.shape(inputs)[1],
                    dtype=inputs.dtype)

        if memory is not None and not tf.contrib.framework.nest.is_sequence(
                memory):
            memory = (memory, )
        if memory_sequence_length is not None:
            if not tf.contrib.framework.nest.is_sequence(
                    memory_sequence_length):
                memory_sequence_length = (memory_sequence_length, )
            if step is None:
                memory_mask = [
                    transformer.build_history_mask(
                        length,
                        num_heads=self.num_heads,
                        maximum_length=tf.shape(m)[1],
                        band=H_band_width_tmp)
                    for m, length in zip(memory, memory_sequence_length)
                ]
            else:
                memory_mask = [
                    transformer.build_history_mask(
                        length,
                        num_heads=self.num_heads,
                        maximum_length=tf.shape(m)[1],
                        band=H_band_width_tmp)[:, :, step:step + 1, :]
                    for m, length in zip(memory, memory_sequence_length)
                ]

        # last_attention = None
        attns_x = []
        attns_h = []
        for layer in range(self.num_layers):
            layer_name = 'layer_{}'.format(layer)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                if memory is not None:
                    for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
                        memory_cache = None
                        if layer_cache is not None:
                            memory_cache = layer_cache['memory'][i]
                        scope_name = 'multi_head_{}'.format(i)
                        if i == 0:
                            scope_name = 'multi_head'
                        with tf.variable_scope(scope_name):
                            encoded, attn_x, attn_h = transformer.multi_head_attention_PNCA(
                                self.num_heads,
                                transformer.norm(inputs),
                                mem,
                                mode,
                                num_units=self.num_units,
                                mask=decoder_mask,
                                mask_h=mask,
                                cache=layer_cache,
                                cache_h=memory_cache,
                                dropout=self.attention_dropout,
                                return_attention=True,
                                layer_name=layer_name,
                                X_band_width=self.X_band_width)
                            attns_x.append(attn_x)
                            attns_h.append(attn_h)
                            context = transformer.drop_and_add(
                                inputs, encoded, mode, dropout=self.dropout)

                with tf.variable_scope('ffn'):
                    transformed = transformer.feed_forward_ori(
                        transformer.norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(
                        context, transformed, mode, dropout=self.dropout)

                inputs = transformed

        outputs = transformer.norm(inputs)
        outputs = tf.layers.dense(
            outputs, units=self.num_mels * self.outputs_per_step)
        return outputs, attns_x, attns_h

    def decode_from_inputs(self,
                           inputs,
                           sequence_length,
                           initial_state=None,
                           mode=True,
                           memory=None,
                           memory_sequence_length=None):
        outputs, attention_x, attention_h = self._self_attention_stack(
            inputs,
            sequence_length=sequence_length,
            mode=mode,
            memory=memory,
            memory_sequence_length=memory_sequence_length)
        return outputs, attention_x, attention_h

    def step_fn(self,
                mode,
                batch_size,
                initial_state=None,
                memory=None,
                memory_sequence_length=None,
                dtype=tf.float32):
        if memory is None:
            num_sources = 0
        elif tf.contrib.framework.nest.is_sequence(memory):
            num_sources = len(memory)
        else:
            num_sources = 1
        cache = self._init_cache(
            batch_size, dtype=dtype, num_sources=num_sources)
        attention_x = self._init_attn(dtype=dtype)
        attention_h = self._init_attn(dtype=dtype)

        def _fn(step, inputs, cache):
            outputs, attention_x, attention_h = self._self_attention_stack(
                inputs,
                mode=mode,
                cache=cache,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                step=step)
            attention_x_tmp = []
            for layer in range(len(attention_h)):
                attention_x_tmp_l = tf.zeros_like(attention_h[layer])
                if self.X_band_width is not None:
                    pred = tf.less(step, self.X_band_width + 1)
                    attention_x_tmp_l_1 = tf.cond(pred,  # yapf:disable
                                                  lambda: attention_x_tmp_l[:, :, :, :step + 1] + attention_x[layer],
                                                  lambda: tf.concat([
                                                                    attention_x_tmp_l[:, :, :,
                                                                                      :step - self.X_band_width],
                                                                    attention_x_tmp_l[:, :, :,
                                                                                      step - self.X_band_width:step + 1]
                                                                    + attention_x[layer]],
                                                                    axis=-1))  # yapf:disable
                    attention_x_tmp_l_2 = attention_x_tmp_l[:, :, :, step + 1:]
                    attention_x_tmp.append(
                        tf.concat([attention_x_tmp_l_1, attention_x_tmp_l_2],
                                  axis=-1))
                else:
                    attention_x_tmp_l_1 = attention_x_tmp_l[:, :, :, :step + 1]
                    attention_x_tmp_l_2 = attention_x_tmp_l[:, :, :, step + 1:]
                    attention_x_tmp.append(
                        tf.concat([
                            attention_x_tmp_l_1 + attention_x[layer],
                            attention_x_tmp_l_2
                        ], axis=-1))  # yapf:disable
            attention_x = attention_x_tmp
            return outputs, cache, attention_x, attention_h

        return _fn, cache, attention_x, attention_h

    def dynamic_decode_and_search(self, init_decoder_input, maximum_iterations,
                                  mode, memory, memory_sequence_length):
        batch_size = tf.shape(init_decoder_input)[0]
        step_fn, init_cache, init_attn_x, init_attn_h = self.step_fn(
            mode,
            batch_size,
            memory=memory,
            memory_sequence_length=memory_sequence_length)

        outputs, attention_x, attention_h, cache = self.dynamic_decode(
            step_fn,
            init_decoder_input,
            init_cache=init_cache,
            init_attn_x=init_attn_x,
            init_attn_h=init_attn_h,
            maximum_iterations=maximum_iterations,
            batch_size=batch_size)
        return outputs, attention_x, attention_h

    def dynamic_decode_and_search_teacher_forcing(self, decoder_input,
                                                  maximum_iterations, mode,
                                                  memory,
                                                  memory_sequence_length):
        batch_size = tf.shape(decoder_input)[0]
        step_fn, init_cache, init_attn_x, init_attn_h = self.step_fn(
            mode,
            batch_size,
            memory=memory,
            memory_sequence_length=memory_sequence_length)

        outputs, attention_x, attention_h, cache = self.dynamic_decode_teacher_forcing(
            step_fn,
            decoder_input,
            init_cache=init_cache,
            init_attn_x=init_attn_x,
            init_attn_h=init_attn_h,
            maximum_iterations=maximum_iterations,
            batch_size=batch_size)
        return outputs, attention_x, attention_h

    def dynamic_decode(self,
                       step_fn,
                       init_decoder_input,
                       init_cache=None,
                       init_attn_x=None,
                       init_attn_h=None,
                       maximum_iterations=None,
                       batch_size=None):

        def _cond(step, cache, inputs, outputs, attention_x, attention_h):  # pylint: disable=unused-argument
            return tf.less(step, maximum_iterations)

        def _body(step, cache, inputs, outputs, attention_x, attention_h):
            # output: [1, 1, num_mels * r]
            # attn: [1, 1, T_out]
            output, cache, attn_x, attn_h = step_fn(
                step, inputs, cache)  # outputs, cache, attention, attns
            for layer in range(len(attention_x)):
                attention_x[layer] = attention_x[layer].write(
                    step, tf.cast(attn_x[layer], tf.float32))

            for layer in range(len(attention_h)):
                attention_h[layer] = attention_h[layer].write(
                    step, tf.cast(attn_h[layer], tf.float32))

            outputs = outputs.write(step, tf.cast(output, tf.float32))
            return step + 1, cache, output[:, :, -self.
                                           num_mels:], outputs, attention_x, attention_h

        step = tf.constant(0, dtype=tf.int32)
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        _, cache, _, outputs, attention_x, attention_h = tf.while_loop(
            _cond,
            _body,
            loop_vars=(step, init_cache, init_decoder_input, outputs,
                       init_attn_x, init_attn_h),
            shape_invariants=(step.shape,
                              compat.nest.map_structure(
                                  self._get_shape_invariants, init_cache),
                              compat.nest.map_structure(
                                  self._get_shape_invariants,
                                  init_decoder_input), tf.TensorShape(None),
                              compat.nest.map_structure(
                                  self._get_shape_invariants, init_attn_x),
                              compat.nest.map_structure(
                                  self._get_shape_invariants, init_attn_h)),
            parallel_iterations=1,
            back_prop=False,
            maximum_iterations=maximum_iterations)
        # element of outputs: [N, 1, num_mels * r]
        outputs_stack = outputs.stack()  # [T_out, N, 1, num_mels * r]
        outputs_stack = tf.transpose(
            outputs_stack, perm=[2, 1, 0, 3])  # [1, N, T_out, num_mels * r]
        outputs_stack = tf.squeeze(
            outputs_stack, axis=0)  # [N, T_out, num_mels * r]

        attention_x_stack = []
        for layer in range(len(attention_x)):
            attention_x_stack_tmp = attention_x[layer].stack(
            )  # [T_out, N, H, 1, T_out]
            attention_x_stack_tmp = tf.transpose(
                attention_x_stack_tmp, perm=[3, 1, 2, 0,
                                             4])  # [1, N, H, T_out, T_out]
            attention_x_stack_tmp = tf.squeeze(
                attention_x_stack_tmp, axis=0)  # [N, H, T_out, T_out]
            attention_x_stack.append(attention_x_stack_tmp)

        attention_h_stack = []
        for layer in range(len(attention_h)):
            attention_h_stack_tmp = attention_h[layer].stack(
            )  # [T_out, N, H, 1, T_out]
            attention_h_stack_tmp = tf.transpose(
                attention_h_stack_tmp, perm=[3, 1, 2, 0,
                                             4])  # [1, N, H, T_out, T_out]
            attention_h_stack_tmp = tf.squeeze(
                attention_h_stack_tmp, axis=0)  # [N, H, T_out, T_out]
            attention_h_stack.append(attention_h_stack_tmp)

        return outputs_stack, attention_x_stack, attention_h_stack, cache

    def dynamic_decode_teacher_forcing(self,
                                       step_fn,
                                       decoder_input,
                                       init_cache=None,
                                       init_attn_x=None,
                                       init_attn_h=None,
                                       maximum_iterations=None,
                                       batch_size=None):

        def _cond(step, cache, inputs, outputs, attention_x, attention_h):  # pylint: disable=unused-argument
            return tf.less(step, maximum_iterations)

        def _body(step, cache, inputs, outputs, attention_x, attention_h):
            # output: [1, 1, num_mels * r]
            # attn: [1, 1, T_out]
            output, cache, attn_x, attn_h = step_fn(
                step, inputs[:, step:step + 1, :],
                cache)  # outputs, cache, attention, attns
            for layer in range(len(attention_x)):
                attention_x[layer] = attention_x[layer].write(
                    step, tf.cast(attn_x[layer], tf.float32))

            for layer in range(len(attention_h)):
                attention_h[layer] = attention_h[layer].write(
                    step, tf.cast(attn_h[layer], tf.float32))
            outputs = outputs.write(step, tf.cast(output, tf.float32))
            return step + 1, cache, inputs, outputs, attention_x, attention_h

        step = tf.constant(0, dtype=tf.int32)
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        _, cache, _, outputs, attention_x, attention_h = tf.while_loop(
            _cond,
            _body,
            loop_vars=(step, init_cache, decoder_input, outputs, init_attn_x,
                       init_attn_h),
            shape_invariants=(step.shape,
                              compat.nest.map_structure(
                                  self._get_shape_invariants,
                                  init_cache), decoder_input.shape,
                              tf.TensorShape(None),
                              compat.nest.map_structure(
                                  self._get_shape_invariants, init_attn_x),
                              compat.nest.map_structure(
                                  self._get_shape_invariants, init_attn_h)),
            parallel_iterations=1,
            back_prop=False,
            maximum_iterations=maximum_iterations)
        # element of outputs: [N, 1, num_mels * r]
        outputs_stack = outputs.stack()  # [T_out, N, 1, num_mels * r]
        outputs_stack = tf.transpose(
            outputs_stack, perm=[2, 1, 0, 3])  # [1, N, T_out, num_mels * r]
        outputs_stack = tf.squeeze(
            outputs_stack, axis=0)  # [N, T_out, num_mels * r]

        attention_x_stack = []
        for layer in range(len(attention_x)):
            attention_x_stack_tmp = attention_x[layer].stack(
            )  # [T_out, N, H, 1, T_out]
            attention_x_stack_tmp = tf.transpose(
                attention_x_stack_tmp, perm=[3, 1, 2, 0,
                                             4])  # [1, N, H, T_out, T_out]
            attention_x_stack_tmp = tf.squeeze(
                attention_x_stack_tmp, axis=0)  # [N, H, T_out, T_out]
            attention_x_stack.append(attention_x_stack_tmp)

        attention_h_stack = []
        for layer in range(len(attention_h)):
            attention_h_stack_tmp = attention_h[layer].stack(
            )  # [T_out, N, H, 1, T_out]
            attention_h_stack_tmp = tf.transpose(
                attention_h_stack_tmp, perm=[3, 1, 2, 0,
                                             4])  # [1, N, H, T_out, T_out]
            attention_h_stack_tmp = tf.squeeze(
                attention_h_stack_tmp, axis=0)  # [N, H, T_out, T_out]
            attention_h_stack.append(attention_h_stack_tmp)

        return outputs_stack, attention_x_stack, attention_h_stack, cache

    def _get_shape_invariants(self, tensor):
        """Returns the shape of the tensor but sets middle dims to None."""
        if isinstance(tensor, tf.TensorArray):
            shape = None
        else:
            shape = tensor.shape.as_list()
            for i in range(1, len(shape) - 1):
                shape[i] = None
        return tf.TensorShape(shape)


class SelfAttentionDecoderOri():
    """Decoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder(),
                 self_attention_type='scaled_dot'):
        """Initializes the parameters of the decoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
          self_attention_type: Type of self attention, "scaled_dot" or "average" (case
            insensitive).

        Raises:
          ValueError: if :obj:`self_attention_type` is invalid.
        """
        super(SelfAttentionDecoderOri, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder
        self.self_attention_type = self_attention_type.lower()
        if self.self_attention_type not in ('scaled_dot', 'average'):
            raise ValueError('invalid attention type %s'
                             % self.self_attention_type)
        if self.self_attention_type == 'average':
            tf.logging.warning(
                'Support for average attention network is experimental '
                'and may change in future versions.')

    @property
    def output_size(self):
        """Returns the decoder output size."""
        return self.num_units

    @property
    def support_alignment_history(self):
        return True

    @property
    def support_multi_source(self):
        return True

    def _init_cache(self, batch_size, dtype=tf.float32, num_sources=1):
        cache = {}

        for layer in range(self.num_layers):
            proj_cache_shape = [
                batch_size, self.num_heads, 0, self.num_units // self.num_heads
            ]
            layer_cache = {}
            layer_cache['memory'] = [{
                'memory_keys':
                tf.zeros(proj_cache_shape, dtype=dtype),
                'memory_values':
                tf.zeros(proj_cache_shape, dtype=dtype)
            } for _ in range(num_sources)]
            if self.self_attention_type == 'scaled_dot':
                layer_cache['self_keys'] = tf.zeros(
                    proj_cache_shape, dtype=dtype)
                layer_cache['self_values'] = tf.zeros(
                    proj_cache_shape, dtype=dtype)
            elif self.self_attention_type == 'average':
                layer_cache['prev_g'] = tf.zeros(
                    [batch_size, 1, self.num_units], dtype=dtype)
            cache['layer_{}'.format(layer)] = layer_cache

        return cache

    def _self_attention_stack(self,
                              inputs,
                              sequence_length=None,
                              mode=True,
                              cache=None,
                              memory=None,
                              memory_sequence_length=None,
                              step=None):
        inputs *= self.num_units**0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(
                inputs, position=step + 1 if step is not None else None)

        inputs = tf.layers.dropout(inputs, rate=self.dropout, training=mode)

        decoder_mask = None
        memory_mask = None
        last_attention = None

        if self.self_attention_type == 'scaled_dot':
            if sequence_length is not None:
                decoder_mask = transformer.build_future_mask(
                    sequence_length,
                    num_heads=self.num_heads,
                    maximum_length=tf.shape(inputs)[1])
        elif self.self_attention_type == 'average':
            if cache is None:
                if sequence_length is None:
                    sequence_length = tf.fill([tf.shape(inputs)[0]],
                                              tf.shape(inputs)[1])
                decoder_mask = transformer.cumulative_average_mask(
                    sequence_length,
                    maximum_length=tf.shape(inputs)[1],
                    dtype=inputs.dtype)

        if memory is not None and not tf.contrib.framework.nest.is_sequence(
                memory):
            memory = (memory, )
        if memory_sequence_length is not None:
            if not tf.contrib.framework.nest.is_sequence(
                    memory_sequence_length):
                memory_sequence_length = (memory_sequence_length, )
            memory_mask = [
                transformer.build_sequence_mask(
                    length,
                    num_heads=self.num_heads,
                    maximum_length=tf.shape(m)[1])
                for m, length in zip(memory, memory_sequence_length)
            ]

        for layer in range(self.num_layers):
            layer_name = 'layer_{}'.format(layer)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                if self.self_attention_type == 'scaled_dot':
                    with tf.variable_scope('masked_multi_head'):
                        encoded = transformer.multi_head_attention(
                            self.num_heads,
                            transformer.norm(inputs),
                            None,
                            mode,
                            num_units=self.num_units,
                            mask=decoder_mask,
                            cache=layer_cache,
                            dropout=self.attention_dropout)
                        last_context = transformer.drop_and_add(
                            inputs, encoded, mode, dropout=self.dropout)
                elif self.self_attention_type == 'average':
                    with tf.variable_scope('average_attention'):
                        # Cumulative average.
                        x = transformer.norm(inputs)
                        y = transformer.cumulative_average(
                            x,
                            decoder_mask if cache is None else step,
                            cache=layer_cache)
                        # FFN.
                        y = transformer.feed_forward(
                            y,
                            self.ffn_inner_dim,
                            mode,
                            dropout=self.relu_dropout)
                        # Gating layer.
                        z = tf.layers.dense(
                            tf.concat([x, y], -1), self.num_units * 2)
                        i, f = tf.split(z, 2, axis=-1)
                        y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
                        last_context = transformer.drop_and_add(
                            inputs, y, mode, dropout=self.dropout)

                if memory is not None:
                    for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
                        memory_cache = layer_cache['memory'][i] if layer_cache is not None else None  # yapf:disable
                        with tf.variable_scope('multi_head' if i
                                               == 0 else 'multi_head_%d' % i):  # yapf:disable
                            context, last_attention = transformer.multi_head_attention(
                                self.num_heads,
                                transformer.norm(last_context),
                                mem,
                                mode,
                                mask=mask,
                                cache=memory_cache,
                                dropout=self.attention_dropout,
                                return_attention=True)
                            last_context = transformer.drop_and_add(
                                last_context,
                                context,
                                mode,
                                dropout=self.dropout)
                            if i > 0:  # Do not return attention in case of multi source.
                                last_attention = None

                with tf.variable_scope('ffn'):
                    transformed = transformer.feed_forward_ori(
                        transformer.norm(last_context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(
                        last_context, transformed, mode, dropout=self.dropout)

                inputs = transformed

        if last_attention is not None:
            # The first head of the last layer is returned.
            first_head_attention = last_attention[:, 0]
        else:
            first_head_attention = None

        outputs = transformer.norm(inputs)
        return outputs, first_head_attention

    def decode_from_inputs(self,
                           inputs,
                           sequence_length,
                           initial_state=None,
                           mode=True,
                           memory=None,
                           memory_sequence_length=None):
        outputs, attention = self._self_attention_stack(
            inputs,
            sequence_length=sequence_length,
            mode=mode,
            memory=memory,
            memory_sequence_length=memory_sequence_length)
        return outputs, None, attention

    def step_fn(self,
                mode,
                batch_size,
                initial_state=None,
                memory=None,
                memory_sequence_length=None,
                dtype=tf.float32):
        if memory is None:
            num_sources = 0
        elif tf.contrib.framework.nest.is_sequence(memory):
            num_sources = len(memory)
        else:
            num_sources = 1
        cache = self._init_cache(
            batch_size, dtype=dtype, num_sources=num_sources)

        def _fn(step, inputs, cache, mode):
            inputs = tf.expand_dims(inputs, 1)
            outputs, attention = self._self_attention_stack(
                inputs,
                mode=mode,
                cache=cache,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                step=step)
            outputs = tf.squeeze(outputs, axis=1)
            if attention is not None:
                attention = tf.squeeze(attention, axis=1)
            return outputs, cache, attention

        return _fn, cache
