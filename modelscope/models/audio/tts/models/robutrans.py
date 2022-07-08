import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell, MultiRNNCell
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.python.ops.ragged.ragged_util import repeat

from .am_models import conv_prenet, decoder_prenet, encoder_prenet
from .fsmn_encoder import FsmnEncoderV2
from .helpers import VarTestHelper, VarTrainingHelper
from .position import (BatchSinusodalPositionalEncoding,
                       SinusodalPositionalEncoding)
from .rnn_wrappers import DurPredictorCell, VarPredictorCell
from .self_attention_decoder import SelfAttentionDecoder
from .self_attention_encoder import SelfAttentionEncoder


class RobuTrans():

    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self,
                   inputs,
                   inputs_emotion,
                   inputs_speaker,
                   input_lengths,
                   output_lengths=None,
                   mel_targets=None,
                   durations=None,
                   pitch_contours=None,
                   uv_masks=None,
                   pitch_scales=None,
                   duration_scales=None,
                   energy_contours=None,
                   energy_scales=None):
        '''Initializes the model for inference.

        Sets "mel_outputs", "linear_outputs", "stop_token_outputs", and "alignments" fields.

        Args:
          inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
            steps in the input time series, and values are character IDs
          input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
          output_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in outputs.
          mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
        '''
        with tf.variable_scope('inference') as _:
            is_training = mel_targets is not None
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            input_mask = None
            if input_lengths is not None and is_training:
                input_mask = tf.sequence_mask(
                    input_lengths, tf.shape(inputs)[1], dtype=tf.float32)

            if input_mask is not None:
                inputs = inputs * tf.expand_dims(input_mask, -1)

            # speaker embedding
            embedded_inputs_speaker = tf.layers.dense(
                inputs_speaker,
                32,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.5))

            # emotion embedding
            embedded_inputs_emotion = tf.layers.dense(
                inputs_emotion,
                32,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.5))

            # symbol embedding
            with tf.variable_scope('Embedding'):
                embedded_inputs = tf.layers.dense(
                    inputs,
                    hp.embedding_dim,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=0.5))

            # Encoder
            with tf.variable_scope('Encoder'):
                Encoder = SelfAttentionEncoder(
                    num_layers=hp.encoder_num_layers,
                    num_units=hp.encoder_num_units,
                    num_heads=hp.encoder_num_heads,
                    ffn_inner_dim=hp.encoder_ffn_inner_dim,
                    dropout=hp.encoder_dropout,
                    attention_dropout=hp.encoder_attention_dropout,
                    relu_dropout=hp.encoder_relu_dropout)
                encoder_outputs, state_mo, sequence_length_mo, attns = Encoder.encode(
                    embedded_inputs,
                    sequence_length=input_lengths,
                    mode=is_training)
                encoder_outputs = tf.layers.dense(
                    encoder_outputs,
                    hp.encoder_projection_units,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=0.5))

            # pitch and energy
            var_inputs = tf.concat([
                encoder_outputs, embedded_inputs_speaker,
                embedded_inputs_emotion
            ], 2)
            if input_mask is not None:
                var_inputs = var_inputs * tf.expand_dims(input_mask, -1)

            with tf.variable_scope('Pitch_Predictor'):
                Pitch_Predictor_FSMN = FsmnEncoderV2(
                    filter_size=hp.predictor_filter_size,
                    fsmn_num_layers=hp.predictor_fsmn_num_layers,
                    dnn_num_layers=hp.predictor_dnn_num_layers,
                    num_memory_units=hp.predictor_num_memory_units,
                    ffn_inner_dim=hp.predictor_ffn_inner_dim,
                    dropout=hp.predictor_dropout,
                    shift=hp.predictor_shift,
                    position_encoder=None)
                pitch_contour_outputs, _, _ = Pitch_Predictor_FSMN.encode(
                    tf.concat([
                        encoder_outputs, embedded_inputs_speaker,
                        embedded_inputs_emotion
                    ], 2),
                    sequence_length=input_lengths,
                    mode=is_training)
                pitch_contour_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    LSTMBlockCell(hp.predictor_lstm_units),
                    LSTMBlockCell(hp.predictor_lstm_units),
                    pitch_contour_outputs,
                    sequence_length=input_lengths,
                    dtype=tf.float32)
                pitch_contour_outputs = tf.concat(
                    pitch_contour_outputs, axis=-1)
                pitch_contour_outputs = tf.layers.dense(
                    pitch_contour_outputs, units=1)  # [N, T_in, 1]
                pitch_contour_outputs = tf.squeeze(
                    pitch_contour_outputs, axis=2)  # [N, T_in]

            with tf.variable_scope('Energy_Predictor'):
                Energy_Predictor_FSMN = FsmnEncoderV2(
                    filter_size=hp.predictor_filter_size,
                    fsmn_num_layers=hp.predictor_fsmn_num_layers,
                    dnn_num_layers=hp.predictor_dnn_num_layers,
                    num_memory_units=hp.predictor_num_memory_units,
                    ffn_inner_dim=hp.predictor_ffn_inner_dim,
                    dropout=hp.predictor_dropout,
                    shift=hp.predictor_shift,
                    position_encoder=None)
                energy_contour_outputs, _, _ = Energy_Predictor_FSMN.encode(
                    tf.concat([
                        encoder_outputs, embedded_inputs_speaker,
                        embedded_inputs_emotion
                    ], 2),
                    sequence_length=input_lengths,
                    mode=is_training)
                energy_contour_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    LSTMBlockCell(hp.predictor_lstm_units),
                    LSTMBlockCell(hp.predictor_lstm_units),
                    energy_contour_outputs,
                    sequence_length=input_lengths,
                    dtype=tf.float32)
                energy_contour_outputs = tf.concat(
                    energy_contour_outputs, axis=-1)
                energy_contour_outputs = tf.layers.dense(
                    energy_contour_outputs, units=1)  # [N, T_in, 1]
                energy_contour_outputs = tf.squeeze(
                    energy_contour_outputs, axis=2)  # [N, T_in]

            if is_training:
                pitch_embeddings = tf.expand_dims(
                    pitch_contours, axis=2)  # [N, T_in, 1]
                pitch_embeddings = tf.layers.conv1d(
                    pitch_embeddings,
                    filters=hp.encoder_projection_units,
                    kernel_size=9,
                    padding='same',
                    name='pitch_embeddings')  # [N, T_in, 32]

                energy_embeddings = tf.expand_dims(
                    energy_contours, axis=2)  # [N, T_in, 1]
                energy_embeddings = tf.layers.conv1d(
                    energy_embeddings,
                    filters=hp.encoder_projection_units,
                    kernel_size=9,
                    padding='same',
                    name='energy_embeddings')  # [N, T_in, 32]
            else:
                pitch_contour_outputs *= pitch_scales
                pitch_embeddings = tf.expand_dims(
                    pitch_contour_outputs, axis=2)  # [N, T_in, 1]
                pitch_embeddings = tf.layers.conv1d(
                    pitch_embeddings,
                    filters=hp.encoder_projection_units,
                    kernel_size=9,
                    padding='same',
                    name='pitch_embeddings')  # [N, T_in, 32]

                energy_contour_outputs *= energy_scales
                energy_embeddings = tf.expand_dims(
                    energy_contour_outputs, axis=2)  # [N, T_in, 1]
                energy_embeddings = tf.layers.conv1d(
                    energy_embeddings,
                    filters=hp.encoder_projection_units,
                    kernel_size=9,
                    padding='same',
                    name='energy_embeddings')  # [N, T_in, 32]

            encoder_outputs_ = encoder_outputs + pitch_embeddings + energy_embeddings

            # duration
            dur_inputs = tf.concat([
                encoder_outputs_, embedded_inputs_speaker,
                embedded_inputs_emotion
            ], 2)
            if input_mask is not None:
                dur_inputs = dur_inputs * tf.expand_dims(input_mask, -1)
            with tf.variable_scope('Duration_Predictor'):
                duration_predictor_cell = MultiRNNCell([
                    LSTMBlockCell(hp.predictor_lstm_units),
                    LSTMBlockCell(hp.predictor_lstm_units)
                ], state_is_tuple=True)  # yapf:disable
                duration_output_cell = DurPredictorCell(
                    duration_predictor_cell, is_training, 1,
                    hp.predictor_prenet_units)
                duration_predictor_init_state = duration_output_cell.zero_state(
                    batch_size=batch_size, dtype=tf.float32)
                if is_training:
                    duration_helper = VarTrainingHelper(
                        tf.expand_dims(
                            tf.log(tf.cast(durations, tf.float32) + 1),
                            axis=2), dur_inputs, 1)
                else:
                    duration_helper = VarTestHelper(batch_size, dur_inputs, 1)
                (
                    duration_outputs, _
                ), final_duration_predictor_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    BasicDecoder(duration_output_cell, duration_helper,
                                 duration_predictor_init_state),
                    maximum_iterations=1000)
                duration_outputs = tf.squeeze(
                    duration_outputs, axis=2)  # [N, T_in]
                if input_mask is not None:
                    duration_outputs = duration_outputs * input_mask
                duration_outputs_ = tf.exp(duration_outputs) - 1

            # Length Regulator
            with tf.variable_scope('Length_Regulator'):
                if is_training:
                    i = tf.constant(1)
                    # position embedding
                    j = tf.constant(1)
                    dur_len = tf.shape(durations)[-1]
                    embedded_position_i = tf.range(1, durations[0, 0] + 1)

                    def condition_pos(j, e):
                        return tf.less(j, dur_len)

                    def loop_body_pos(j, embedded_position_i):
                        embedded_position_i = tf.concat([
                            embedded_position_i,
                            tf.range(1, durations[0, j] + 1)
                        ], axis=0)  # yapf:disable
                        return [j + 1, embedded_position_i]

                    j, embedded_position_i = tf.while_loop(
                        condition_pos,
                        loop_body_pos, [j, embedded_position_i],
                        shape_invariants=[
                            j.get_shape(),
                            tf.TensorShape([None])
                        ])
                    embedded_position = tf.reshape(embedded_position_i,
                                                   (1, -1))

                    # others
                    LR_outputs = repeat(
                        encoder_outputs_[0:1, :, :], durations[0, :], axis=1)
                    embedded_outputs_speaker = repeat(
                        embedded_inputs_speaker[0:1, :, :],
                        durations[0, :],
                        axis=1)
                    embedded_outputs_emotion = repeat(
                        embedded_inputs_emotion[0:1, :, :],
                        durations[0, :],
                        axis=1)

                    def condition(i, pos, layer, s, e):
                        return tf.less(i, tf.shape(mel_targets)[0])

                    def loop_body(i, embedded_position, LR_outputs,
                                  embedded_outputs_speaker,
                                  embedded_outputs_emotion):
                        # position embedding
                        jj = tf.constant(1)
                        embedded_position_i = tf.range(1, durations[i, 0] + 1)

                        def condition_pos_i(j, e):
                            return tf.less(j, dur_len)

                        def loop_body_pos_i(j, embedded_position_i):
                            embedded_position_i = tf.concat([
                                embedded_position_i,
                                tf.range(1, durations[i, j] + 1)
                            ], axis=0)  # yapf:disable
                            return [j + 1, embedded_position_i]

                        jj, embedded_position_i = tf.while_loop(
                            condition_pos_i,
                            loop_body_pos_i, [jj, embedded_position_i],
                            shape_invariants=[
                                jj.get_shape(),
                                tf.TensorShape([None])
                            ])
                        embedded_position = tf.concat([
                            embedded_position,
                            tf.reshape(embedded_position_i, (1, -1))
                        ], 0)

                        # others
                        LR_outputs = tf.concat([
                            LR_outputs,
                            repeat(
                                encoder_outputs_[i:i + 1, :, :],
                                durations[i, :],
                                axis=1)
                        ], 0)
                        embedded_outputs_speaker = tf.concat([
                            embedded_outputs_speaker,
                            repeat(
                                embedded_inputs_speaker[i:i + 1, :, :],
                                durations[i, :],
                                axis=1)
                        ], 0)
                        embedded_outputs_emotion = tf.concat([
                            embedded_outputs_emotion,
                            repeat(
                                embedded_inputs_emotion[i:i + 1, :, :],
                                durations[i, :],
                                axis=1)
                        ], 0)
                        return [
                            i + 1, embedded_position, LR_outputs,
                            embedded_outputs_speaker, embedded_outputs_emotion
                        ]

                    i, embedded_position, LR_outputs,
                    embedded_outputs_speaker,
                    embedded_outputs_emotion = tf.while_loop(
                        condition,
                        loop_body, [
                            i, embedded_position, LR_outputs,
                            embedded_outputs_speaker, embedded_outputs_emotion
                        ],
                        shape_invariants=[
                            i.get_shape(),
                            tf.TensorShape([None, None]),
                            tf.TensorShape([None, None, None]),
                            tf.TensorShape([None, None, None]),
                            tf.TensorShape([None, None, None])
                        ],
                        parallel_iterations=hp.batch_size)

                    ori_framenum = tf.shape(mel_targets)[1]
                else:
                    # position
                    j = tf.constant(1)
                    dur_len = tf.shape(duration_outputs_)[-1]
                    embedded_position_i = tf.range(
                        1,
                        tf.cast(tf.round(duration_outputs_)[0, 0], tf.int32)
                        + 1)

                    def condition_pos(j, e):
                        return tf.less(j, dur_len)

                    def loop_body_pos(j, embedded_position_i):
                        embedded_position_i = tf.concat([
                            embedded_position_i,
                            tf.range(
                                1,
                                tf.cast(
                                    tf.round(duration_outputs_)[0, j],
                                    tf.int32) + 1)
                        ], axis=0)  # yapf:disable
                        return [j + 1, embedded_position_i]

                    j, embedded_position_i = tf.while_loop(
                        condition_pos,
                        loop_body_pos, [j, embedded_position_i],
                        shape_invariants=[
                            j.get_shape(),
                            tf.TensorShape([None])
                        ])
                    embedded_position = tf.reshape(embedded_position_i,
                                                   (1, -1))
                    # others
                    duration_outputs_ *= duration_scales
                    LR_outputs = repeat(
                        encoder_outputs_[0:1, :, :],
                        tf.cast(tf.round(duration_outputs_)[0, :], tf.int32),
                        axis=1)
                    embedded_outputs_speaker = repeat(
                        embedded_inputs_speaker[0:1, :, :],
                        tf.cast(tf.round(duration_outputs_)[0, :], tf.int32),
                        axis=1)
                    embedded_outputs_emotion = repeat(
                        embedded_inputs_emotion[0:1, :, :],
                        tf.cast(tf.round(duration_outputs_)[0, :], tf.int32),
                        axis=1)
                    ori_framenum = tf.shape(LR_outputs)[1]

                    left = hp.outputs_per_step - tf.mod(
                        ori_framenum, hp.outputs_per_step)
                    LR_outputs = tf.cond(
                        tf.equal(left,
                                 hp.outputs_per_step), lambda: LR_outputs,
                        lambda: tf.pad(LR_outputs, [[0, 0], [0, left], [0, 0]],
                                       'CONSTANT'))
                    embedded_outputs_speaker = tf.cond(
                        tf.equal(left, hp.outputs_per_step),
                        lambda: embedded_outputs_speaker, lambda: tf.pad(
                            embedded_outputs_speaker, [[0, 0], [0, left],
                                                       [0, 0]], 'CONSTANT'))
                    embedded_outputs_emotion = tf.cond(
                        tf.equal(left, hp.outputs_per_step),
                        lambda: embedded_outputs_emotion, lambda: tf.pad(
                            embedded_outputs_emotion, [[0, 0], [0, left],
                                                       [0, 0]], 'CONSTANT'))
                    embedded_position = tf.cond(
                        tf.equal(left, hp.outputs_per_step),
                        lambda: embedded_position,
                        lambda: tf.pad(embedded_position, [[0, 0], [0, left]],
                                       'CONSTANT'))

            # Pos_Embedding
            with tf.variable_scope('Position_Embedding'):
                Pos_Embedding = BatchSinusodalPositionalEncoding()
                position_embeddings = Pos_Embedding.positional_encoding(
                    batch_size,
                    tf.shape(LR_outputs)[1], hp.encoder_projection_units,
                    embedded_position)
            LR_outputs += position_embeddings

            # multi-frame
            LR_outputs = tf.reshape(LR_outputs, [
                batch_size, -1,
                hp.outputs_per_step * hp.encoder_projection_units
            ])
            embedded_outputs_speaker = tf.reshape(
                embedded_outputs_speaker,
                [batch_size, -1, hp.outputs_per_step * 32])[:, :, :32]
            embedded_outputs_emotion = tf.reshape(
                embedded_outputs_emotion,
                [batch_size, -1, hp.outputs_per_step * 32])[:, :, :32]
            # [N, T_out, D_LR_outputs] (D_LR_outputs = hp.outputs_per_step * hp.encoder_projection_units + 64)
            LR_outputs = tf.concat([
                LR_outputs, embedded_outputs_speaker, embedded_outputs_emotion
            ], -1)

            # auto bandwidth
            if is_training:
                durations_mask = tf.cast(durations,
                                         tf.float32) * input_mask  # [N, T_in]
            else:
                durations_mask = duration_outputs_
            X_band_width = tf.cast(
                tf.round(tf.reduce_max(durations_mask) / hp.outputs_per_step),
                tf.int32)
            H_band_width = X_band_width

            with tf.variable_scope('Decoder'):
                Decoder = SelfAttentionDecoder(
                    num_layers=hp.decoder_num_layers,
                    num_units=hp.decoder_num_units,
                    num_heads=hp.decoder_num_heads,
                    ffn_inner_dim=hp.decoder_ffn_inner_dim,
                    dropout=hp.decoder_dropout,
                    attention_dropout=hp.decoder_attention_dropout,
                    relu_dropout=hp.decoder_relu_dropout,
                    prenet_units=hp.prenet_units,
                    dense_units=hp.prenet_proj_units,
                    num_mels=hp.num_mels,
                    outputs_per_step=hp.outputs_per_step,
                    X_band_width=X_band_width,
                    H_band_width=H_band_width,
                    position_encoder=None)
                if is_training:
                    if hp.free_run:
                        r = hp.outputs_per_step
                        init_decoder_input = tf.expand_dims(
                            tf.tile([[0.0]], [batch_size, hp.num_mels]),
                            axis=1)  # [N, 1, hp.num_mels]
                        decoder_input_lengths = tf.cast(
                            output_lengths / r, tf.int32)
                        decoder_outputs, attention_x, attention_h = Decoder.dynamic_decode_and_search(
                            init_decoder_input,
                            maximum_iterations=tf.shape(LR_outputs)[1],
                            mode=is_training,
                            memory=LR_outputs,
                            memory_sequence_length=decoder_input_lengths)
                    else:
                        r = hp.outputs_per_step
                        decoder_input = mel_targets[:, r - 1::
                                                    r, :]  # [N, T_out / r, hp.num_mels]
                        init_decoder_input = tf.expand_dims(
                            tf.tile([[0.0]], [batch_size, hp.num_mels]),
                            axis=1)  # [N, 1, hp.num_mels]
                        decoder_input = tf.concat(
                            [init_decoder_input, decoder_input],
                            axis=1)  # [N, T_out / r + 1, hp.num_mels]
                        decoder_input = decoder_input[:, :
                                                      -1, :]  # [N, T_out / r, hp.num_mels]
                        decoder_input_lengths = tf.cast(
                            output_lengths / r, tf.int32)
                        decoder_outputs, attention_x, attention_h = Decoder.decode_from_inputs(
                            decoder_input,
                            decoder_input_lengths,
                            mode=is_training,
                            memory=LR_outputs,
                            memory_sequence_length=decoder_input_lengths)
                else:
                    init_decoder_input = tf.expand_dims(
                        tf.tile([[0.0]], [batch_size, hp.num_mels]),
                        axis=1)  # [N, 1, hp.num_mels]
                    decoder_outputs, attention_x, attention_h = Decoder.dynamic_decode_and_search(
                        init_decoder_input,
                        maximum_iterations=tf.shape(LR_outputs)[1],
                        mode=is_training,
                        memory=LR_outputs,
                        memory_sequence_length=tf.expand_dims(
                            tf.shape(LR_outputs)[1], axis=0))

                if is_training:
                    mel_outputs_ = tf.reshape(decoder_outputs,
                                              [batch_size, -1, hp.num_mels])
                else:
                    mel_outputs_ = tf.reshape(
                        decoder_outputs,
                        [batch_size, -1, hp.num_mels])[:, :ori_framenum, :]
                mel_outputs = mel_outputs_

            with tf.variable_scope('Postnet'):
                Postnet_FSMN = FsmnEncoderV2(
                    filter_size=hp.postnet_filter_size,
                    fsmn_num_layers=hp.postnet_fsmn_num_layers,
                    dnn_num_layers=hp.postnet_dnn_num_layers,
                    num_memory_units=hp.postnet_num_memory_units,
                    ffn_inner_dim=hp.postnet_ffn_inner_dim,
                    dropout=hp.postnet_dropout,
                    shift=hp.postnet_shift,
                    position_encoder=None)
                if is_training:
                    postnet_fsmn_outputs, _, _ = Postnet_FSMN.encode(
                        mel_outputs,
                        sequence_length=output_lengths,
                        mode=is_training)
                    hidden_lstm_outputs, _ = tf.nn.dynamic_rnn(
                        LSTMBlockCell(hp.postnet_lstm_units),
                        postnet_fsmn_outputs,
                        sequence_length=output_lengths,
                        dtype=tf.float32)
                else:
                    postnet_fsmn_outputs, _, _ = Postnet_FSMN.encode(
                        mel_outputs,
                        sequence_length=[tf.shape(mel_outputs_)[1]],
                        mode=is_training)
                    hidden_lstm_outputs, _ = tf.nn.dynamic_rnn(
                        LSTMBlockCell(hp.postnet_lstm_units),
                        postnet_fsmn_outputs,
                        sequence_length=[tf.shape(mel_outputs_)[1]],
                        dtype=tf.float32)

            mel_residual_outputs = tf.layers.dense(
                hidden_lstm_outputs, units=hp.num_mels)
            mel_outputs += mel_residual_outputs

            self.inputs = inputs
            self.inputs_speaker = inputs_speaker
            self.inputs_emotion = inputs_emotion
            self.input_lengths = input_lengths
            self.durations = durations
            self.output_lengths = output_lengths
            self.mel_outputs_ = mel_outputs_
            self.mel_outputs = mel_outputs
            self.mel_targets = mel_targets
            self.duration_outputs = duration_outputs
            self.duration_outputs_ = duration_outputs_
            self.duration_scales = duration_scales
            self.pitch_contour_outputs = pitch_contour_outputs
            self.pitch_contours = pitch_contours
            self.pitch_scales = pitch_scales
            self.energy_contour_outputs = energy_contour_outputs
            self.energy_contours = energy_contours
            self.energy_scales = energy_scales
            self.uv_masks_ = uv_masks

            self.embedded_inputs_emotion = embedded_inputs_emotion
            self.embedding_fsmn_outputs = embedded_inputs
            self.encoder_outputs = encoder_outputs
            self.encoder_outputs_ = encoder_outputs_
            self.LR_outputs = LR_outputs
            self.postnet_fsmn_outputs = postnet_fsmn_outputs

            self.pitch_embeddings = pitch_embeddings
            self.energy_embeddings = energy_embeddings

            self.attns = attns
            self.attention_x = attention_x
            self.attention_h = attention_h
            self.X_band_width = X_band_width
            self.H_band_width = H_band_width

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as _:
            hp = self._hparams
            mask = tf.sequence_mask(
                self.output_lengths,
                tf.shape(self.mel_targets)[1],
                dtype=tf.float32)
            valid_outputs = tf.reduce_sum(mask)

            mask_input = tf.sequence_mask(
                self.input_lengths,
                tf.shape(self.durations)[1],
                dtype=tf.float32)
            valid_inputs = tf.reduce_sum(mask_input)

            # mel loss
            if self.uv_masks_ is not None:
                valid_outputs_mask = tf.reduce_sum(
                    tf.expand_dims(mask, -1) * self.uv_masks_)
                self.mel_loss_ = tf.reduce_sum(
                    tf.abs(self.mel_targets - self.mel_outputs_)
                    * tf.expand_dims(mask, -1) * self.uv_masks_) / (
                        valid_outputs_mask * hp.num_mels)
                self.mel_loss = tf.reduce_sum(
                    tf.abs(self.mel_targets - self.mel_outputs)
                    * tf.expand_dims(mask, -1) * self.uv_masks_) / (
                        valid_outputs_mask * hp.num_mels)
            else:
                self.mel_loss_ = tf.reduce_sum(
                    tf.abs(self.mel_targets - self.mel_outputs_)
                    * tf.expand_dims(mask, -1)) / (
                        valid_outputs * hp.num_mels)
                self.mel_loss = tf.reduce_sum(
                    tf.abs(self.mel_targets - self.mel_outputs)
                    * tf.expand_dims(mask, -1)) / (
                        valid_outputs * hp.num_mels)

            # duration loss
            self.duration_loss = tf.reduce_sum(
                tf.abs(
                    tf.log(tf.cast(self.durations, tf.float32) + 1)
                    - self.duration_outputs) * mask_input) / valid_inputs

            # pitch contour loss
            self.pitch_contour_loss = tf.reduce_sum(
                tf.abs(self.pitch_contours - self.pitch_contour_outputs)
                * mask_input) / valid_inputs

            # energy contour loss
            self.energy_contour_loss = tf.reduce_sum(
                tf.abs(self.energy_contours - self.energy_contour_outputs)
                * mask_input) / valid_inputs

            # final loss
            self.loss = self.mel_loss_ + self.mel_loss + self.duration_loss \
                + self.pitch_contour_loss + self.energy_contour_loss

            # guided attention loss
            self.guided_attention_loss = tf.constant(0.0)
            if hp.guided_attention:
                i0 = tf.constant(0)
                loss0 = tf.constant(0.0)

                def c(i, _):
                    return tf.less(i, tf.shape(mel_targets)[0])

                def loop_body(i, loss):
                    decoder_input_lengths = tf.cast(
                        self.output_lengths / hp.outputs_per_step, tf.int32)
                    input_len = decoder_input_lengths[i]
                    output_len = decoder_input_lengths[i]
                    input_w = tf.expand_dims(
                        tf.range(tf.cast(input_len, dtype=tf.float32)),
                        axis=1) / tf.cast(
                            input_len, dtype=tf.float32)  # [T_in, 1]
                    output_w = tf.expand_dims(
                        tf.range(tf.cast(output_len, dtype=tf.float32)),
                        axis=0) / tf.cast(
                            output_len, dtype=tf.float32)  # [1, T_out]
                    guided_attention_w = 1.0 - tf.exp(
                        -(1 / hp.guided_attention_2g_squared)
                        * tf.square(input_w - output_w))  # [T_in, T_out]
                    guided_attention_w = tf.expand_dims(
                        guided_attention_w, axis=0)  # [1, T_in, T_out]
                    # [hp.decoder_num_heads, T_in, T_out]
                    guided_attention_w = tf.tile(guided_attention_w,
                                                 [hp.decoder_num_heads, 1, 1])
                    loss_i = tf.constant(0.0)
                    for j in range(hp.decoder_num_layers):
                        loss_i += tf.reduce_mean(
                            self.attention_h[j][i, :, :input_len, :output_len]
                            * guided_attention_w)

                    return [tf.add(i, 1), tf.add(loss, loss_i)]

                _, loss = tf.while_loop(
                    c,
                    loop_body,
                    loop_vars=[i0, loss0],
                    parallel_iterations=hp.batch_size)
                self.guided_attention_loss = loss / hp.batch_size
                self.loss += hp.guided_attention_loss_weight * self.guided_attention_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as _:
            hp = self._hparams
            if hp.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(
                    hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(
                    hp.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(
                    zip(clipped_gradients, variables), global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5,
                                                    step**-0.5)
