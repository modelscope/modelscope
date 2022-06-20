import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import AttentionWrapperState
from tensorflow.python.ops import rnn_cell_impl

from .modules import prenet


class VarPredictorCell(RNNCell):
    '''Wrapper wrapper knock knock.'''

    def __init__(self, var_predictor_cell, is_training, dim, prenet_units):
        super(VarPredictorCell, self).__init__()
        self._var_predictor_cell = var_predictor_cell
        self._is_training = is_training
        self._dim = dim
        self._prenet_units = prenet_units

    @property
    def state_size(self):
        return tuple([self.output_size, self._var_predictor_cell.state_size])

    @property
    def output_size(self):
        return self._dim

    def zero_state(self, batch_size, dtype):
        return tuple([
            rnn_cell_impl._zero_state_tensors(self.output_size, batch_size,
                                              dtype),
            self._var_predictor_cell.zero_state(batch_size, dtype)
        ])

    def call(self, inputs, state):
        '''Run the Tacotron2 super decoder cell.'''
        super_cell_out, decoder_state = state

        # split
        prenet_input = inputs[:, 0:self._dim]
        encoder_output = inputs[:, self._dim:]

        # prenet and concat
        prenet_output = prenet(
            prenet_input,
            self._prenet_units,
            self._is_training,
            scope='var_prenet')
        decoder_input = tf.concat([prenet_output, encoder_output], axis=-1)

        # decoder LSTM/GRU
        new_super_cell_out, new_decoder_state = self._var_predictor_cell(
            decoder_input, decoder_state)

        # projection
        new_super_cell_out = tf.layers.dense(
            new_super_cell_out, units=self._dim)

        new_states = tuple([new_super_cell_out, new_decoder_state])

        return new_super_cell_out, new_states


class DurPredictorCell(RNNCell):
    '''Wrapper wrapper knock knock.'''

    def __init__(self, var_predictor_cell, is_training, dim, prenet_units):
        super(DurPredictorCell, self).__init__()
        self._var_predictor_cell = var_predictor_cell
        self._is_training = is_training
        self._dim = dim
        self._prenet_units = prenet_units

    @property
    def state_size(self):
        return tuple([self.output_size, self._var_predictor_cell.state_size])

    @property
    def output_size(self):
        return self._dim

    def zero_state(self, batch_size, dtype):
        return tuple([
            rnn_cell_impl._zero_state_tensors(self.output_size, batch_size,
                                              dtype),
            self._var_predictor_cell.zero_state(batch_size, dtype)
        ])

    def call(self, inputs, state):
        '''Run the Tacotron2 super decoder cell.'''
        super_cell_out, decoder_state = state

        # split
        prenet_input = inputs[:, 0:self._dim]
        encoder_output = inputs[:, self._dim:]

        # prenet and concat
        prenet_output = prenet(
            prenet_input,
            self._prenet_units,
            self._is_training,
            scope='dur_prenet')
        decoder_input = tf.concat([prenet_output, encoder_output], axis=-1)

        # decoder LSTM/GRU
        new_super_cell_out, new_decoder_state = self._var_predictor_cell(
            decoder_input, decoder_state)

        # projection
        new_super_cell_out = tf.layers.dense(
            new_super_cell_out, units=self._dim)
        new_super_cell_out = tf.nn.relu(new_super_cell_out)
        #    new_super_cell_out = tf.log(tf.cast(tf.round(tf.exp(new_super_cell_out) - 1), tf.float32) + 1)

        new_states = tuple([new_super_cell_out, new_decoder_state])

        return new_super_cell_out, new_states


class DurPredictorCECell(RNNCell):
    '''Wrapper wrapper knock knock.'''

    def __init__(self, var_predictor_cell, is_training, dim, prenet_units,
                 max_dur, dur_embedding_dim):
        super(DurPredictorCECell, self).__init__()
        self._var_predictor_cell = var_predictor_cell
        self._is_training = is_training
        self._dim = dim
        self._prenet_units = prenet_units
        self._max_dur = max_dur
        self._dur_embedding_dim = dur_embedding_dim

    @property
    def state_size(self):
        return tuple([self.output_size, self._var_predictor_cell.state_size])

    @property
    def output_size(self):
        return self._max_dur

    def zero_state(self, batch_size, dtype):
        return tuple([
            rnn_cell_impl._zero_state_tensors(self.output_size, batch_size,
                                              dtype),
            self._var_predictor_cell.zero_state(batch_size, dtype)
        ])

    def call(self, inputs, state):
        '''Run the Tacotron2 super decoder cell.'''
        super_cell_out, decoder_state = state

        # split
        prenet_input = tf.squeeze(
            tf.cast(inputs[:, 0:self._dim], tf.int32), axis=-1)  # [N]
        prenet_input = tf.one_hot(
            prenet_input, self._max_dur, on_value=1.0, off_value=0.0,
            axis=-1)  # [N, 120]
        prenet_input = tf.layers.dense(
            prenet_input, units=self._dur_embedding_dim)
        encoder_output = inputs[:, self._dim:]

        # prenet and concat
        prenet_output = prenet(
            prenet_input,
            self._prenet_units,
            self._is_training,
            scope='dur_prenet')
        decoder_input = tf.concat([prenet_output, encoder_output], axis=-1)

        # decoder LSTM/GRU
        new_super_cell_out, new_decoder_state = self._var_predictor_cell(
            decoder_input, decoder_state)

        # projection
        new_super_cell_out = tf.layers.dense(
            new_super_cell_out, units=self._max_dur)  # [N, 120]
        new_super_cell_out = tf.nn.softmax(new_super_cell_out)  # [N, 120]

        new_states = tuple([new_super_cell_out, new_decoder_state])

        return new_super_cell_out, new_states


class VarPredictorCell2(RNNCell):
    '''Wrapper wrapper knock knock.'''

    def __init__(self, var_predictor_cell, is_training, dim, prenet_units):
        super(VarPredictorCell2, self).__init__()
        self._var_predictor_cell = var_predictor_cell
        self._is_training = is_training
        self._dim = dim
        self._prenet_units = prenet_units

    @property
    def state_size(self):
        return tuple([self.output_size, self._var_predictor_cell.state_size])

    @property
    def output_size(self):
        return self._dim

    def zero_state(self, batch_size, dtype):
        return tuple([
            rnn_cell_impl._zero_state_tensors(self.output_size, batch_size,
                                              dtype),
            self._var_predictor_cell.zero_state(batch_size, dtype)
        ])

    def call(self, inputs, state):
        '''Run the Tacotron2 super decoder cell.'''
        super_cell_out, decoder_state = state

        # split
        prenet_input = inputs[:, 0:self._dim]
        encoder_output = inputs[:, self._dim:]

        # prenet and concat
        prenet_output = prenet(
            prenet_input,
            self._prenet_units,
            self._is_training,
            scope='var_prenet')
        decoder_input = tf.concat([prenet_output, encoder_output], axis=-1)

        # decoder LSTM/GRU
        new_super_cell_out, new_decoder_state = self._var_predictor_cell(
            decoder_input, decoder_state)

        # projection
        new_super_cell_out = tf.layers.dense(
            new_super_cell_out, units=self._dim)

        # split and relu
        new_super_cell_out = tf.concat([
            tf.nn.relu(new_super_cell_out[:, 0:1]), new_super_cell_out[:, 1:]
        ], axis=-1)  # yapf:disable

        new_states = tuple([new_super_cell_out, new_decoder_state])

        return new_super_cell_out, new_states
