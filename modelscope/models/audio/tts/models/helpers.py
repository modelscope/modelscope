import numpy as np
import tensorflow as tf


class VarTestHelper(tf.contrib.seq2seq.Helper):

    def __init__(self, batch_size, inputs, dim):
        with tf.name_scope('VarTestHelper'):
            self._batch_size = batch_size
            self._inputs = inputs
            self._dim = dim

            num_steps = tf.shape(self._inputs)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

            self._inputs = tf.roll(inputs, shift=-1, axis=1)
            self._init_inputs = inputs[:, 0, :]

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]),
                _go_frames(self._batch_size, self._dim, self._init_inputs))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope('VarTestHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = tf.concat([outputs, self._inputs[:, time, :]],
                                    axis=-1)
            return (finished, next_inputs, state)


class VarTrainingHelper(tf.contrib.seq2seq.Helper):

    def __init__(self, targets, inputs, dim):
        with tf.name_scope('VarTrainingHelper'):
            self._targets = targets  # [N, T_in, 1]
            self._batch_size = tf.shape(inputs)[0]  # N
            self._inputs = inputs
            self._dim = dim

            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

            self._inputs = tf.roll(inputs, shift=-1, axis=1)
            self._init_inputs = inputs[:, 0, :]

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]),
                _go_frames(self._batch_size, self._dim, self._init_inputs))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'VarTrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = tf.concat(
                [self._targets[:, time, :], self._inputs[:, time, :]], axis=-1)
            return (finished, next_inputs, state)


class VarTrainingSSHelper(tf.contrib.seq2seq.Helper):

    def __init__(self, targets, inputs, dim, global_step, schedule_begin,
                 alpha, decay_steps):
        with tf.name_scope('VarTrainingSSHelper'):
            self._targets = targets  # [N, T_in, 1]
            self._batch_size = tf.shape(inputs)[0]  # N
            self._inputs = inputs
            self._dim = dim

            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

            self._inputs = tf.roll(inputs, shift=-1, axis=1)
            self._init_inputs = inputs[:, 0, :]

            # for schedule sampling
            self._global_step = global_step
            self._schedule_begin = schedule_begin
            self._alpha = alpha
            self._decay_steps = decay_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        self._ratio = _tf_decay(self._global_step, self._schedule_begin,
                                self._alpha, self._decay_steps)
        return (tf.tile([False], [self._batch_size]),
                _go_frames(self._batch_size, self._dim, self._init_inputs))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'VarTrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs_tmp = tf.cond(
                tf.less(
                    tf.random_uniform([], minval=0, maxval=1,
                                      dtype=tf.float32), self._ratio),
                lambda: self._targets[:, time, :], lambda: outputs)
            next_inputs = tf.concat(
                [next_inputs_tmp, self._inputs[:, time, :]], axis=-1)
            return (finished, next_inputs, state)


def _go_frames(batch_size, dim, init_inputs):
    '''Returns all-zero <GO> frames for a given batch size and output dimension'''
    return tf.concat([tf.tile([[0.0]], [batch_size, dim]), init_inputs],
                     axis=-1)


def _tf_decay(global_step, schedule_begin, alpha, decay_steps):
    tfr = tf.train.exponential_decay(
        1.0,
        global_step=global_step - schedule_begin,
        decay_steps=decay_steps,
        decay_rate=alpha,
        name='tfr_decay')
    final_tfr = tf.cond(
        tf.less(global_step, schedule_begin), lambda: 1.0, lambda: tfr)
    return final_tfr
