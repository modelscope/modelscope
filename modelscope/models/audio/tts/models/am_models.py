import tensorflow as tf


def encoder_prenet(inputs,
                   n_conv_layers,
                   filters,
                   kernel_size,
                   dense_units,
                   is_training,
                   mask=None,
                   scope='encoder_prenet'):
    x = inputs
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                mask=mask,
                scope='conv1d_{}'.format(i))
        x = tf.layers.dense(
            x, units=dense_units, activation=None, name='dense')
    return x


def decoder_prenet(inputs,
                   prenet_units,
                   dense_units,
                   is_training,
                   scope='decoder_prenet'):
    x = inputs
    with tf.variable_scope(scope):
        for i, units in enumerate(prenet_units):
            x = tf.layers.dense(
                x,
                units=units,
                activation=tf.nn.relu,
                name='dense_{}'.format(i))
            x = tf.layers.dropout(
                x, rate=0.5, training=is_training, name='dropout_{}'.format(i))
        x = tf.layers.dense(
            x, units=dense_units, activation=None, name='dense')
    return x


def encoder(inputs,
            input_lengths,
            n_conv_layers,
            filters,
            kernel_size,
            lstm_units,
            is_training,
            embedded_inputs_speaker,
            mask=None,
            scope='encoder'):
    with tf.variable_scope(scope):
        x = conv_and_lstm(
            inputs,
            input_lengths,
            n_conv_layers,
            filters,
            kernel_size,
            lstm_units,
            is_training,
            embedded_inputs_speaker,
            mask=mask)
    return x


def prenet(inputs, prenet_units, is_training, scope='prenet'):
    x = inputs
    with tf.variable_scope(scope):
        for i, units in enumerate(prenet_units):
            x = tf.layers.dense(
                x,
                units=units,
                activation=tf.nn.relu,
                name='dense_{}'.format(i))
            x = tf.layers.dropout(
                x, rate=0.5, training=is_training, name='dropout_{}'.format(i))
    return x


def postnet_residual_ulstm(inputs,
                           n_conv_layers,
                           filters,
                           kernel_size,
                           lstm_units,
                           output_units,
                           is_training,
                           scope='postnet_residual_ulstm'):
    with tf.variable_scope(scope):
        x = conv_and_ulstm(inputs, None, n_conv_layers, filters, kernel_size,
                           lstm_units, is_training)
        x = conv1d(
            x,
            output_units,
            kernel_size,
            is_training,
            activation=None,
            dropout=False,
            scope='conv1d_{}'.format(n_conv_layers - 1))
    return x


def postnet_residual_lstm(inputs,
                          n_conv_layers,
                          filters,
                          kernel_size,
                          lstm_units,
                          output_units,
                          is_training,
                          scope='postnet_residual_lstm'):
    with tf.variable_scope(scope):
        x = conv_and_lstm(inputs, None, n_conv_layers, filters, kernel_size,
                          lstm_units, is_training)
        x = conv1d(
            x,
            output_units,
            kernel_size,
            is_training,
            activation=None,
            dropout=False,
            scope='conv1d_{}'.format(n_conv_layers - 1))
    return x


def postnet_linear_ulstm(inputs,
                         n_conv_layers,
                         filters,
                         kernel_size,
                         lstm_units,
                         output_units,
                         is_training,
                         scope='postnet_linear'):
    with tf.variable_scope(scope):
        x = conv_and_ulstm(inputs, None, n_conv_layers, filters, kernel_size,
                           lstm_units, is_training)
        x = tf.layers.dense(x, units=output_units)
    return x


def postnet_linear_lstm(inputs,
                        n_conv_layers,
                        filters,
                        kernel_size,
                        lstm_units,
                        output_units,
                        output_lengths,
                        is_training,
                        embedded_inputs_speaker2,
                        mask=None,
                        scope='postnet_linear'):
    with tf.variable_scope(scope):
        x = conv_and_lstm_dec(
            inputs,
            output_lengths,
            n_conv_layers,
            filters,
            kernel_size,
            lstm_units,
            is_training,
            embedded_inputs_speaker2,
            mask=mask)
        x = tf.layers.dense(x, units=output_units)
    return x


def postnet_linear(inputs,
                   n_conv_layers,
                   filters,
                   kernel_size,
                   lstm_units,
                   output_units,
                   output_lengths,
                   is_training,
                   embedded_inputs_speaker2,
                   mask=None,
                   scope='postnet_linear'):
    with tf.variable_scope(scope):
        x = conv_dec(
            inputs,
            output_lengths,
            n_conv_layers,
            filters,
            kernel_size,
            lstm_units,
            is_training,
            embedded_inputs_speaker2,
            mask=mask)
    return x


def conv_and_lstm(inputs,
                  sequence_lengths,
                  n_conv_layers,
                  filters,
                  kernel_size,
                  lstm_units,
                  is_training,
                  embedded_inputs_speaker,
                  mask=None,
                  scope='conv_and_lstm'):
    from tensorflow.contrib.rnn import LSTMBlockCell
    x = inputs
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                mask=mask,
                scope='conv1d_{}'.format(i))

        x = tf.concat([x, embedded_inputs_speaker], axis=2)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            LSTMBlockCell(lstm_units),
            LSTMBlockCell(lstm_units),
            x,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        x = tf.concat(outputs, axis=-1)

    return x


def conv_and_lstm_dec(inputs,
                      sequence_lengths,
                      n_conv_layers,
                      filters,
                      kernel_size,
                      lstm_units,
                      is_training,
                      embedded_inputs_speaker2,
                      mask=None,
                      scope='conv_and_lstm'):
    x = inputs
    from tensorflow.contrib.rnn import LSTMBlockCell
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                mask=mask,
                scope='conv1d_{}'.format(i))

        x = tf.concat([x, embedded_inputs_speaker2], axis=2)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            LSTMBlockCell(lstm_units),
            LSTMBlockCell(lstm_units),
            x,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        x = tf.concat(outputs, axis=-1)
    return x


def conv_dec(inputs,
             sequence_lengths,
             n_conv_layers,
             filters,
             kernel_size,
             lstm_units,
             is_training,
             embedded_inputs_speaker2,
             mask=None,
             scope='conv_and_lstm'):
    x = inputs
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                mask=mask,
                scope='conv1d_{}'.format(i))
        x = tf.concat([x, embedded_inputs_speaker2], axis=2)
    return x


def conv_and_ulstm(inputs,
                   sequence_lengths,
                   n_conv_layers,
                   filters,
                   kernel_size,
                   lstm_units,
                   is_training,
                   scope='conv_and_ulstm'):
    x = inputs
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                scope='conv1d_{}'.format(i))

        outputs, states = tf.nn.dynamic_rnn(
            LSTMBlockCell(lstm_units),
            x,
            sequence_length=sequence_lengths,
            dtype=tf.float32)

    return outputs


def conv1d(inputs,
           filters,
           kernel_size,
           is_training,
           activation=None,
           dropout=False,
           mask=None,
           scope='conv1d'):
    with tf.variable_scope(scope):
        if mask is not None:
            inputs = inputs * tf.expand_dims(mask, -1)
        x = tf.layers.conv1d(
            inputs, filters=filters, kernel_size=kernel_size, padding='same')
        if mask is not None:
            x = x * tf.expand_dims(mask, -1)

        x = tf.layers.batch_normalization(x, training=is_training)
        if activation is not None:
            x = activation(x)
        if dropout:
            x = tf.layers.dropout(x, rate=0.5, training=is_training)
    return x


def conv1d_dp(inputs,
              filters,
              kernel_size,
              is_training,
              activation=None,
              dropout=False,
              dropoutrate=0.5,
              mask=None,
              scope='conv1d'):
    with tf.variable_scope(scope):
        if mask is not None:
            inputs = inputs * tf.expand_dims(mask, -1)
        x = tf.layers.conv1d(
            inputs, filters=filters, kernel_size=kernel_size, padding='same')
        if mask is not None:
            x = x * tf.expand_dims(mask, -1)

        x = tf.contrib.layers.layer_norm(x)
        if activation is not None:
            x = activation(x)
        if dropout:
            x = tf.layers.dropout(x, rate=dropoutrate, training=is_training)
    return x


def duration_predictor(inputs,
                       n_conv_layers,
                       filters,
                       kernel_size,
                       lstm_units,
                       input_lengths,
                       is_training,
                       embedded_inputs_speaker,
                       mask=None,
                       scope='duration_predictor'):
    with tf.variable_scope(scope):
        x = inputs
        for i in range(n_conv_layers):
            x = conv1d_dp(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                dropoutrate=0.1,
                mask=mask,
                scope='conv1d_{}'.format(i))

        x = tf.concat([x, embedded_inputs_speaker], axis=2)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            LSTMBlockCell(lstm_units),
            LSTMBlockCell(lstm_units),
            x,
            sequence_length=input_lengths,
            dtype=tf.float32)
        x = tf.concat(outputs, axis=-1)

        x = tf.layers.dense(x, units=1)
        x = tf.nn.relu(x)
    return x


def duration_predictor2(inputs,
                        n_conv_layers,
                        filters,
                        kernel_size,
                        input_lengths,
                        is_training,
                        mask=None,
                        scope='duration_predictor'):
    with tf.variable_scope(scope):
        x = inputs
        for i in range(n_conv_layers):
            x = conv1d_dp(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                dropoutrate=0.1,
                mask=mask,
                scope='conv1d_{}'.format(i))

        x = tf.layers.dense(x, units=1)
        x = tf.nn.relu(x)
    return x


def conv_prenet(inputs,
                n_conv_layers,
                filters,
                kernel_size,
                is_training,
                mask=None,
                scope='conv_prenet'):
    x = inputs
    with tf.variable_scope(scope):
        for i in range(n_conv_layers):
            x = conv1d(
                x,
                filters,
                kernel_size,
                is_training,
                activation=tf.nn.relu,
                dropout=True,
                mask=mask,
                scope='conv1d_{}'.format(i))

    return x
