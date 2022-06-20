import tensorflow as tf

from . import fsmn


class FsmnEncoder():
    """Encoder using Fsmn
    """

    def __init__(self,
                 filter_size,
                 fsmn_num_layers,
                 dnn_num_layers,
                 num_memory_units=512,
                 ffn_inner_dim=2048,
                 dropout=0.0,
                 position_encoder=None):
        """Initializes the parameters of the encoder.

        Args:
          filter_size: the total order of memory block
          fsmn_num_layers: The number of fsmn layers.
          dnn_num_layers: The number of dnn layers
          num_units: The number of memory units.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(FsmnEncoder, self).__init__()
        self.filter_size = filter_size
        self.fsmn_num_layers = fsmn_num_layers
        self.dnn_num_layers = dnn_num_layers
        self.num_memory_units = num_memory_units
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.position_encoder = position_encoder

    def encode(self, inputs, sequence_length=None, mode=True):
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        inputs = tf.layers.dropout(inputs, rate=self.dropout, training=mode)

        mask = fsmn.build_sequence_mask(
            sequence_length, maximum_length=tf.shape(inputs)[1])

        state = ()

        for layer in range(self.fsmn_num_layers):
            with tf.variable_scope('fsmn_layer_{}'.format(layer)):
                with tf.variable_scope('ffn'):
                    context = fsmn.feed_forward(
                        inputs,
                        self.ffn_inner_dim,
                        self.num_memory_units,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope('memory'):
                    memory = fsmn.MemoryBlock(
                        context,
                        self.filter_size,
                        mode,
                        mask=mask,
                        dropout=self.dropout)

                    memory = fsmn.drop_and_add(
                        inputs, memory, mode, dropout=self.dropout)

                inputs = memory
                state += (tf.reduce_mean(inputs, axis=1), )

        for layer in range(self.dnn_num_layers):
            with tf.variable_scope('dnn_layer_{}'.format(layer)):
                transformed = fsmn.feed_forward(
                    inputs,
                    self.ffn_inner_dim,
                    self.num_memory_units,
                    mode,
                    dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1), )

        outputs = inputs
        return (outputs, state, sequence_length)


class FsmnEncoderV2():
    """Encoder using Fsmn
    """

    def __init__(self,
                 filter_size,
                 fsmn_num_layers,
                 dnn_num_layers,
                 num_memory_units=512,
                 ffn_inner_dim=2048,
                 dropout=0.0,
                 shift=0,
                 position_encoder=None):
        """Initializes the parameters of the encoder.

        Args:
          filter_size: the total order of memory block
          fsmn_num_layers: The number of fsmn layers.
          dnn_num_layers: The number of dnn layers
          num_units: The number of memory units.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          shift: left padding, to control delay
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(FsmnEncoderV2, self).__init__()
        self.filter_size = filter_size
        self.fsmn_num_layers = fsmn_num_layers
        self.dnn_num_layers = dnn_num_layers
        self.num_memory_units = num_memory_units
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.shift = shift
        if not isinstance(shift, list):
            self.shift = [shift for _ in range(self.fsmn_num_layers)]
        self.position_encoder = position_encoder

    def encode(self, inputs, sequence_length=None, mode=True):
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        inputs = tf.layers.dropout(inputs, rate=self.dropout, training=mode)

        mask = fsmn.build_sequence_mask(
            sequence_length, maximum_length=tf.shape(inputs)[1])

        state = ()
        for layer in range(self.fsmn_num_layers):
            with tf.variable_scope('fsmn_layer_{}'.format(layer)):
                with tf.variable_scope('ffn'):
                    context = fsmn.feed_forward(
                        inputs,
                        self.ffn_inner_dim,
                        self.num_memory_units,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope('memory'):
                    memory = fsmn.MemoryBlockV2(
                        context,
                        self.filter_size,
                        mode,
                        shift=self.shift[layer],
                        mask=mask,
                        dropout=self.dropout)

                    memory = fsmn.drop_and_add(
                        inputs, memory, mode, dropout=self.dropout)

                inputs = memory
                state += (tf.reduce_mean(inputs, axis=1), )

        for layer in range(self.dnn_num_layers):
            with tf.variable_scope('dnn_layer_{}'.format(layer)):
                transformed = fsmn.feed_forward(
                    inputs,
                    self.ffn_inner_dim,
                    self.num_memory_units,
                    mode,
                    dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1), )

        outputs = inputs
        return (outputs, state, sequence_length)
