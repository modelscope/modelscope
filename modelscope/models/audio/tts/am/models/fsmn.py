import tensorflow as tf


def build_sequence_mask(sequence_length,
                        maximum_length=None,
                        dtype=tf.float32):
    """Builds the dot product mask.

    Args:
      sequence_length: The sequence length.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The type of the mask tensor.

    Returns:
      A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[batch_size, max_length]``.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=maximum_length, dtype=dtype)

    return mask


def norm(inputs):
    """Layer normalizes :obj:`inputs`."""
    return tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)


def pad_in_time(x, padding_shape):
    """Helper function to pad a tensor in the time dimension and retain the static depth dimension.

       Agrs:
        x: [Batch, Time, Frequency]
        padding_length: padding size of constant value (0) before the time dimension

      return:
        padded x
    """

    depth = x.get_shape().as_list()[-1]
    x = tf.pad(x, [[0, 0], padding_shape, [0, 0]])
    x.set_shape((None, None, depth))

    return x


def pad_in_time_right(x, padding_length):
    """Helper function to pad a tensor in the time dimension and retain the static depth dimension.

       Agrs:
        x: [Batch, Time, Frequency]
        padding_length: padding size of constant value (0) before the time dimension

      return:
        padded x
    """
    depth = x.get_shape().as_list()[-1]
    x = tf.pad(x, [[0, 0], [0, padding_length], [0, 0]])
    x.set_shape((None, None, depth))

    return x


def feed_forward(x, ffn_dim, memory_units, mode, dropout=0.0):
    """Implements the Transformer's "Feed Forward" layer.

    .. math::

        ffn(x) = max(0, x*W_1 + b_1)*W_2

    Args:
      x: The input.
      ffn_dim: The number of units of the nonlinear transformation.
      memory_units: the number of units of linear transformation
      mode: A ``tf.estimator.ModeKeys`` mode.
      dropout: The probability to drop units from the inner transformation.

    Returns:
      The transformed input.
    """
    inner = tf.layers.conv1d(x, ffn_dim, 1, activation=tf.nn.relu)
    inner = tf.layers.dropout(
        inner, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    outer = tf.layers.conv1d(inner, memory_units, 1, use_bias=False)

    return outer


def drop_and_add(inputs, outputs, mode, dropout=0.0):
    """Drops units in the outputs and adds the previous values.

    Args:
      inputs: The input of the previous layer.
      outputs: The output of the previous layer.
      mode: A ``tf.estimator.ModeKeys`` mode.
      dropout: The probability to drop units in :obj:`outputs`.

    Returns:
      The residual and normalized output.
    """
    outputs = tf.layers.dropout(outputs, rate=dropout, training=mode)

    input_dim = inputs.get_shape().as_list()[-1]
    output_dim = outputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        outputs += inputs

    return outputs


def MemoryBlock(
    inputs,
    filter_size,
    mode,
    mask=None,
    dropout=0.0,
):
    """
    Define the bidirectional memory block in FSMN

    Agrs:
      inputs: The output of the previous layer. [Batch, Time, Frequency]
      filter_size: memory block filter size
      mode: Training or Evaluation
      mask: A ``tf.Tensor`` applied to the memory block output

    return:
      output: 3-D tensor ([Batch, Time, Frequency])
    """
    static_shape = inputs.get_shape().as_list()
    depth = static_shape[-1]
    inputs = tf.expand_dims(inputs, axis=1)  # [Batch, 1, Time, Frequency]
    depthwise_filter = tf.get_variable(
        'depth_conv_w',
        shape=[1, filter_size, depth, 1],
        initializer=tf.glorot_uniform_initializer(),
        dtype=tf.float32)
    memory = tf.nn.depthwise_conv2d(
        input=inputs,
        filter=depthwise_filter,
        strides=[1, 1, 1, 1],
        padding='SAME',
        rate=[1, 1],
        data_format='NHWC')
    memory = memory + inputs
    output = tf.layers.dropout(memory, rate=dropout, training=mode)
    output = tf.reshape(
        output,
        [tf.shape(output)[0], tf.shape(output)[2], depth])
    if mask is not None:
        output = output * tf.expand_dims(mask, -1)

    return output


def MemoryBlockV2(
    inputs,
    filter_size,
    mode,
    shift=0,
    mask=None,
    dropout=0.0,
):
    """
    Define the bidirectional memory block in FSMN

    Agrs:
      inputs: The output of the previous layer. [Batch, Time, Frequency]
      filter_size: memory block filter size
      mode: Training or Evaluation
      shift: left padding, to control delay
      mask: A ``tf.Tensor`` applied to the memory block output

    return:
      output: 3-D tensor ([Batch, Time, Frequency])
    """
    if mask is not None:
        inputs = inputs * tf.expand_dims(mask, -1)

    static_shape = inputs.get_shape().as_list()
    depth = static_shape[-1]
    # padding
    left_padding = int(round((filter_size - 1) / 2))
    right_padding = int((filter_size - 1) / 2)
    if shift > 0:
        left_padding = left_padding + shift
        right_padding = right_padding - shift
    pad_inputs = pad_in_time(inputs, [left_padding, right_padding])
    pad_inputs = tf.expand_dims(
        pad_inputs, axis=1)  # [Batch, 1, Time, Frequency]
    depthwise_filter = tf.get_variable(
        'depth_conv_w',
        shape=[1, filter_size, depth, 1],
        initializer=tf.glorot_uniform_initializer(),
        dtype=tf.float32)
    memory = tf.nn.depthwise_conv2d(
        input=pad_inputs,
        filter=depthwise_filter,
        strides=[1, 1, 1, 1],
        padding='VALID',
        rate=[1, 1],
        data_format='NHWC')
    memory = tf.reshape(
        memory,
        [tf.shape(memory)[0], tf.shape(memory)[2], depth])
    memory = memory + inputs
    output = tf.layers.dropout(memory, rate=dropout, training=mode)
    if mask is not None:
        output = output * tf.expand_dims(mask, -1)

    return output


def UniMemoryBlock(
    inputs,
    filter_size,
    mode,
    cache=None,
    mask=None,
    dropout=0.0,
):
    """
    Define the unidirectional memory block in FSMN

    Agrs:
      inputs: The output of the previous layer. [Batch, Time, Frequency]
      filter_size: memory block filter size
      cache: for streaming inference
      mode: Training or Evaluation
      mask: A ``tf.Tensor`` applied to the memory block output
      dropout: dorpout factor
    return:
      output: 3-D tensor ([Batch, Time, Frequency])
    """
    if cache is not None:
        static_shape = cache['queries'].get_shape().as_list()
        depth = static_shape[-1]
        queries = tf.slice(cache['queries'], [0, 1, 0], [
            tf.shape(cache['queries'])[0],
            tf.shape(cache['queries'])[1] - 1, depth
        ])
        queries = tf.concat([queries, inputs], axis=1)
        cache['queries'] = queries
    else:
        padding_length = filter_size - 1
        queries = pad_in_time(inputs, [padding_length, 0])

    queries = tf.expand_dims(queries, axis=1)  # [Batch, 1, Time, Frequency]
    static_shape = queries.get_shape().as_list()
    depth = static_shape[-1]
    depthwise_filter = tf.get_variable(
        'depth_conv_w',
        shape=[1, filter_size, depth, 1],
        initializer=tf.glorot_uniform_initializer(),
        dtype=tf.float32)
    memory = tf.nn.depthwise_conv2d(
        input=queries,
        filter=depthwise_filter,
        strides=[1, 1, 1, 1],
        padding='VALID',
        rate=[1, 1],
        data_format='NHWC')
    memory = tf.reshape(
        memory,
        [tf.shape(memory)[0], tf.shape(memory)[2], depth])
    memory = memory + inputs
    output = tf.layers.dropout(memory, rate=dropout, training=mode)
    if mask is not None:
        output = output * tf.expand_dims(mask, -1)

    return output
