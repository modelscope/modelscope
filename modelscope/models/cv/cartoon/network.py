import tensorflow as tf
import tensorflow.contrib.slim as slim


def resblock(inputs, out_channel=32, name='resblock'):
    with tf.variable_scope(name):
        x = slim.convolution2d(
            inputs, out_channel, [3, 3], activation_fn=None, scope='conv1')

        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(
            x, out_channel, [3, 3], activation_fn=None, scope='conv2')

        return x + inputs


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        'u', [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(),
        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv_spectral_norm(x, channel, k_size, stride=1, name='conv_snorm'):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'kernel', shape=[k_size[0], k_size[1],
                             x.get_shape()[-1], channel])
        b = tf.get_variable(
            'bias', [channel], initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(
            input=x,
            filter=spectral_norm(w),
            strides=[1, stride, stride, 1],
            padding='SAME') + b

        return x


def unet_generator(inputs,
                   channel=32,
                   num_blocks=4,
                   name='generator',
                   reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(
            x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(
            x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(
                x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(
            x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        # x4 = tf.clip_by_value(x4, -1, 1)
        return x4


def disc_sn(x,
            scale=1,
            channel=32,
            patch=True,
            name='discriminator',
            reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        for idx in range(3):
            x = conv_spectral_norm(
                x,
                channel * 2**idx, [3, 3],
                stride=2,
                name='conv{}_1'.format(idx))
            x = tf.nn.leaky_relu(x)

            x = conv_spectral_norm(
                x, channel * 2**idx, [3, 3], name='conv{}_2'.format(idx))
            x = tf.nn.leaky_relu(x)

        if patch is True:
            x = conv_spectral_norm(x, 1, [1, 1], name='conv_out')

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


if __name__ == '__main__':
    pass
