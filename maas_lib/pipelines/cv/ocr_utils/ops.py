import math
import os
import shutil
import uuid

import cv2
import numpy as np
import tensorflow as tf

from . import utils

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('weight_init_method', 'xavier',
                           'Weight initialization method')

LIB_NAME = 'seglink'

# constants
OFFSET_DIM = 6

N_LOCAL_LINKS = 8
N_CROSS_LINKS = 4
N_SEG_CLASSES = 2
N_LNK_CLASSES = 4

MATCH_STATUS_POS = 1
MATCH_STATUS_NEG = -1
MATCH_STATUS_IGNORE = 0
MUT_LABEL = 3
POS_LABEL = 1
NEG_LABEL = 0

N_DET_LAYERS = 6


def load_oplib(lib_name):
    """
  Load TensorFlow operator library.
  """
    # use absolute path so that ops.py can be called from other directory
    lib_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'lib{0}.so'.format(lib_name))
    # duplicate library with a random new name so that
    # a running program will not be interrupted when the original library is updated
    lib_copy_path = '/tmp/lib{0}_{1}.so'.format(
        str(uuid.uuid4())[:8], LIB_NAME)
    shutil.copyfile(lib_path, lib_copy_path)
    oplib = tf.load_op_library(lib_copy_path)
    return oplib


def _nn_variable(name, shape, init_method, collection=None, **kwargs):
    """
  Create or reuse a variable
  ARGS
    name: variable name
    shape: variable shape
    init_method: 'zero', 'kaiming', 'xavier', or (mean, std)
    collection: if not none, add variable to this collection
    kwargs: extra paramters passed to tf.get_variable
  RETURN
    var: a new or existing variable
  """
    if init_method == 'zero':
        initializer = tf.constant_initializer(0.0)
    elif init_method == 'kaiming':
        if len(shape) == 4:  # convolutional filters
            kh, kw, n_in = shape[:3]
            init_std = math.sqrt(2.0 / (kh * kw * n_in))
        elif len(shape) == 2:  # linear weights
            n_in, n_out = shape
            init_std = math.sqrt(1.0 / n_out)
        else:
            raise 'Unsupported shape'
        initializer = tf.truncated_normal_initializer(0.0, init_std)
    elif init_method == 'xavier':
        if len(shape) == 4:
            initializer = tf.keras.initializers.glorot_normal()
            # initializer = tf.contrib.layers.xavier_initializer_conv2d()
        else:
            initializer = tf.keras.initializers.glorot_normal()
            # initializer = tf.contrib.layers.xavier_initializer()
    elif isinstance(init_method, tuple):
        assert (len(init_method) == 2)
        initializer = tf.truncated_normal_initializer(init_method[0],
                                                      init_method[1])
    else:
        raise 'Unsupported weight initialization method: ' + init_method

    var = tf.get_variable(name, shape=shape, initializer=initializer, **kwargs)
    if collection is not None:
        tf.add_to_collection(collection, var)

    return var


def conv2d(x,
           n_in,
           n_out,
           ksize,
           stride=1,
           padding='SAME',
           weight_init=None,
           bias=True,
           relu=False,
           scope=None,
           **kwargs):
    weight_init = weight_init or FLAGS.weight_init_method
    trainable = kwargs.get('trainable', True)
    # input_dim = n_in
    if (padding == 'SAME'):
        in_height = x.get_shape()[1]
        in_width = x.get_shape()[2]
        if (in_height % stride == 0):
            pad_along_height = max(ksize - stride, 0)
        else:
            pad_along_height = max(ksize - (in_height % stride), 0)
        if (in_width % stride == 0):
            pad_along_width = max(ksize - stride, 0)
        else:
            pad_along_width = max(ksize - (in_width % stride), 0)
        pad_bottom = pad_along_height // 2
        pad_top = pad_along_height - pad_bottom
        pad_right = pad_along_width // 2
        pad_left = pad_along_width - pad_right
        paddings = tf.constant([[0, 0], [pad_top, pad_bottom],
                                [pad_left, pad_right], [0, 0]])
        input_padded = tf.pad(x, paddings, 'CONSTANT')
    else:
        input_padded = x

    with tf.variable_scope(scope or 'conv2d'):
        # convolution
        kernel = _nn_variable(
            'weight', [ksize, ksize, n_in, n_out],
            weight_init,
            collection='weights' if trainable else None,
            **kwargs)
        yc = tf.nn.conv2d(
            input_padded, kernel, [1, stride, stride, 1], padding='VALID')
        # add bias
        # print(^^^^^^^^^^^^^^^^^^^^^^^,bias)
        if bias is True:
            bias = _nn_variable(
                'bias', [n_out],
                'zero',
                collection='biases' if trainable else None,
                **kwargs)
            yb = tf.nn.bias_add(yc, bias)
        # apply ReLU
        y = yb
        if relu is True:
            y = tf.nn.relu(yb)
    return yb, y


def group_conv2d_relu(x,
                      n_in,
                      n_out,
                      ksize,
                      stride=1,
                      group=4,
                      padding='SAME',
                      weight_init=None,
                      bias=True,
                      relu=False,
                      name='group_conv2d',
                      **kwargs):
    group_axis = len(x.get_shape()) - 1
    splits = tf.split(x, [int(n_in / group)] * group, group_axis)

    conv_list = []
    for i in range(group):
        conv_split, relu_split = conv2d(
            splits[i],
            n_in / group,
            n_out / group,
            ksize=ksize,
            stride=stride,
            padding=padding,
            weight_init=weight_init,
            bias=bias,
            relu=relu,
            scope='%s_%d' % (name, i))
        conv_list.append(conv_split)
    conv = tf.concat(values=conv_list, axis=group_axis, name=name + '_concat')
    relu = tf.nn.relu(conv)
    return conv, relu


def group_conv2d_bn_relu(x,
                         n_in,
                         n_out,
                         ksize,
                         stride=1,
                         group=4,
                         padding='SAME',
                         weight_init=None,
                         bias=True,
                         relu=False,
                         name='group_conv2d',
                         **kwargs):
    group_axis = len(x.get_shape()) - 1
    splits = tf.split(x, [int(n_in / group)] * group, group_axis)

    conv_list = []
    for i in range(group):
        conv_split, relu_split = conv2d(
            splits[i],
            n_in / group,
            n_out / group,
            ksize=ksize,
            stride=stride,
            padding=padding,
            weight_init=weight_init,
            bias=bias,
            relu=relu,
            scope='%s_%d' % (name, i))
        conv_list.append(conv_split)
    conv = tf.concat(values=conv_list, axis=group_axis, name=name + '_concat')
    with tf.variable_scope(name + '_bn'):
        bn = tf.layers.batch_normalization(
            conv, momentum=0.9, epsilon=1e-5, scale=True, training=True)
    relu = tf.nn.relu(bn)
    return conv, relu


def next_conv(x,
              n_in,
              n_out,
              ksize,
              stride=1,
              group=4,
              padding='SAME',
              weight_init=None,
              bias=True,
              relu=False,
              name='next_conv2d',
              **kwargs):
    conv_a, relu_a = conv_relu(
        x,
        n_in,
        n_in / 2,
        ksize=1,
        stride=1,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        scope=name + '_a',
        **kwargs)

    conv_b, relu_b = group_conv2d_relu(
        relu_a,
        n_in / 2,
        n_out / 2,
        ksize=ksize,
        stride=stride,
        group=group,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        name=name + '_b',
        **kwargs)

    conv_c, relu_c = conv_relu(
        relu_b,
        n_out / 2,
        n_out,
        ksize=1,
        stride=1,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        scope=name + '_c',
        **kwargs)

    return conv_c, relu_c


def next_conv_bn(x,
                 n_in,
                 n_out,
                 ksize,
                 stride=1,
                 group=4,
                 padding='SAME',
                 weight_init=None,
                 bias=True,
                 relu=False,
                 name='next_conv2d',
                 **kwargs):
    conv_a, relu_a = conv_bn_relu(
        x,
        n_in,
        n_in / 2,
        ksize=1,
        stride=1,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        scope=name + '_a',
        **kwargs)

    conv_b, relu_b = group_conv2d_bn_relu(
        relu_a,
        n_in / 2,
        n_out / 2,
        ksize=ksize,
        stride=stride,
        group=group,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        name=name + '_b',
        **kwargs)

    conv_c, relu_c = conv_bn_relu(
        relu_b,
        n_out / 2,
        n_out,
        ksize=1,
        stride=1,
        padding=padding,
        weight_init=weight_init,
        bias=bias,
        relu=relu,
        scope=name + '_c',
        **kwargs)

    return conv_c, relu_c


def conv2d_ori(x,
               n_in,
               n_out,
               ksize,
               stride=1,
               padding='SAME',
               weight_init=None,
               bias=True,
               relu=False,
               scope=None,
               **kwargs):
    weight_init = weight_init or FLAGS.weight_init_method
    trainable = kwargs.get('trainable', True)

    with tf.variable_scope(scope or 'conv2d'):
        # convolution
        kernel = _nn_variable(
            'weight', [ksize, ksize, n_in, n_out],
            weight_init,
            collection='weights' if trainable else None,
            **kwargs)
        y = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
        # add bias
        if bias is True:
            bias = _nn_variable(
                'bias', [n_out],
                'zero',
                collection='biases' if trainable else None,
                **kwargs)
            y = tf.nn.bias_add(y, bias)
        # apply ReLU
        if relu is True:
            y = tf.nn.relu(y)
    return y


def conv_relu(*args, **kwargs):
    kwargs['relu'] = True
    if 'scope' not in kwargs:
        kwargs['scope'] = 'conv_relu'
    return conv2d(*args, **kwargs)


def conv_bn_relu(*args, **kwargs):
    kwargs['relu'] = True
    if 'scope' not in kwargs:
        kwargs['scope'] = 'conv_relu'
    conv, relu = conv2d(*args, **kwargs)
    with tf.variable_scope(kwargs['scope'] + '_bn'):
        bn = tf.layers.batch_normalization(
            conv, momentum=0.9, epsilon=1e-5, scale=True, training=True)
    bn_relu = tf.nn.relu(bn)
    return bn, bn_relu


def conv_relu_ori(*args, **kwargs):
    kwargs['relu'] = True
    if 'scope' not in kwargs:
        kwargs['scope'] = 'conv_relu'
    return conv2d_ori(*args, **kwargs)


def atrous_conv2d(x,
                  n_in,
                  n_out,
                  ksize,
                  dilation,
                  padding='SAME',
                  weight_init=None,
                  bias=True,
                  relu=False,
                  scope=None,
                  **kwargs):
    weight_init = weight_init or FLAGS.weight_init_method
    trainable = kwargs.get('trainable', True)
    with tf.variable_scope(scope or 'atrous_conv2d'):
        # atrous convolution
        kernel = _nn_variable(
            'weight', [ksize, ksize, n_in, n_out],
            weight_init,
            collection='weights' if trainable else None,
            **kwargs)
        y = tf.nn.atrous_conv2d(x, kernel, dilation, padding=padding)
        # add bias
        if bias is True:
            bias = _nn_variable(
                'bias', [n_out],
                'zero',
                collection='biases' if trainable else None,
                **kwargs)
            y = tf.nn.bias_add(y, bias)
        # apply ReLU
        if relu is True:
            y = tf.nn.relu(y)
        return y


def avg_pool(x, ksize, stride, padding='SAME', scope=None):
    with tf.variable_scope(scope or 'avg_pool'):
        y = tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                           padding)
    return y


def max_pool(x, ksize, stride, padding='SAME', scope=None):
    with tf.variable_scope(scope or 'max_pool'):
        y = tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                           padding)
    return y


def score_loss(gt_labels, match_scores, n_classes):
    """
  Classification loss
  ARGS
    gt_labels: int32 [n]
    match_scores: [n, n_classes]
  RETURN
    loss
  """
    embeddings = tf.one_hot(tf.cast(gt_labels, tf.int64), n_classes, 1.0, 0.0)
    losses = tf.nn.softmax_cross_entropy_with_logits(match_scores, embeddings)
    return tf.reduce_sum(losses)


def smooth_l1_loss(offsets, gt_offsets, scope=None):
    """
  Smooth L1 loss between offsets and encoded_gt
  ARGS
    offsets: [m?, 5], predicted offsets for one example
    gt_offsets: [m?, 5], correponding groundtruth offsets
  RETURN
    loss: scalar
  """
    with tf.variable_scope(scope or 'smooth_l1_loss'):
        gt_offsets = tf.stop_gradient(gt_offsets)
        diff = tf.abs(offsets - gt_offsets)
        lesser_mask = tf.cast(tf.less(diff, 1.0), tf.float32)
        larger_mask = 1.0 - lesser_mask
        losses = (0.5 * tf.square(diff)) * lesser_mask + (diff 
                                                          - 0.5) * larger_mask
        return tf.reduce_sum(losses, 1)


def polygon_to_rboxe(polygon):
    x1 = polygon[0]
    y1 = polygon[1]
    x2 = polygon[2]
    y2 = polygon[3]
    x3 = polygon[4]
    y3 = polygon[5]
    x4 = polygon[6]
    y4 = polygon[7]
    c_x = (x1 + x2 + x3 + x4) / 4
    c_y = (y1 + y2 + y3 + y4) / 4
    w1 = point_dist(x1, y1, x2, y2)
    w2 = point_dist(x3, y3, x4, y4)
    h1 = point_line_dist(c_x, c_y, x1, y1, x2, y2)
    h2 = point_line_dist(c_x, c_y, x3, y3, x4, y4)
    h = h1 + h2
    w = (w1 + w2) / 2
    theta1 = np.arctan2(y2 - y1, x2 - x1)
    theta2 = np.arctan2(y3 - y4, x3 - x4)
    theta = (theta1 + theta2) / 2
    return np.array([c_x, c_y, w, h, theta])


def point_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def point_line_dist(px, py, x1, y1, x2, y2):
    eps = 1e-6
    dx = x2 - x1
    dy = y2 - y1
    div = np.sqrt(dx * dx + dy * dy) + eps
    dist = np.abs(px * dy - py * dx + x2 * y1 - y2 * x1) / div
    return dist


def get_combined_polygon(rboxes, resize_size):
    image_w = resize_size[1]
    image_h = resize_size[0]
    img = np.zeros((image_h, image_w, 3), np.uint8)
    for i in range(rboxes.shape[0]):
        segment = np.reshape(
            np.array(utils.rboxes_to_polygons(rboxes)[i, :], np.int32),
            (-1, 1, 2))
        cv2.drawContours(img, [segment], 0, (255, 255, 255), -1)
    # cv2.imwrite('/home/xixing.tj/visu/visu_mask/full_mask.jpg', img)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # print(len(contours))
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)
        # get max_area
        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)
        rect = cv2.minAreaRect(cnt)
        combined_polygon = np.array(cv2.boxPoints(rect)).reshape(-1)
        print(combined_polygon)
        # visu approx polygon
        # img = np.zeros((image_h, image_w, 3), np.uint8)
        # cv2.drawContours(img, [approx], 0, (255,255,255), -1)
        # cv2.imwrite('/home/xixing.tj/visu/visu_mask/approx_mask.jpg', img)
        # input()
    else:
        combined_polygon = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    return combined_polygon


# decode_segments_links rewrite in python version
def get_coord(offsets, map_size, offsets_defaults):
    if offsets < offsets_defaults[1][0]:
        l_idx = 0
        x = offsets % map_size[0][1]
        y = offsets // map_size[0][1]
    elif offsets < offsets_defaults[2][0]:
        l_idx = 1
        x = (offsets - offsets_defaults[1][0]) % map_size[1][1]
        y = (offsets - offsets_defaults[1][0]) // map_size[1][1]
    elif offsets < offsets_defaults[3][0]:
        l_idx = 2
        x = (offsets - offsets_defaults[2][0]) % map_size[2][1]
        y = (offsets - offsets_defaults[2][0]) // map_size[2][1]
    elif offsets < offsets_defaults[4][0]:
        l_idx = 3
        x = (offsets - offsets_defaults[3][0]) % map_size[3][1]
        y = (offsets - offsets_defaults[3][0]) // map_size[3][1]
    elif offsets < offsets_defaults[5][0]:
        l_idx = 4
        x = (offsets - offsets_defaults[4][0]) % map_size[4][1]
        y = (offsets - offsets_defaults[4][0]) // map_size[4][1]
    else:
        l_idx = 5
        x = (offsets - offsets_defaults[5][0]) % map_size[5][1]
        y = (offsets - offsets_defaults[5][0]) // map_size[5][1]

    return l_idx, x, y


def get_coord_link(offsets, map_size, offsets_defaults):
    if offsets < offsets_defaults[1][1]:
        offsets_node = offsets // N_LOCAL_LINKS
        link_idx = offsets % N_LOCAL_LINKS
    else:
        offsets_node = (offsets - offsets_defaults[1][1]) // (
            N_LOCAL_LINKS + N_CROSS_LINKS) + offsets_defaults[1][0]
        link_idx = (offsets - offsets_defaults[1][1]) % (
            N_LOCAL_LINKS + N_CROSS_LINKS)
    l_idx, x, y = get_coord(offsets_node, map_size, offsets_defaults)
    return l_idx, x, y, link_idx


def is_valid_coord(l_idx, x, y, map_size):
    w = map_size[l_idx][1]
    h = map_size[l_idx][0]
    return x >= 0 and x < w and y >= 0 and y < h


def get_neighbours(l_idx, x, y, map_size, offsets_defaults):
    if l_idx == 0:
        coord = [(0, x - 1, y - 1), (0, x, y - 1), (0, x + 1, y - 1),
                 (0, x - 1, y), (0, x + 1, y), (0, x - 1, y + 1),
                 (0, x, y + 1), (0, x + 1, y + 1)]
    else:
        coord = [(l_idx, x - 1, y - 1),
                 (l_idx, x, y - 1), (l_idx, x + 1, y - 1), (l_idx, x - 1, y),
                 (l_idx, x + 1, y), (l_idx, x - 1, y + 1), (l_idx, x, y + 1),
                 (l_idx, x + 1, y + 1), (l_idx - 1, 2 * x, 2 * y),
                 (l_idx - 1, 2 * x + 1, 2 * y), (l_idx - 1, 2 * x, 2 * y + 1),
                 (l_idx - 1, 2 * x + 1, 2 * y + 1)]
    neighbours_offsets = []
    link_idx = 0
    for nl_idx, nx, ny in coord:
        if is_valid_coord(nl_idx, nx, ny, map_size):
            neighbours_offset_node = offsets_defaults[nl_idx][
                0] + map_size[nl_idx][1] * ny + nx
            if l_idx == 0:
                neighbours_offset_link = offsets_defaults[l_idx][1] + (
                    map_size[l_idx][1] * y + x) * N_LOCAL_LINKS + link_idx
            else:
                neighbours_offset_link = offsets_defaults[l_idx][1] + (
                    map_size[l_idx][1] * y + x) * (N_LOCAL_LINKS 
                                                   + N_CROSS_LINKS) + link_idx
            neighbours_offsets.append(
                [neighbours_offset_node, neighbours_offset_link, link_idx])
        link_idx += 1
    # [node_offsets, link_offsets, link_idx(0-7/11)]
    return neighbours_offsets


def decode_segments_links_(image_size, all_nodes, all_links, all_reg,
                           anchor_sizes):
    batch_size = FLAGS.test_batch_size
    # offsets = 12285 #768
    all_nodes_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, N_SEG_CLASSES]) for o in all_nodes],
        axis=1)
    all_links_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, N_LNK_CLASSES]) for o in all_links],
        axis=1)
    all_reg_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, OFFSET_DIM]) for o in all_reg], axis=1)
    segments, group_indices, segment_counts, group_indices_all = tf.py_func(
        decode_batch, [
            all_nodes_flat, all_links_flat, all_reg_flat, image_size,
            tf.constant(anchor_sizes)
        ], [tf.float32, tf.int32, tf.int32, tf.int32])
    # TO DO: tf.set_shape(xixing.tj)
    # segments.set_shape([batch_size, offsets, OFFSET_DIM])
    # group_indices.set_shape([batch_size, offsets])
    # segment_counts.set_shape([batch_size])
    # group_indices_all.set_shape([batch_size, offsets])
    return segments, group_indices, segment_counts, group_indices_all


def decode_segments_links_train(image_size, all_nodes, all_links, all_reg,
                                anchor_sizes):
    batch_size = FLAGS.train_batch_size
    # offsets = 12285 #768
    all_nodes_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, N_SEG_CLASSES]) for o in all_nodes],
        axis=1)
    all_links_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, N_LNK_CLASSES]) for o in all_links],
        axis=1)
    all_reg_flat = tf.concat(
        [tf.reshape(o, [batch_size, -1, OFFSET_DIM]) for o in all_reg], axis=1)
    segments, group_indices, segment_counts, group_indices_all = tf.py_func(
        decode_batch, [
            all_nodes_flat, all_links_flat, all_reg_flat, image_size,
            tf.constant(anchor_sizes)
        ], [tf.float32, tf.int32, tf.int32, tf.int32])
    # TO DO: tf.set_shape(xixing.tj)
    # segments.set_shape([batch_size, offsets, OFFSET_DIM])
    # group_indices.set_shape([batch_size, offsets])
    # segment_counts.set_shape([batch_size])
    # group_indices_all.set_shape([batch_size, offsets])
    return segments, group_indices, segment_counts, group_indices_all


def decode_batch(all_nodes, all_links, all_reg, image_size, anchor_sizes):
    batch_size = all_nodes.shape[0]
    batch_segments = []
    batch_group_indices = []
    batch_segments_counts = []
    batch_group_indices_all = []
    for image_id in range(batch_size):
        image_node_scores = all_nodes[image_id, :, :]
        image_link_scores = all_links[image_id, :, :]
        image_reg = all_reg[image_id, :, :]
        image_segments, image_group_indices, image_segments_counts, image_group_indices_all = decode_image(
            image_node_scores, image_link_scores, image_reg, image_size,
            anchor_sizes)
        # print(image_segments.shape)
        # print(image_group_indices.shape)
        # print(image_segments_counts)
        # print(image_group_indices_all.shape)
        batch_segments.append(image_segments)
        batch_group_indices.append(image_group_indices)
        batch_segments_counts.append(image_segments_counts)
        batch_group_indices_all.append(image_group_indices_all)
    max_count = np.max(batch_segments_counts)
    for image_id in range(batch_size):
        if not batch_segments_counts[image_id] == max_count:
            batch_segments[image_id] = np.vstack(
                (batch_segments[image_id],
                 np.array((max_count - batch_segments_counts[image_id]) 
                          * [(OFFSET_DIM) * [0.0]])))
            batch_group_indices[image_id] = np.hstack(
                (batch_group_indices[image_id],
                 np.array(
                     (max_count - batch_segments_counts[image_id]) * [-1])))
    return np.asarray(batch_segments, np.float32), np.asarray(
        batch_group_indices,
        np.int32), np.asarray(batch_segments_counts,
                              np.int32), np.asarray(batch_group_indices_all,
                                                    np.int32)


def decode_image(image_node_scores, image_link_scores, image_reg, image_size,
                 anchor_sizes):
    map_size = []
    offsets_defaults = []
    offsets_default_node = 0
    offsets_default_link = 0
    for i in range(N_DET_LAYERS):
        offsets_defaults.append([offsets_default_node, offsets_default_link])
        map_size.append(image_size // (2**(2 + i)))
        offsets_default_node += map_size[i][0] * map_size[i][1]
        if i == 0:
            offsets_default_link += map_size[i][0] * map_size[i][
                1] * N_LOCAL_LINKS
        else:
            offsets_default_link += map_size[i][0] * map_size[i][1] * (
                N_LOCAL_LINKS + N_CROSS_LINKS)

    image_group_indices_all = decode_image_by_mutex(image_node_scores,
                                                    image_link_scores,
                                                    FLAGS.node_threshold,
                                                    FLAGS.link_threshold,
                                                    map_size, offsets_defaults)
    image_group_indices_all -= 1
    image_group_indices = image_group_indices_all[np.where(
        image_group_indices_all >= 0)[0]]
    image_segments_counts = len(image_group_indices)
    # convert image_reg to segments with scores(OFFSET_DIM+1)
    image_segments = np.zeros((image_segments_counts, OFFSET_DIM),
                              dtype=np.float32)
    for i, offsets in enumerate(np.where(image_group_indices_all >= 0)[0]):
        encoded_cx = image_reg[offsets, 0]
        encoded_cy = image_reg[offsets, 1]
        encoded_width = image_reg[offsets, 2]
        encoded_height = image_reg[offsets, 3]
        encoded_theta_sin = image_reg[offsets, 4]
        encoded_theta_cos = image_reg[offsets, 5]

        l_idx, x, y = get_coord(offsets, map_size, offsets_defaults)
        rs = anchor_sizes[l_idx]
        eps = 1e-6
        image_segments[i, 0] = encoded_cx * rs + (2**(2 + l_idx)) * (x + 0.5)
        image_segments[i, 1] = encoded_cy * rs + (2**(2 + l_idx)) * (y + 0.5)
        image_segments[i, 2] = np.exp(encoded_width) * rs - eps
        image_segments[i, 3] = np.exp(encoded_height) * rs - eps
        image_segments[i, 4] = encoded_theta_sin
        image_segments[i, 5] = encoded_theta_cos
        # image_segments[i, 6] = image_node_scores[offsets, POS_LABEL]

    return image_segments, image_group_indices, image_segments_counts, image_group_indices_all


def decode_image_by_join(node_scores, link_scores_raw, node_threshold,
                         link_threshold, map_size, offsets_defaults):
    link_scores = get_link8(link_scores_raw, map_size)
    node_mask = node_scores[:, POS_LABEL] >= node_threshold
    link_mask = link_scores[:, POS_LABEL] >= link_threshold
    group_mask = np.zeros_like(node_mask, np.int32) - 1
    offsets_pos = np.where(node_mask == 1)[0]

    # print(len(offsets_pos))

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)

        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(node_mask, dtype=np.int32)
        for i, point in enumerate(offsets_pos):
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    # join by link
    pos_link = 0
    for i, offsets in enumerate(offsets_pos):
        l_idx, x, y = get_coord(offsets, map_size, offsets_defaults)
        neighbours = get_neighbours(l_idx, x, y, map_size, offsets_defaults)
        for n_idx, noffsets in enumerate(neighbours):
            link_value = link_mask[noffsets[1]]
            node_cls = node_mask[noffsets[0]]
            if link_value and node_cls:
                pos_link += 1
                join(offsets, noffsets[0])
    # print(pos_link)
    mask = get_all()
    return mask


def get_link_mask(node_mask, offsets_defaults, link_max):
    link_mask = np.zeros_like(link_max)
    link_mask[0:offsets_defaults[1][1]] = np.tile(
        node_mask[0:offsets_defaults[1][0]],
        (N_LOCAL_LINKS, 1)).transpose().reshape(offsets_defaults[1][1])
    link_mask[offsets_defaults[1][1]:offsets_defaults[2][1]] = np.tile(
        node_mask[offsets_defaults[1][0]:offsets_defaults[2][0]],
        (N_LOCAL_LINKS + N_CROSS_LINKS, 1)).transpose().reshape(
            (offsets_defaults[2][1] - offsets_defaults[1][1]))
    link_mask[offsets_defaults[2][1]:offsets_defaults[3][1]] = np.tile(
        node_mask[offsets_defaults[2][0]:offsets_defaults[3][0]],
        (N_LOCAL_LINKS + N_CROSS_LINKS, 1)).transpose().reshape(
            (offsets_defaults[3][1] - offsets_defaults[2][1]))
    link_mask[offsets_defaults[3][1]:offsets_defaults[4][1]] = np.tile(
        node_mask[offsets_defaults[3][0]:offsets_defaults[4][0]],
        (N_LOCAL_LINKS + N_CROSS_LINKS, 1)).transpose().reshape(
            (offsets_defaults[4][1] - offsets_defaults[3][1]))
    link_mask[offsets_defaults[4][1]:offsets_defaults[5][1]] = np.tile(
        node_mask[offsets_defaults[4][0]:offsets_defaults[5][0]],
        (N_LOCAL_LINKS + N_CROSS_LINKS, 1)).transpose().reshape(
            (offsets_defaults[5][1] - offsets_defaults[4][1]))
    link_mask[offsets_defaults[5][1]:] = np.tile(
        node_mask[offsets_defaults[5][0]:],
        (N_LOCAL_LINKS + N_CROSS_LINKS, 1)).transpose().reshape(
            (len(link_mask) - offsets_defaults[5][1]))

    return link_mask


def get_link8(link_scores_raw, map_size):
    # link[i-1] -local- start -16- end -cross- link[i]
    # print(link_scores_raw.shape)
    link8_mask = np.zeros((link_scores_raw.shape[0]))
    for i in range(N_DET_LAYERS):
        if i == 0:
            offsets_start = map_size[i][0] * map_size[i][1] * N_LOCAL_LINKS
            offsets_end = map_size[i][0] * map_size[i][1] * (
                N_LOCAL_LINKS + 16)
            offsets_link = map_size[i][0] * map_size[i][1] * (
                N_LOCAL_LINKS + 16)
            link8_mask[:offsets_start] = 1
        else:
            offsets_start = offsets_link + map_size[i][0] * map_size[i][
                1] * N_LOCAL_LINKS
            offsets_end = offsets_link + map_size[i][0] * map_size[i][1] * (
                N_LOCAL_LINKS + 16)
            offsets_link_pre = offsets_link
            offsets_link += map_size[i][0] * map_size[i][1] * (
                N_LOCAL_LINKS + 16 + N_CROSS_LINKS)
            link8_mask[offsets_link_pre:offsets_start] = 1
            link8_mask[offsets_end:offsets_link] = 1
    return link_scores_raw[np.where(link8_mask > 0)[0], :]


def decode_image_by_mutex(node_scores, link_scores, node_threshold,
                          link_threshold, map_size, offsets_defaults):
    node_mask = node_scores[:, POS_LABEL] >= node_threshold
    # link_scores = get_link8(link_scores_raw, map_size)
    # print(link_scores.shape)
    link_pos = link_scores[:, POS_LABEL]
    link_mut = link_scores[:, MUT_LABEL]
    link_max = np.max(np.vstack((link_pos, link_mut)), axis=0)
    # link_sum = link_pos + link_mut

    offsets_pos_list = np.where(node_mask == 1)[0].tolist()
    # offsets_link_pos_list = np.where(link_mask == 1)[0].tolist()

    link_mask_th = link_max >= link_threshold
    link_mask = get_link_mask(node_mask, offsets_defaults, link_max)
    offsets_link_max = np.argsort(-(link_max * link_mask 
                                    * link_mask_th))[:len(offsets_pos_list) * 8]

    group_mask = np.zeros_like(node_mask, dtype=np.int32) - 1
    mutex_mask = len(node_mask) * [[]]

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def set_mutex_constraint(point, mutex_point_list):
        mutex_mask[point] = mutex_point_list

    def find_mutex_constraint(point):
        mutex_point_list = mutex_mask[point]
        # update mutex_point_list
        mutex_point_list_new = []
        if not mutex_point_list == []:
            for mutex_point in mutex_point_list:
                if not is_root(mutex_point):
                    mutex_point = find_root(mutex_point)
                if mutex_point not in mutex_point_list_new:
                    mutex_point_list_new.append(mutex_point)
        set_mutex_constraint(point, mutex_point_list_new)
        return mutex_point_list_new

    def combine_mutex_constraint(point, parent):
        mutex_point_list = find_mutex_constraint(point)
        mutex_parent_list = find_mutex_constraint(parent)
        for mutex_point in mutex_point_list:
            if not is_root(mutex_point):
                mutex_point = find_root(mutex_point)
            if mutex_point not in mutex_parent_list:
                mutex_parent_list.append(mutex_point)
        set_mutex_constraint(parent, mutex_parent_list)

    def add_mutex_constraint(p1, p2):
        mutex_point_list1 = find_mutex_constraint(p1)
        mutex_point_list2 = find_mutex_constraint(p2)

        if p1 not in mutex_point_list2:
            mutex_point_list2.append(p1)
        if p2 not in mutex_point_list1:
            mutex_point_list1.append(p2)
        set_mutex_constraint(p1, mutex_point_list1)
        set_mutex_constraint(p2, mutex_point_list2)

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)

        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2 and (root1 not in find_mutex_constraint(root2)):
            set_parent(root1, root2)
            combine_mutex_constraint(root1, root2)

    def disjoin(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            add_mutex_constraint(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(node_mask, dtype=np.int32)
        for _, point in enumerate(offsets_pos_list):
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    # join by link
    pos_link = 0
    mut_link = 0
    for _, offsets_link in enumerate(offsets_link_max):
        l_idx, x, y, link_idx = get_coord_link(offsets_link, map_size,
                                               offsets_defaults)
        offsets = offsets_defaults[l_idx][0] + map_size[l_idx][1] * y + x
        if offsets in offsets_pos_list:
            neighbours = get_neighbours(l_idx, x, y, map_size,
                                        offsets_defaults)
            if not len(np.where(np.array(neighbours)[:,
                                                     2] == link_idx)[0]) == 0:
                noffsets = neighbours[np.where(
                    np.array(neighbours)[:, 2] == link_idx)[0][0]]
                link_pos_value = link_pos[noffsets[1]]
                link_mut_value = link_mut[noffsets[1]]
                node_cls = node_mask[noffsets[0]]
                # node_score = node_scores[noffsets[0], POS_LABEL]
                if node_cls and (link_pos_value > link_mut_value):
                    pos_link += 1
                    join(offsets, noffsets[0])
                elif node_cls and (link_pos_value < link_mut_value):
                    mut_link += 1
                    disjoin(offsets, noffsets[0])

    mask = get_all()
    # print(len(offsets_pos_list))
    # print(pos_link)
    # print(mut_link)
    return mask


oplib = load_oplib(LIB_NAME)

# map C++ operators to python objects
sample_crop_bbox = oplib.sample_crop_bbox
# decode_local_rboxes = oplib.decode_local_rboxes
decode_segments_links = oplib.decode_segments_links
combine_segments = oplib.combine_segments
combine_segments_filter = oplib.combine_segments_filter
clip_rboxes = oplib.clip_rboxes
polygons_to_rboxes = oplib.polygons_to_rboxes
detection_mask = oplib.detection_mask
project_polygons = oplib.project_polygons
