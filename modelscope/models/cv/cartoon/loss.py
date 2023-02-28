'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''

import os.path as osp

import numpy as np
import scipy.stats as st
import tensorflow as tf
from joblib import Parallel, delayed
from skimage import color, segmentation

from .network import disc_sn

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:

    def __init__(self, vgg19_npy_path=None):

        self.data_dict = np.load(
            vgg19_npy_path, encoding='latin1', allow_pickle=True).item()

        print('Finished loading vgg19.npy')

    def build_conv4_4(self, rgb, include_fc=False):

        rgb_scaled = (rgb + 1) * 127.5

        blue, green, red = tf.split(
            axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(
            axis=3,
            values=[
                blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]
            ])

        self.conv1_1 = self.conv_layer(bgr, 'conv1_1')
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, 'conv1_2')
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, 'conv2_2')
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, 'conv3_2')
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, 'conv3_3')
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, 'conv3_4')
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, 'conv4_2')
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, 'conv4_3')
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, 'conv4_4')
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='filter')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='weights')


def content_loss(model_dir, input_photo, transfer_res, input_superpixel):
    vgg_model = Vgg19(osp.join(model_dir, 'vgg19.npy'))
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(transfer_res)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]
    abs_photo = tf.losses.absolute_difference(vgg_photo, vgg_output)
    photo_loss = tf.reduce_mean(abs_photo) / (h * w * c)
    abs_superpixel = tf.losses.absolute_difference(vgg_superpixel, vgg_output)
    superpixel_loss = tf.reduce_mean(abs_superpixel) / (h * w * c)
    loss = photo_loss + superpixel_loss

    return loss


def style_loss(input_cartoon, output_cartoon):
    blur_fake = guided_filter(output_cartoon, output_cartoon, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = color_shift(output_cartoon, input_cartoon)

    d_loss_gray, g_loss_gray = lsgan_loss(
        disc_sn,
        gray_cartoon,
        gray_fake,
        scale=1,
        patch=True,
        name='disc_gray')
    d_loss_blur, g_loss_blur = lsgan_loss(
        disc_sn,
        blur_cartoon,
        blur_fake,
        scale=1,
        patch=True,
        name='disc_blur')
    sty_g_loss = (g_loss_blur) + g_loss_gray
    sty_d_loss = d_loss_blur + d_loss_gray

    return sty_g_loss, sty_d_loss


def gan_loss(discriminator,
             real,
             fake,
             scale=1,
             channel=32,
             patch=False,
             name='discriminator'):

    real_logit = discriminator(
        real, scale, channel, name=name, patch=patch, reuse=False)
    fake_logit = discriminator(
        fake, scale, channel, name=name, patch=patch, reuse=True)

    real_logit = tf.nn.sigmoid(real_logit)
    fake_logit = tf.nn.sigmoid(fake_logit)

    g_loss_blur = -tf.reduce_mean(tf.log(fake_logit))
    d_loss_blur = -tf.reduce_mean(tf.log(real_logit) + tf.log(1. - fake_logit))

    return d_loss_blur, g_loss_blur


def lsgan_loss(discriminator,
               real,
               fake,
               scale=1,
               channel=32,
               patch=False,
               name='discriminator'):

    real_logit = discriminator(
        real, scale, channel, name=name, patch=patch, reuse=False)
    fake_logit = discriminator(
        fake, scale, channel, name=name, patch=patch, reuse=True)

    g_loss = tf.reduce_mean((fake_logit - 1)**2)
    d_loss = 0.5 * (
        tf.reduce_mean((real_logit - 1)**2) + tf.reduce_mean(fake_logit**2))

    return d_loss, g_loss


def total_variation_loss(image, k_size=1):
    h, w = image.get_shape().as_list()[1:3]
    tv_h = tf.reduce_mean(
        (image[:, k_size:, :, :] - image[:, :h - k_size, :, :])**2)
    tv_w = tf.reduce_mean(
        (image[:, :, k_size:, :] - image[:, :, :w - k_size, :])**2)
    tv_loss = (tv_h + tv_w) / (3 * h * w)
    return tv_loss


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    N = tf_box_filter(
        tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = tf.split(image1, num_or_size_splits=3, axis=3)
    b2, g2, r2 = tf.split(image2, num_or_size_splits=3, axis=3)
    if mode == 'normal':
        b_weight = tf.random.normal(shape=[1], mean=0.114, stddev=0.1)
        g_weight = np.random.normal(shape=[1], mean=0.587, stddev=0.1)
        r_weight = np.random.normal(shape=[1], mean=0.299, stddev=0.1)
    elif mode == 'uniform':
        b_weight = tf.random.uniform(shape=[1], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[1], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[1], minval=0.199, maxval=0.399)
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (
        b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (
        b_weight + g_weight + r_weight)
    return output1, output2


def simple_superpixel(batch_image, seg_num=200):

    def process_slic(image):
        seg_label = segmentation.slic(
            image,
            n_segments=seg_num,
            sigma=1,
            compactness=10,
            convert2lab=True,
            start_label=1)
        image = color.label2rgb(seg_label, image, kind='avg', bg_label=0)
        return image

    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(
        delayed(process_slic)(image) for image in batch_image)
    return np.array(batch_out)


def tf_box_filter(x, r):
    ch = x.get_shape().as_list()[-1]
    weight = 1 / ((2 * r + 1)**2)
    box_kernel = weight * np.ones((2 * r + 1, 2 * r + 1, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


if __name__ == '__main__':
    pass
