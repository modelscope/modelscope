# Part of the implementation is borrowed and modified from SegLink,
# publicly available at https://github.com/bgshih/seglink
import tensorflow as tf

from . import ops, resnet18_v1, resnet_utils

if tf.__version__ >= '2.0':
    import tf_slim as slim
else:
    from tensorflow.contrib import slim

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

# constants
OFFSET_DIM = 6

N_LOCAL_LINKS = 8
N_CROSS_LINKS = 4
N_SEG_CLASSES = 2
N_LNK_CLASSES = 4

POS_LABEL = 1
NEG_LABEL = 0


class SegLinkDetector():

    def __init__(self):
        self.anchor_sizes = [6., 11.84210526, 23.68421053, 45., 90., 150.]

    def _detection_classifier(self,
                              maps,
                              ksize,
                              weight_decay,
                              cross_links=False,
                              scope=None):

        with tf.variable_scope(scope):
            seg_depth = N_SEG_CLASSES
            if cross_links:
                lnk_depth = N_LNK_CLASSES * (N_LOCAL_LINKS + N_CROSS_LINKS)
            else:
                lnk_depth = N_LNK_CLASSES * N_LOCAL_LINKS
            reg_depth = OFFSET_DIM
            map_depth = maps.get_shape()[3]
            inter_maps, inter_relu = ops.conv2d(
                maps, map_depth, 256, 1, 1, 'SAME', scope='conv_inter')

            dir_maps, dir_relu = ops.conv2d(
                inter_relu, 256, 2, ksize, 1, 'SAME', scope='conv_dir')
            cen_maps, cen_relu = ops.conv2d(
                inter_relu, 256, 2, ksize, 1, 'SAME', scope='conv_cen')
            pol_maps, pol_relu = ops.conv2d(
                inter_relu, 256, 8, ksize, 1, 'SAME', scope='conv_pol')
            concat_relu = tf.concat([dir_relu, cen_relu, pol_relu], axis=-1)
            _, lnk_embedding = ops.conv_relu(
                concat_relu, 12, 256, 1, 1, scope='lnk_embedding')
            lnk_maps, lnk_relu = ops.conv2d(
                inter_relu + lnk_embedding,
                256,
                lnk_depth,
                ksize,
                1,
                'SAME',
                scope='conv_lnk')

            char_seg_maps, char_seg_relu = ops.conv2d(
                inter_relu,
                256,
                seg_depth,
                ksize,
                1,
                'SAME',
                scope='conv_char_cls')
            char_reg_maps, char_reg_relu = ops.conv2d(
                inter_relu,
                256,
                reg_depth,
                ksize,
                1,
                'SAME',
                scope='conv_char_reg')
            concat_char_relu = tf.concat([char_seg_relu, char_reg_relu],
                                         axis=-1)
            _, char_embedding = ops.conv_relu(
                concat_char_relu, 8, 256, 1, 1, scope='conv_char_embedding')
            seg_maps, seg_relu = ops.conv2d(
                inter_relu + char_embedding,
                256,
                seg_depth,
                ksize,
                1,
                'SAME',
                scope='conv_cls')
            reg_maps, reg_relu = ops.conv2d(
                inter_relu + char_embedding,
                256,
                reg_depth,
                ksize,
                1,
                'SAME',
                scope='conv_reg')

        return seg_relu, lnk_relu, reg_relu

    def _build_cnn(self, images, weight_decay, is_training):
        with slim.arg_scope(
                resnet18_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet18_v1.resnet_v1_18(
                images, is_training=is_training, scope='resnet_v1_18')

        outputs = {
            'conv3_3': end_points['pool1'],
            'conv4_3': end_points['pool2'],
            'fc7': end_points['pool3'],
            'conv8_2': end_points['pool4'],
            'conv9_2': end_points['pool5'],
            'conv10_2': end_points['pool6'],
        }
        return outputs

    def build_model(self, images, is_training=True, scope=None):

        weight_decay = 5e-4  # FLAGS.weight_decay
        cnn_outputs = self._build_cnn(images, weight_decay, is_training)
        det_0 = self._detection_classifier(
            cnn_outputs['conv3_3'],
            3,
            weight_decay,
            cross_links=False,
            scope='dete_0')
        det_1 = self._detection_classifier(
            cnn_outputs['conv4_3'],
            3,
            weight_decay,
            cross_links=True,
            scope='dete_1')
        det_2 = self._detection_classifier(
            cnn_outputs['fc7'],
            3,
            weight_decay,
            cross_links=True,
            scope='dete_2')
        det_3 = self._detection_classifier(
            cnn_outputs['conv8_2'],
            3,
            weight_decay,
            cross_links=True,
            scope='dete_3')
        det_4 = self._detection_classifier(
            cnn_outputs['conv9_2'],
            3,
            weight_decay,
            cross_links=True,
            scope='dete_4')
        det_5 = self._detection_classifier(
            cnn_outputs['conv10_2'],
            3,
            weight_decay,
            cross_links=True,
            scope='dete_5')
        outputs = [det_0, det_1, det_2, det_3, det_4, det_5]
        return outputs
