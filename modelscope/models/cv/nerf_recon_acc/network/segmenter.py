# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf


class ObjectSegmenter(object):
    """use ObjectSegmenter to segment object from input video.

    Args:
        model_path (str): the segment model path.
    """

    def __init__(self, model_path):
        super(ObjectSegmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        self.image_node = self.sess.graph.get_tensor_by_name('input_image:0')
        self.output_node = self.sess.graph.get_tensor_by_name('output_png:0')
        self.logits_node = self.sess.graph.get_tensor_by_name('if_person:0')

    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(float)
        return img

    def run_mask(self, img):
        image_feed = self.image_preprocess(img)
        output_img_value, logits_value = self.sess.run(
            [self.output_node, self.logits_node],
            feed_dict={self.image_node: image_feed})
        mask = output_img_value[:, :, 3:]
        return mask
