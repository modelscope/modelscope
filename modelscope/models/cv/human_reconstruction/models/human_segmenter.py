# The implementation is also open-sourced by the authors, and available at
# https://www.modelscope.cn/models/damo/cv_unet_image-matting/summary
import cv2
import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class human_segmenter(object):

    def __init__(self, model_path):
        super(human_segmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用GPU 30%的显存
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        self.image_node = self.sess.graph.get_tensor_by_name('input_image:0')
        self.output_node = self.sess.graph.get_tensor_by_name('output_png:0')
        self.logits_node = self.sess.graph.get_tensor_by_name('if_person:0')
        print('human_segmenter init done')

    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.astype(float)
        return img

    def run(self, img):
        image_feed = self.image_preprocess(img)
        output_img_value, logits_value = self.sess.run(
            [self.output_node, self.logits_node],
            feed_dict={self.image_node: image_feed})
        mask = output_img_value[:, :, -1]
        return mask

    def get_human_bbox(self, mask):
        print('dtype:{}, max:{},shape:{}'.format(mask.dtype, np.max(mask),
                                                 mask.shape))
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contoursArea = [cv2.contourArea(c) for c in contours]
        max_area_index = contoursArea.index(max(contoursArea))
        bbox = cv2.boundingRect(contours[max_area_index])
        return bbox

    def release(self):
        self.sess.close()
