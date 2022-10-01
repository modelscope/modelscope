# The implementation is adopted from https://github.com/610265158/Peppa_Pig_Face_Engine

import cv2
import numpy as np
import tensorflow as tf

from .config import config as cfg

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class FaceLandmark:

    def __init__(self, dir):
        self.model_path = dir + '/keypoints.pb'
        self.min_face = 60
        self.keypoint_num = cfg.KEYPOINTS.p_num * 2

        self._graph = tf.Graph()

        with self._graph.as_default():

            self._graph, self._sess = self.init_model(self.model_path)
            self.img_input = tf.get_default_graph().get_tensor_by_name(
                'tower_0/images:0')
            self.embeddings = tf.get_default_graph().get_tensor_by_name(
                'tower_0/prediction:0')
            self.training = tf.get_default_graph().get_tensor_by_name(
                'training_flag:0')

            self.landmark = self.embeddings[:, :self.keypoint_num]
            self.headpose = self.embeddings[:, -7:-4] * 90.
            self.state = tf.nn.sigmoid(self.embeddings[:, -4:])

    def __call__(self, img, bboxes):
        landmark_result = []
        state_result = []
        for i, bbox in enumerate(bboxes):
            landmark, state = self._one_shot_run(img, bbox, i)
            if landmark is not None:
                landmark_result.append(landmark)
                state_result.append(state)
        return np.array(landmark_result), np.array(state_result)

    def simple_run(self, cropped_img):
        with self._graph.as_default():

            cropped_img = np.expand_dims(cropped_img, axis=0)
            landmark, p, states = self._sess.run(
                [self.landmark, self.headpose, self.state],
                feed_dict={
                    self.img_input: cropped_img,
                    self.training: False
                })

        return landmark, states

    def _one_shot_run(self, image, bbox, i):

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if (bbox_width <= self.min_face and bbox_height <= self.min_face):
            return None, None
        add = int(max(bbox_width, bbox_height))
        bimg = cv2.copyMakeBorder(
            image,
            add,
            add,
            add,
            add,
            borderType=cv2.BORDER_CONSTANT,
            value=cfg.DATA.pixel_means)
        bbox += add

        one_edge = (1 + 2 * cfg.KEYPOINTS.base_extend_range[0]) * bbox_width
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        bbox[0] = center[0] - one_edge // 2
        bbox[1] = center[1] - one_edge // 2
        bbox[2] = center[0] + one_edge // 2
        bbox[3] = center[1] + one_edge // 2

        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(
            crop_image,
            (cfg.KEYPOINTS.input_shape[1], cfg.KEYPOINTS.input_shape[0]))
        crop_image = crop_image.astype(np.float32)

        keypoints, state = self.simple_run(crop_image)

        res = keypoints[0][:self.keypoint_num].reshape((-1, 2))
        res[:, 0] = res[:, 0] * w / cfg.KEYPOINTS.input_shape[1]
        res[:, 1] = res[:, 1] * h / cfg.KEYPOINTS.input_shape[0]

        landmark = []
        for _index in range(res.shape[0]):
            x_y = res[_index]
            landmark.append([
                int(x_y[0] * cfg.KEYPOINTS.input_shape[0] + bbox[0] - add),
                int(x_y[1] * cfg.KEYPOINTS.input_shape[1] + bbox[1] - add)
            ])

        landmark = np.array(landmark, np.float32)

        return landmark, state

    def init_model(self, *args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print('Model restred!')
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess
