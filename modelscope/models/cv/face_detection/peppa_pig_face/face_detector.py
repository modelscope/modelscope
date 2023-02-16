# The implementation here is modified based on InsightFace_Pytorch, originally Apache License and publicly available
# at https://github.com/610265158/Peppa_Pig_Face_Engine
import cv2
import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class FaceDetector:

    def __init__(self, dir):

        self.model_path = dir + '/detector.pb'
        self.thres = 0.8
        self.input_shape = (512, 512, 3)
        self.pixel_means = np.array([123., 116., 103.])

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._graph, self._sess = self.init_model(self.model_path)

            self.input_image = tf.get_default_graph().get_tensor_by_name(
                'tower_0/images:0')
            self.training = tf.get_default_graph().get_tensor_by_name(
                'training_flag:0')
            self.output_ops = [
                tf.get_default_graph().get_tensor_by_name('tower_0/boxes:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/scores:0'),
                tf.get_default_graph().get_tensor_by_name(
                    'tower_0/num_detections:0'),
            ]

    def __call__(self, image):

        image, scale_x, scale_y = self.preprocess(
            image,
            target_width=self.input_shape[1],
            target_height=self.input_shape[0])

        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self._sess.run(
            self.output_ops,
            feed_dict={
                self.input_image: image,
                self.training: False
            })

        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]

        scores = scores[0][:num_boxes]

        to_keep = scores > self.thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        y1 = self.input_shape[0] / scale_y
        x1 = self.input_shape[1] / scale_x
        y2 = self.input_shape[0] / scale_y
        x2 = self.input_shape[1] / scale_x
        scaler = np.array([y1, x1, y2, x2], dtype='float32')
        boxes = boxes * scaler

        scores = np.expand_dims(scores, 0).reshape([-1, 1])

        for i in range(boxes.shape[0]):
            boxes[i] = np.array(
                [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        return np.concatenate([boxes, scores], axis=1)

    def preprocess(self, image, target_height, target_width, label=None):

        h, w, c = image.shape

        bimage = np.zeros(
            shape=[target_height, target_width, c],
            dtype=image.dtype) + np.array(
                self.pixel_means, dtype=image.dtype)
        long_side = max(h, w)

        scale_x = scale_y = target_height / long_side

        image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

        h_, w_, _ = image.shape
        bimage[:h_, :w_, :] = image

        return bimage, scale_x, scale_y

    def init_model(self, *args):
        pb_path = args[0]

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

            return (compute_graph, sess)

        model = init_pb(pb_path)

        graph = model[0]
        sess = model[1]

        return graph, sess
