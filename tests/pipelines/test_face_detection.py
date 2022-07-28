# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np

from modelscope.fileio import File
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class FaceDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'

    def show_result(self, img_path, bboxes, kpss, scores):
        bboxes = np.array(bboxes)
        kpss = np.array(kpss)
        scores = np.array(scores)
        img = cv2.imread(img_path)
        assert img is not None, f"Can't read img: {img_path}"
        for i in range(len(scores)):
            bbox = bboxes[i].astype(np.int32)
            kps = kpss[i].reshape(-1, 2).astype(np.int32)
            score = scores[i]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for kp in kps:
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 1)
            cv2.putText(
                img,
                f'{score:.2f}', (x1, y2),
                1,
                1.0, (0, 255, 0),
                thickness=1,
                lineType=8)
        cv2.imwrite('result.png', img)
        print(
            f'Found {len(scores)} faces, output written to {osp.abspath("result.png")}'
        )

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        input_location = ['data/test/images/face_detection.png']
        # alternatively:
        # input_location = '/dir/to/images'

        dataset = MsDataset.load(input_location, target='image')
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        # note that for dataset output, the inference-output is a Generator that can be iterated.
        result = face_detection(dataset)
        result = next(result)
        self.show_result(input_location[0], result[OutputKeys.BOXES],
                         result[OutputKeys.KEYPOINTS],
                         result[OutputKeys.SCORES])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        img_path = 'data/test/images/face_detection.png'

        result = face_detection(img_path)
        self.show_result(img_path, result[OutputKeys.BOXES],
                         result[OutputKeys.KEYPOINTS],
                         result[OutputKeys.SCORES])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        face_detection = pipeline(Tasks.face_detection)
        img_path = 'data/test/images/face_detection.png'
        result = face_detection(img_path)
        self.show_result(img_path, result[OutputKeys.BOXES],
                         result[OutputKeys.KEYPOINTS],
                         result[OutputKeys.SCORES])


if __name__ == '__main__':
    unittest.main()
