# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FaceEmotionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model = 'damo/cv_face-emotion'
        self.img = 'data/test/images/face_emotion.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        print(result)

    @unittest.skip('skip since the model is set to private for now')
    def test_run_modelhub(self):
        face_emotion = pipeline(Tasks.face_emotion, model=self.model)
        self.pipeline_inference(face_emotion, self.img)

    @unittest.skip('skip since the model is set to private for now')
    def test_run_modelhub_default_model(self):
        face_emotion = pipeline(Tasks.face_emotion)
        self.pipeline_inference(face_emotion, self.img)


if __name__ == '__main__':
    unittest.main()
