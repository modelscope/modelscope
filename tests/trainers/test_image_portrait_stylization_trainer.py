# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import cv2

from modelscope.exporters.cv import CartoonTranslationExporter
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.trainers.cv import CartoonTranslationTrainer
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TestImagePortraitStylizationTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_portrait_stylization
        self.test_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        model_id = 'damo/cv_unet_person-image-cartoon_compound-models'

        data_dir = MsDataset.load(
            'dctnet_train_clipart_mini_ms',
            namespace='menyifang',
            split='train').config_kwargs['split_config']['train']

        data_photo = os.path.join(data_dir, 'face_photo')
        data_cartoon = os.path.join(data_dir, 'face_cartoon')
        work_dir = 'exp_localtoon'
        max_steps = 10
        trainer = CartoonTranslationTrainer(
            model=model_id,
            work_dir=work_dir,
            photo=data_photo,
            cartoon=data_cartoon,
            max_steps=max_steps)
        trainer.train()

        ckpt_path = os.path.join(work_dir, 'saved_models', 'model-' + str(0))
        pb_path = os.path.join(trainer.model_dir, 'cartoon_h.pb')
        exporter = CartoonTranslationExporter()
        exporter.export_frozen_graph_def(
            ckpt_path=ckpt_path, frozen_graph_path=pb_path)

        self.pipeline_person_image_cartoon(trainer.model_dir)

    def pipeline_person_image_cartoon(self, model_dir):
        pipeline_cartoon = pipeline(task=self.task, model=model_dir)
        result = pipeline_cartoon(input=self.test_image)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {os.path.abspath("result.png")}')


if __name__ == '__main__':
    unittest.main()
