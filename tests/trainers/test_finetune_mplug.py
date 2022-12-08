# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.multi_modal import MPlugForAllTasks
from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestFinetuneMPlug(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        datadict = MsDataset.load('coco_captions_small_slice')
        self.train_dataset = MsDataset(
            datadict['train'].remap_columns({
                'image:FILE': 'image',
                'answer:Value': 'answer'
            }).map(lambda _: {'question': 'what the picture describes?'}))
        self.test_dataset = MsDataset(
            datadict['test'].remap_columns({
                'image:FILE': 'image',
                'answer:Value': 'answer'
            }).map(lambda _: {'question': 'what the picture describes?'}))
        self.max_epochs = 2

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_caption(self):
        kwargs = dict(
            model='damo/mplug_backbone_base_en',
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir,
            task=Tasks.image_captioning)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_caption_with_model_and_args(self):
        cache_path = snapshot_download('damo/mplug_backbone_base_en')
        model = MPlugForAllTasks.from_pretrained(
            cache_path, task=Tasks.image_captioning)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_vqa(self):
        kwargs = dict(
            model='damo/mplug_backbone_base_en',
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir,
            task=Tasks.visual_question_answering)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_vqa_with_model_and_args(self):
        cache_path = snapshot_download(
            'damo/mplug_visual-question-answering_coco_large_en')
        model = MPlugForAllTasks.from_pretrained(
            cache_path, task=Tasks.visual_question_answering)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_retrieval(self):
        kwargs = dict(
            model='damo/mplug_backbone_base_en',
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir,
            task=Tasks.image_text_retrieval)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_retrieval_with_model_and_args(self):
        cache_path = snapshot_download('damo/mplug_backbone_base_en')
        model = MPlugForAllTasks.from_pretrained(
            cache_path, task=Tasks.image_text_retrieval)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.mplug, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
