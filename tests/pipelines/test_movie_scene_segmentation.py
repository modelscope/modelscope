# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class MovieSceneSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.movie_scene_segmentation
        self.model_id = 'damo/cv_resnet50-bert_video-scene-segmentation_movienet'

        cache_path = snapshot_download(self.model_id)
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)

        self.tmp_dir = tempfile.TemporaryDirectory().name

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_movie_scene_segmentation(self):
        input_location = 'data/test/videos/movie_scene_segmentation_test_video.mp4'
        movie_scene_segmentation_pipeline = pipeline(
            Tasks.movie_scene_segmentation, model=self.model_id)
        result = movie_scene_segmentation_pipeline(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_movie_scene_segmentation_finetune(self):

        train_data_cfg = ConfigDict(
            name='movie_scene_seg_toydata',
            split='train',
            cfg=self.cfg.preprocessor,
            test_mode=False)

        train_dataset = MsDataset.load(
            dataset_name=train_data_cfg.name,
            split=train_data_cfg.split,
            cfg=train_data_cfg.cfg,
            test_mode=train_data_cfg.test_mode)

        test_data_cfg = ConfigDict(
            name='movie_scene_seg_toydata',
            split='test',
            cfg=self.cfg.preprocessor,
            test_mode=True)

        test_dataset = MsDataset.load(
            dataset_name=test_data_cfg.name,
            split=test_data_cfg.split,
            cfg=test_data_cfg.cfg,
            test_mode=test_data_cfg.test_mode)

        kwargs = dict(
            model=self.model_id,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.movie_scene_segmentation, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(trainer.work_dir)
        print(results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_movie_scene_segmentation_finetune_with_custom_dataset(self):

        data_cfg = ConfigDict(
            dataset_name='movie_scene_seg_toydata',
            namespace='modelscope',
            train_split='train',
            test_split='test',
            model_cfg=self.cfg)

        train_dataset = MsDataset.load(
            dataset_name=data_cfg.dataset_name,
            namespace=data_cfg.namespace,
            split=data_cfg.train_split,
            custom_cfg=data_cfg.model_cfg,
            test_mode=False)

        test_dataset = MsDataset.load(
            dataset_name=data_cfg.dataset_name,
            namespace=data_cfg.namespace,
            split=data_cfg.test_split,
            custom_cfg=data_cfg.model_cfg,
            test_mode=True)

        kwargs = dict(
            model=self.model_id,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.movie_scene_segmentation, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(trainer.work_dir)
        print(results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_movie_scene_segmentation_with_default_task(self):
        input_location = 'data/test/videos/movie_scene_segmentation_test_video.mp4'
        movie_scene_segmentation_pipeline = pipeline(
            Tasks.movie_scene_segmentation)
        result = movie_scene_segmentation_pipeline(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
