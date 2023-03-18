# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from datasets import Dataset

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets.dataset_cls.custom_datasets import \
    TorchCustomDataset
from modelscope.preprocessors import Preprocessor
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils import logger as logging
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys, ModelFile, Tasks
from modelscope.utils.test_utils import test_level

logger = logging.get_logger()


class TestDummyEpochBasedTrainer(EpochBasedTrainer):

    def __init__(self,
                 dataset: Dataset = None,
                 mode: str = ModeKeys.TRAIN,
                 preprocessor: Preprocessor = None,
                 **kwargs):
        super(TestDummyEpochBasedTrainer, self).__init__(**kwargs)
        self.train_dataset = self.to_task_dataset(dataset, mode, preprocessor)

    def to_task_dataset(self, dataset: Dataset, mode: str,
                        preprocessor: Preprocessor,
                        **kwargs) -> TorchCustomDataset:
        src_dataset_dict = {
            'src_txt': [
                'This is test sentence1-1', 'This is test sentence2-1',
                'This is test sentence3-1'
            ]
        }
        dataset = Dataset.from_dict(src_dataset_dict)
        dataset_res = TorchCustomDataset(
            datasets=dataset, mode=mode, preprocessor=preprocessor)
        dataset_res.trainer = self
        return dataset_res


class TestCustomDatasetsCompatibility(unittest.TestCase):

    def setUp(self):
        self.task = Tasks.movie_scene_segmentation
        self.model_id = 'damo/cv_resnet50-bert_video-scene-segmentation_movienet'

        cache_path = snapshot_download(self.model_id)
        self.config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(self.config_path)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_adaseq_import_task_datasets(self):
        from modelscope.msdatasets.task_datasets.torch_base_dataset import TorchTaskDataset
        from modelscope.msdatasets.task_datasets import GoproImageDeblurringDataset
        from modelscope.msdatasets.task_datasets import RedsImageDeblurringDataset
        from modelscope.msdatasets.task_datasets import SiddImageDenoisingDataset
        from modelscope.msdatasets.task_datasets import VideoSummarizationDataset

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_adaseq_trainer_overwrite(self):
        test_trainer = TestDummyEpochBasedTrainer(cfg_file=self.config_path)

        assert isinstance(test_trainer.train_dataset.trainer,
                          TestDummyEpochBasedTrainer)
        assert test_trainer.train_dataset.mode == ModeKeys.TRAIN
        assert isinstance(test_trainer.train_dataset._inner_dataset, Dataset)


if __name__ == '__main__':
    unittest.main()
