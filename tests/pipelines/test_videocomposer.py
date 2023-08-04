# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class VideoDeinterlaceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_video_synthesis
        self.model_id = 'buptwq/videocomposer'
        self.model_revision = 'v1.0.1'
        self.dataset_id = 'buptwq/videocomposer-depths-style'
        self.text = 'A glittering and translucent fish swimming in a \
                     small glass bowl with multicolored piece of stone, like a glass fish'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_pipeline(self):
        pipe = pipeline(
            task=Tasks.text_to_video_synthesis,
            model=self.model_id,
            model_revision=self.model_revision)
        ds = MsDataset.load(
            self.dataset_id,
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        inputs = next(iter(ds))
        inputs.update({'text': self.text})
        output = pipe(inputs)


if __name__ == '__main__':
    unittest.main()
