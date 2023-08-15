import sys
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class VideoDeinterlaceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_to_video_task
        self.model_id = '/mnt/workspace/Video_Generation/creative_space_image_to_video/models'
        # self.model_revision = 'v1.0.1'
        # self.dataset_id = 'buptwq/videocomposer-depths-style'
        self.path = '/mnt/workspace/Video_Generation/creative_space_image_to_video/test.jpeg'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_pipeline(self):
        pipe = pipeline(task=self.task, model=self.model_id)
        # ds = MsDataset.load(
        #     self.dataset_id,
        #     split='train',
        #     download_mode=DownloadMode.FORCE_REDOWNLOAD)
        # inputs = next(iter(ds))
        # inputs.update({'text': self.text})
        
        inputs = {'img_path': self.path}
        # _ = pipe(inputs)
        
        output_video_path = pipe(inputs, output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)


if __name__ == '__main__':
    unittest.main()