import sys
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class Image2VideoTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_to_video
        self.model_id = 'damo/Image-to-Video'
        self.path = 'https://video-generation-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/baishao/test.jpeg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        pipe = pipeline(task=self.task, model=self.model_id)

        output_video_path = pipe(
            self.path, output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)


if __name__ == '__main__':
    unittest.main()
