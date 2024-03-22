import sys
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Video2VideoTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_to_video
        self.model_id = 'damo/Video-to-Video'
        self.path = 'https://video-generation-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/baishao/test.mp4'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        pipe = pipeline(task=self.task, model=self.model_id)
        p_input = {
            'video_path': self.path,
            'text': 'A panda is surfing on the sea'
        }

        output_video_path = pipe(
            p_input, output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)


if __name__ == '__main__':
    unittest.main()
