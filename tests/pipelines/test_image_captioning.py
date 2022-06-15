# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile
import unittest

from modelscope.fileio import File
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ImageCaptionTest(unittest.TestCase):

    @unittest.skip('skip before model is restored in model hub')
    def test_run(self):
        model = 'https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_large_best_clean.pt'

        os.system(
            'wget  https://jirenmr.oss-cn-zhangjiakou.aliyuncs.com/ofa/BPE.zip'
        )
        os.system('unzip BPE.zip')
        bpe_dir = './BPE'

        with tempfile.NamedTemporaryFile('wb', suffix='.pb') as ofile:
            ofile.write(File.read(model))
            img_captioning = pipeline(
                Tasks.image_captioning, model=ofile.name, bpe_dir=bpe_dir)

            result = img_captioning(
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
            )
            print(result['caption'])


if __name__ == '__main__':
    unittest.main()
