# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import tempfile
import unittest

import cv2
from ali_maas_datasets import PyDataset

from maas_lib.fileio import File
from maas_lib.pipelines import pipeline
from maas_lib.utils.constant import Tasks


class ImageMattingTest(unittest.TestCase):

    def test_run(self):
        model_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs' \
                     '.com/data/test/maas/image_matting/matting_person.pb'
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = osp.join(tmp_dir, 'matting_person.pb')
            with open(model_file, 'wb') as ofile:
                ofile.write(File.read(model_path))
            img_matting = pipeline(Tasks.image_matting, model=tmp_dir)

            result = img_matting(
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
            )
            cv2.imwrite('result.png', result['output_png'])

    def test_run_with_dataset(self):
        input_location = [
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
        ]
        # alternatively:
        # input_location = '/dir/to/images'

        dataset = PyDataset.load(input_location, target='image')
        img_matting = pipeline(
            Tasks.image_matting, model='damo/image-matting-person')
        # note that for dataset output, the inference-output is a Generator that can be iterated.
        result = img_matting(dataset)
        cv2.imwrite('result.png', next(result)['output_png'])
        print(f'Output written to {osp.abspath("result.png")}')

    def test_run_modelhub(self):
        img_matting = pipeline(
            Tasks.image_matting, model='damo/image-matting-person')

        result = img_matting(
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
        )
        cv2.imwrite('result.png', result['output_png'])
        print(f'Output written to {osp.abspath("result.png")}')


if __name__ == '__main__':
    unittest.main()
