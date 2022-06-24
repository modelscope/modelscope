import unittest

import datasets as hfdata

from modelscope.models import Model
from modelscope.preprocessors import SequenceClassificationPreprocessor
from modelscope.preprocessors.base import Preprocessor
from modelscope.pydatasets import PyDataset
from modelscope.utils.constant import Hubs
from modelscope.utils.test_utils import require_tf, require_torch, test_level


class ImgPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_field = kwargs.pop('image_path', 'image_path')
        self.width = kwargs.pop('width', 'width')
        self.height = kwargs.pop('height', 'width')

    def __call__(self, data):
        import cv2
        image_path = data.get(self.path_field)
        if not image_path:
            return None
        img = cv2.imread(image_path)
        return {
            'image':
            cv2.resize(img,
                       (data.get(self.height, 128), data.get(self.width, 128)))
        }


class PyDatasetTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_ds_basic(self):
        ms_ds_full = PyDataset.load('squad')
        ms_ds_full_hf = hfdata.load_dataset('squad')
        ms_ds_train = PyDataset.load('squad', split='train')
        ms_ds_train_hf = hfdata.load_dataset('squad', split='train')
        ms_image_train = PyDataset.from_hf_dataset(
            hfdata.load_dataset('beans', split='train'))
        self.assertEqual(ms_ds_full['train'][0], ms_ds_full_hf['train'][0])
        self.assertEqual(ms_ds_full['validation'][0],
                         ms_ds_full_hf['validation'][0])
        self.assertEqual(ms_ds_train[0], ms_ds_train_hf[0])
        print(next(iter(ms_ds_full['train'])))
        print(next(iter(ms_ds_train)))
        print(next(iter(ms_image_train)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_torch
    def test_to_torch_dataset_text(self):
        model_id = 'damo/bert-base-sst2'
        nlp_model = Model.from_pretrained(model_id)
        preprocessor = SequenceClassificationPreprocessor(
            nlp_model.model_dir,
            first_sequence='context',
            second_sequence=None)
        ms_ds_train = PyDataset.load('squad', split='train')
        pt_dataset = ms_ds_train.to_torch_dataset(preprocessors=preprocessor)
        import torch
        dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=5)
        print(next(iter(dataloader)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_tf
    def test_to_tf_dataset_text(self):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        model_id = 'damo/bert-base-sst2'
        nlp_model = Model.from_pretrained(model_id)
        preprocessor = SequenceClassificationPreprocessor(
            nlp_model.model_dir,
            first_sequence='context',
            second_sequence=None)
        ms_ds_train = PyDataset.load('squad', split='train')
        tf_dataset = ms_ds_train.to_tf_dataset(
            batch_size=5,
            shuffle=True,
            preprocessors=preprocessor,
            drop_remainder=True)
        print(next(iter(tf_dataset)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_torch
    def test_to_torch_dataset_img(self):
        ms_image_train = PyDataset.from_hf_dataset(
            hfdata.load_dataset('beans', split='train'))
        pt_dataset = ms_image_train.to_torch_dataset(
            preprocessors=ImgPreprocessor(
                image_path='image_file_path', label='labels'))
        import torch
        dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=5)
        print(next(iter(dataloader)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_tf
    def test_to_tf_dataset_img(self):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        ms_image_train = PyDataset.load('beans', split='train')
        tf_dataset = ms_image_train.to_tf_dataset(
            batch_size=5,
            shuffle=True,
            preprocessors=ImgPreprocessor(image_path='image_file_path'),
            drop_remainder=True,
            label_cols='labels')
        print(next(iter(tf_dataset)))


if __name__ == '__main__':
    unittest.main()
