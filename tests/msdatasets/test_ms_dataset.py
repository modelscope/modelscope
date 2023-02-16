# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.audio.asr_dataset import ASRDataset
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.preprocessors.base import Preprocessor
from modelscope.utils.constant import DEFAULT_DATASET_NAMESPACE, DownloadMode
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


class MsDatasetTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_movie_scene_seg_toydata(self):
        ms_ds_train = MsDataset.load('movie_scene_seg_toydata', split='train')
        print(ms_ds_train._hf_ds.config_kwargs)
        assert next(iter(ms_ds_train.config_kwargs['split_config'].values()))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_coco(self):
        ms_ds_train = MsDataset.load(
            'pets_small',
            namespace=DEFAULT_DATASET_NAMESPACE,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            split='train')
        print(ms_ds_train.config_kwargs)
        assert next(iter(ms_ds_train.config_kwargs['split_config'].values()))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ms_csv_basic(self):
        ms_ds_train = MsDataset.load(
            'clue', subset_name='afqmc',
            split='train').to_hf_dataset().select(range(5))
        print(next(iter(ms_ds_train)))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ds_basic(self):
        ms_ds_full = MsDataset.load(
            'xcopa', subset_name='translation-et', namespace='damotest')
        ms_ds = MsDataset.load(
            'xcopa',
            subset_name='translation-et',
            namespace='damotest',
            split='test')
        print(next(iter(ms_ds_full['test'])))
        print(next(iter(ms_ds)))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    @require_torch
    def test_to_torch_dataset_text(self):
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
        nlp_model = Model.from_pretrained(model_id)
        preprocessor = TextClassificationTransformersPreprocessor(
            nlp_model.model_dir,
            first_sequence='premise',
            second_sequence=None,
            padding='max_length')
        ms_ds_train = MsDataset.load(
            'xcopa',
            subset_name='translation-et',
            namespace='damotest',
            split='test')
        pt_dataset = ms_ds_train.to_torch_dataset(preprocessors=preprocessor)
        import torch
        dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=5)
        print(next(iter(dataloader)))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    @require_tf
    def test_to_tf_dataset_text(self):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
        nlp_model = Model.from_pretrained(model_id)
        preprocessor = TextClassificationTransformersPreprocessor(
            nlp_model.model_dir,
            first_sequence='premise',
            second_sequence=None)
        ms_ds_train = MsDataset.load(
            'xcopa',
            subset_name='translation-et',
            namespace='damotest',
            split='test')
        tf_dataset = ms_ds_train.to_tf_dataset(
            batch_size=5,
            shuffle=True,
            preprocessors=preprocessor,
            drop_remainder=True)
        print(next(iter(tf_dataset)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_to_dataset_asr(self):
        ms_ds_asr = ASRDataset.load(
            'speech_asr_aishell1_trainsets', namespace='speech_asr')
        print(next(iter(ms_ds_asr['train'])))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_torch
    def test_to_torch_dataset_img(self):
        ms_image_train = MsDataset.load(
            'fixtures_image_utils', namespace='damotest', split='test')
        pt_dataset = ms_image_train.to_torch_dataset(
            preprocessors=ImgPreprocessor(image_path='file'))
        import torch
        dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=5)
        print(next(iter(dataloader)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    @require_tf
    def test_to_tf_dataset_img(self):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        ms_image_train = MsDataset.load(
            'fixtures_image_utils', namespace='damotest', split='test')
        tf_dataset = ms_image_train.to_tf_dataset(
            batch_size=5,
            shuffle=True,
            preprocessors=ImgPreprocessor(image_path='file'),
            drop_remainder=True,
        )
        print(next(iter(tf_dataset)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_streaming_load_coco(self):
        small_coco_for_test = MsDataset.load(
            dataset_name='EasyCV/small_coco_for_test',
            split='train',
            use_streaming=True,
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        dataset_sample_dict = next(iter(small_coco_for_test))
        print(dataset_sample_dict)
        assert dataset_sample_dict.values()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_streaming_load_uni_fold(self):
        """Test case for loading large scale datasets."""
        dataset = MsDataset.load(
            dataset_name='Uni-Fold-Data',
            split='train',
            use_streaming=True,
            namespace='DPTech')
        data_example = next(iter(dataset))
        print(data_example)
        assert data_example.values()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_streaming_load_afqmc(self):
        """To streaming-load afqmc dataset, which contains train/dev/validation data in meta-files."""
        dataset = MsDataset.load('afqmc', split='test', use_streaming=True)
        data_example = next(iter(dataset))
        print(data_example)
        assert data_example.values()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_streaming_load_from_hf(self):
        """Use stream mode to load dataset from huggingface hub."""
        from modelscope.utils.constant import Hubs
        ds_train = MsDataset.load(
            'glue',
            subset_name='sst2',
            split='train',
            hub=Hubs.huggingface,
            use_streaming=True)
        data_example = next(iter(ds_train))
        print(data_example)
        assert data_example.values()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_streaming_load_img_object(self):
        """Test case for iterating PIL object."""
        from PIL.PngImagePlugin import PngImageFile
        dataset = MsDataset.load(
            dataset_name='SIDD',
            subset_name='default',
            namespace='huizheng',
            split='train',
            use_streaming=True)
        data_example = next(iter(dataset))
        print(data_example)
        assert isinstance(data_example['Noisy Image:FILE:Object'],
                          PngImageFile)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_to_ms_dataset(self):
        """Test case for converting huggingface dataset to `MsDataset` instance."""
        from datasets.load import load_dataset
        hf_dataset = load_dataset('beans', split='train', streaming=True)
        ms_dataset = MsDataset.to_ms_dataset(hf_dataset)
        data_example = next(iter(ms_dataset))
        print(data_example)
        assert data_example.values()


if __name__ == '__main__':
    unittest.main()
