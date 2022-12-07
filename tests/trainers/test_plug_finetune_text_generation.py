# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


def test_trainer_with_model_and_args():

    def concat_answer_context(dataset):
        dataset['src_txt'] = dataset['answers']['text'][0] + '[SEP]' + dataset[
            'context']
        return dataset

    from datasets import load_dataset
    dataset_dict = load_dataset('luozhouyang/dureader', 'robust')

    train_dataset = dataset_dict['train'].map(concat_answer_context) \
        .rename_columns({'question': 'tgt_txt'}).remove_columns('context') \
        .remove_columns('id').remove_columns('answers')
    eval_dataset = dataset_dict['validation'].map(concat_answer_context) \
        .rename_columns({'question': 'tgt_txt'}).remove_columns('context') \
        .remove_columns('id').remove_columns('answers')

    tmp_dir = tempfile.TemporaryDirectory().name
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    model_id = 'damo/nlp_plug_text-generation_27B'

    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=tmp_dir)

    trainer = build_trainer(
        name=Trainers.nlp_plug_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank')
    test_trainer_with_model_and_args()
