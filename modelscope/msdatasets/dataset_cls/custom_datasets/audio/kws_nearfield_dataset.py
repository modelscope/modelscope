# Copyright (c) 2021 Binbin Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_processor as processor
from modelscope.trainers.audio.kws_utils.file_utils import (make_pair,
                                                            read_lists,
                                                            tokenize)
from modelscope.utils.logger import get_logger

logger = get_logger()


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data

    def dump(self, dump_file):
        with open(dump_file, 'w', encoding='utf8') as fout:
            for obj in self.lists:
                if hasattr(obj, 'get') and obj.get('tokens', None) is not None:
                    assert 'key' in obj
                    assert 'wav' in obj
                    assert 'txt' in obj
                    assert len(obj['tokens']) == len(obj['txt'])
                    dump_line = obj['key'] + ':\n'
                    dump_line += '\t' + obj['wav'] + '\n'
                    dump_line += '\t'
                    for token, idx in zip(obj['tokens'], obj['txt']):
                        dump_line += '%s(%d) ' % (token, idx)
                    dump_line += '\n\n'
                    fout.write(dump_line)
                else:
                    infos = json.loads(obj)
                    assert 'key' in infos
                    assert 'wav' in infos
                    assert 'txt' in infos
                    dump_line = infos['key'] + ':\n'
                    dump_line += '\t' + infos['wav'] + '\n'
                    dump_line += '\t'
                    dump_line += '%d' % infos['txt']
                    dump_line += '\n\n'
                    fout.write(dump_line)


def kws_nearfield_dataset(data_file,
                          trans_file,
                          conf,
                          symbol_table,
                          lexicon_table,
                          need_dump=False,
                          dump_file='',
                          partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_file (str): wave list with kaldi style
            trans_file (str): transcription list with kaldi style
            symbol_table (Dict): token list, [token_str, token_id]
            lexicon_table (Dict): words list defined with basic tokens
            need_dump (bool): whether to dump data with mapping tokens or not
            dump_file (str): dumping file path
            partition (bool): whether to do data partition in terms of rank
    """

    lists = []
    filter_conf = conf.get('filter_conf', {})

    wav_lists = read_lists(data_file)
    trans_lists = read_lists(trans_file)
    lists = make_pair(wav_lists, trans_lists)
    lists = tokenize(lists, symbol_table, lexicon_table)

    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if need_dump:
        dataset.dump(dump_file)

    dataset = Processor(dataset, processor.parse_wav)
    dataset = Processor(dataset, processor.filter, **filter_conf)

    feature_extraction_conf = conf.get('feature_extraction_conf', {})
    if feature_extraction_conf['feature_type'] == 'mfcc':
        dataset = Processor(dataset, processor.compute_mfcc,
                            **feature_extraction_conf)
    elif feature_extraction_conf['feature_type'] == 'fbank':
        dataset = Processor(dataset, processor.compute_fbank,
                            **feature_extraction_conf)

    spec_aug = conf.get('spec_aug', True)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)

    context_expansion = conf.get('context_expansion', False)
    if context_expansion:
        context_expansion_conf = conf.get('context_expansion_conf', {})
        dataset = Processor(dataset, processor.context_expansion,
                            **context_expansion_conf)

    frame_skip = conf.get('frame_skip', 1)
    if frame_skip > 1:
        dataset = Processor(dataset, processor.frame_skip, frame_skip)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset
