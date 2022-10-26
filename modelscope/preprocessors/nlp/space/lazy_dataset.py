# Copyright (c) Alibaba, Inc. and its affiliates.
import json


class LazyDataset(object):
    """
    Lazy load dataset from disk.

    Each line of data file is a preprocessed example.
    """

    def __init__(self, data_file, reader, transform=lambda s: json.loads(s)):
        """
        Initialize lazy dataset.

        By default, loading .jsonl format.

        :param data_file
        :type str

        :param transform
        :type callable
        """
        self.data_file = data_file
        self.transform = transform
        self.reader = reader
        self.offsets = [0]
        with open(data_file, 'r', encoding='utf-8') as fp:
            while fp.readline() != '':
                self.offsets.append(fp.tell())
        self.offsets.pop()
        self.fp = open(data_file, 'r', encoding='utf-8')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        sample = self.transform(self.fp.readline().strip())
        if self.reader.with_mlm:
            sample = self.reader.create_token_masked_lm_predictions(sample)
        return sample
