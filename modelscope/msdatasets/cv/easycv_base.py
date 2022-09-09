# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp


class EasyCVBaseDataset(object):
    """Adapt to MSDataset.
    Subclasses need to implement ``DATA_STRUCTURE``, the format is as follows, e.g.:

    {
        '${data source name}': {
            'train':{
                '${image root arg}': 'images',  # directory name of images relative to the root path
                '${label root arg}': 'labels',  # directory name of lables relative to the root path
                ...
            },
            'validation': {
                '${image root arg}': 'images',
                '${label root arg}': 'labels',
                ...
            }
        }
    }

    Args:
        split_config (dict): Dataset root path from MSDataset, e.g.
            {"train":"local cache path"} or {"evaluation":"local cache path"}.
        preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. Not support yet.
        mode: Training or Evaluation.
    """
    DATA_STRUCTURE = None

    def __init__(self,
                 split_config=None,
                 preprocessor=None,
                 mode=None,
                 args=(),
                 kwargs={}) -> None:
        self.split_config = split_config
        self.preprocessor = preprocessor
        self.mode = mode
        if self.split_config is not None:
            self._update_data_source(kwargs['data_source'])

    def _update_data_source(self, data_source):
        data_root = next(iter(self.split_config.values()))
        split = next(iter(self.split_config.keys()))

        # TODO: msdataset should support these keys to be configured in the dataset's json file and passed in
        if data_source['type'] not in list(self.DATA_STRUCTURE.keys()):
            raise ValueError(
                'Only support %s now, but get %s.' %
                (list(self.DATA_STRUCTURE.keys()), data_source['type']))

        # join data root path of msdataset and default relative name
        update_args = self.DATA_STRUCTURE[data_source['type']][split]
        for k, v in update_args.items():
            data_source.update({k: osp.join(data_root, v)})
