# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp


class EasyCVBaseDataset(object):
    """Adapt to MSDataset.

    Args:
        split_config (dict): Dataset root path from MSDataset, e.g.
            {"train":"local cache path"} or {"evaluation":"local cache path"}.
        preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. Not support yet.
        mode: Training or Evaluation.
    """
    DATA_ROOT_PATTERN = '${data_root}'

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

    def _update_data_root(self, input_dict, data_root):
        for k, v in input_dict.items():
            if isinstance(v, str) and self.DATA_ROOT_PATTERN in v:
                input_dict.update(
                    {k: v.replace(self.DATA_ROOT_PATTERN, data_root)})
            elif isinstance(v, dict):
                self._update_data_root(v, data_root)

    def _update_data_source(self, data_source):
        data_root = next(iter(self.split_config.values()))
        data_root = data_root.rstrip(osp.sep)

        self._update_data_root(data_source, data_root)
