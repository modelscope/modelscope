# Copyright (c) Alibaba, Inc. and its affiliates.


class DataMetaConfig(object):
    """Modelscope data-meta config class."""

    def __init__(self):
        self.dataset_scripts = None
        self.dataset_formation = None
        self.meta_cache_dir = None
        self.meta_data_files = None
        self.zip_data_files = None
        self.meta_args_map = None
        self.target_dataset_structure = None
        self.dataset_py_script = None
