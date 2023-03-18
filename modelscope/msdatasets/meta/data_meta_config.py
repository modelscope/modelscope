# Copyright (c) Alibaba, Inc. and its affiliates.


class DataMetaConfig(object):
    """Modelscope data-meta config class.

    Attributes:
        dataset_scripts(str): The local path of dataset scripts.
        dataset_formation(:obj:`enum.Enum`): Dataset formation, refer to modelscope.utils.constant.DatasetFormations.
        meta_cache_dir(str): Meta cache path.
        meta_data_files(dict): Meta data mapping, Example: {'test': 'https://xxx/mytest.csv'}
        zip_data_files(dict): Data files mapping, Example: {'test': 'pictures.zip'}
        meta_args_map(dict): Meta arguments mapping, Example: {'test': {'file': 'pictures.zip'}, ...}
        target_dataset_structure(dict): Dataset Structure, like
             {
                "default":{
                    "train":{
                        "meta":"my_train.csv",
                        "file":"pictures.zip"
                    }
                },
                "subsetA":{
                    "test":{
                        "meta":"mytest.csv",
                        "file":"pictures.zip"
                    }
                }
            }
        dataset_py_script(str): The python script path of dataset.
        meta_type_map(dict): The custom dataset mapping in meta data,
            Example: {"type": "MovieSceneSegmentationCustomDataset",
                        "preprocessor": "movie-scene-segmentation-preprocessor"}
    """

    def __init__(self):
        self.dataset_scripts = None
        self.dataset_formation = None
        self.meta_cache_dir = None
        self.meta_data_files = None
        self.zip_data_files = None
        self.meta_args_map = None
        self.target_dataset_structure = None
        self.dataset_py_script = None
        self.meta_type_map = {}
