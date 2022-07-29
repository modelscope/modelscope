import os
from collections import defaultdict
from typing import Mapping, Optional, Sequence, Union

from datasets.builder import DatasetBuilder

from modelscope.utils.constant import DEFAULT_DATASET_REVISION
from modelscope.utils.logger import get_logger
from .dataset_builder import MsCsvDatasetBuilder

logger = get_logger()


def get_target_dataset_structure(dataset_structure: dict,
                                 subset_name: Optional[str] = None,
                                 split: Optional[str] = None):
    """
    Args:
        dataset_structure (dict): Dataset Structure, like
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
        subset_name (str, optional): Defining the subset_name of the dataset.
        split (str, optional): Which split of the data to load.
    Returns:
           target_subset_name (str): Name of the chosen subset.
           target_dataset_structure (dict): Structure of the chosen split(s), like
           {
               "test":{
                        "meta":"mytest.csv",
                        "file":"pictures.zip"
                    }
            }
    """
    # verify dataset subset
    if (subset_name and subset_name not in dataset_structure) or (
            not subset_name and len(dataset_structure.keys()) > 1):
        raise ValueError(
            f'subset_name {subset_name} not found. Available: {dataset_structure.keys()}'
        )
    target_subset_name = subset_name
    if not subset_name:
        target_subset_name = next(iter(dataset_structure.keys()))
        logger.info(
            f'No subset_name specified, defaulting to the {target_subset_name}'
        )
    # verify dataset split
    target_dataset_structure = dataset_structure[target_subset_name]
    if split and split not in target_dataset_structure:
        raise ValueError(
            f'split {split} not found. Available: {target_dataset_structure.keys()}'
        )
    if split:
        target_dataset_structure = {split: target_dataset_structure[split]}
    return target_subset_name, target_dataset_structure


def get_dataset_files(subset_split_into: dict,
                      dataset_name: str,
                      namespace: str,
                      revision: Optional[str] = DEFAULT_DATASET_REVISION):
    """
    Return:
        meta_map: Structure of meta files (.csv), the meta file name will be replaced by url, like
        {
           "test": "https://xxx/mytest.csv"
        }
        file_map: Structure of data files (.zip), like
        {
            "test": "pictures.zip"
        }
    """
    meta_map = defaultdict(dict)
    file_map = defaultdict(dict)
    from modelscope.hub.api import HubApi
    modelscope_api = HubApi()
    for split, info in subset_split_into.items():
        meta_map[split] = modelscope_api.get_dataset_file_url(
            info['meta'], dataset_name, namespace, revision)
        if info.get('file'):
            file_map[split] = info['file']
    return meta_map, file_map


def load_dataset_builder(dataset_name: str, subset_name: str, namespace: str,
                         meta_data_files: Mapping[str, Union[str,
                                                             Sequence[str]]],
                         zip_data_files: Mapping[str, Union[str,
                                                            Sequence[str]]],
                         cache_dir: str, version: Optional[Union[str]],
                         split: Sequence[str]) -> DatasetBuilder:
    sub_dir = os.path.join(version, '_'.join(split))
    builder_instance = MsCsvDatasetBuilder(
        dataset_name=dataset_name,
        namespace=namespace,
        cache_dir=cache_dir,
        subset_name=subset_name,
        meta_data_files=meta_data_files,
        zip_data_files=zip_data_files,
        hash=sub_dir)

    return builder_instance
