# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import defaultdict
from typing import Optional, Union

from modelscope.hub.api import HubApi
from modelscope.utils.constant import DEFAULT_DATASET_REVISION, MetaDataFields
from modelscope.utils.logger import get_logger

logger = get_logger()


def format_dataset_structure(dataset_structure):
    return {
        k: v
        for k, v in dataset_structure.items()
        if (v.get('meta') or v.get('file'))
    }


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
    target_dataset_structure = format_dataset_structure(
        dataset_structure[target_subset_name])
    if split and split not in target_dataset_structure:
        raise ValueError(
            f'split {split} not found. Available: {target_dataset_structure.keys()}'
        )
    if split:
        target_dataset_structure = {split: target_dataset_structure[split]}
    return target_subset_name, target_dataset_structure


def list_dataset_objects(hub_api: HubApi, max_limit: int, is_recursive: bool,
                         dataset_name: str, namespace: str,
                         version: str) -> list:
    """
    List all objects for specific dataset.

    Args:
        hub_api (class HubApi): HubApi instance.
        max_limit (int): Max number of objects.
        is_recursive (bool): Whether to list objects recursively.
        dataset_name (str): Dataset name.
        namespace (str): Namespace.
        version (str): Dataset version.
    Returns:
        res (list): List of objects, i.e., ['train/images/001.png', 'train/images/002.png', 'val/images/001.png', ...]
    """
    res = []
    objects = hub_api.list_oss_dataset_objects(
        dataset_name=dataset_name,
        namespace=namespace,
        max_limit=max_limit,
        is_recursive=is_recursive,
        is_filter_dir=True,
        revision=version)

    for item in objects:
        object_key = item.get('Key')
        if not object_key:
            continue
        res.append(object_key)

    return res


def contains_dir(file_map) -> bool:
    """
    To check whether input contains at least one directory.

    Args:
        file_map (dict): Structure of data files. e.g., {'train': 'train.zip', 'validation': 'val.zip'}
    Returns:
        True if input contains at least one directory, False otherwise.
    """
    res = False
    for k, v in file_map.items():
        if isinstance(v, str) and not v.endswith('.zip'):
            res = True
            break
    return res


def get_subdir_hash_from_split(split: Union[str, list], version: str) -> str:
    if isinstance(split, str):
        split = [split]
    return os.path.join(version, '_'.join(split))


def get_split_list(split: Union[str, list]) -> list:
    """ Unify the split to list-format. """
    if isinstance(split, str):
        return [split]
    elif isinstance(split, list):
        return split
    else:
        raise f'Expected format of split: str or list, but got {type(split)}.'


def get_split_objects_map(file_map, objects):
    """
    Get the map between dataset split and oss objects.

    Args:
        file_map (dict): Structure of data files. e.g., {'train': 'train', 'validation': 'val'}, both of train and val
            are dirs.
        objects (list): List of oss objects. e.g., ['train/001/1_123.png', 'train/001/1_124.png', 'val/003/3_38.png']
    Returns:
        A map of split-objects. e.g., {'train': ['train/001/1_123.png', 'train/001/1_124.png'],
            'validation':['val/003/3_38.png']}
    """
    res = {}
    for k, v in file_map.items():
        res[k] = []

    for obj_key in objects:
        for k, v in file_map.items():
            if obj_key.startswith(v + '/'):
                res[k].append(obj_key)

    return res


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
    args_map = defaultdict(dict)
    modelscope_api = HubApi()

    for split, info in subset_split_into.items():
        meta_map[split] = modelscope_api.get_dataset_file_url(
            info.get('meta', ''), dataset_name, namespace, revision)
        if info.get('file'):
            file_map[split] = info['file']
        args_map[split] = info.get('args')

    objects = []
    # If `big_data` is true, then fetch objects from meta-csv file directly.
    for split, args_dict in args_map.items():
        if args_dict and args_dict.get(MetaDataFields.ARGS_BIG_DATA):
            meta_csv_file_url = meta_map[split]
            _, script_content = modelscope_api.fetch_single_csv_script(
                meta_csv_file_url)
            if not script_content:
                raise 'Meta-csv file cannot be empty when meta-args `big_data` is true.'
            for item in script_content:
                if not item:
                    continue
                item = item.strip().split(',')[0]
                if item:
                    objects.append(item)
            file_map[split] = objects
    # More general but low-efficiency.
    if not objects:
        objects = list_dataset_objects(
            hub_api=modelscope_api,
            max_limit=-1,
            is_recursive=True,
            dataset_name=dataset_name,
            namespace=namespace,
            version=revision)
        if contains_dir(file_map):
            file_map = get_split_objects_map(file_map, objects)

    return meta_map, file_map, args_map
