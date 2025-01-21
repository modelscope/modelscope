# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import List, Union

import addict
import json

from modelscope.trainers.cli_argument_parser import CliArgumentParser
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_DATASET_NAMESPACE


def set_flatten_value(values: Union[str, List[str]]):
    pairs = values.split(',') if isinstance(values, str) else values
    _params = {}
    for kv in pairs or []:
        if len(kv.strip()) == 0:
            continue
        key, value = kv.split('=')
        _params[key] = parse_value(value)
    return _params


@dataclass
class DatasetArgs:

    train_dataset_name: str = field(
        default=None,
        metadata={
            'help':
            'The dataset name used for training, can be an id in the datahub or a local dir',
        })

    val_dataset_name: str = field(
        default=None,
        metadata={
            'help':
            'The subset name used for evaluating, can be an id in the datahub or a local dir',
        })

    train_subset_name: str = field(
        default=None,
        metadata={
            'help': 'The subset name used for training, can be None',
        })

    val_subset_name: str = field(
        default=None,
        metadata={
            'help': 'The subset name used for evaluating, can be None',
        })

    train_split: str = field(
        default=None, metadata={
            'help': 'The split of train dataset',
        })

    val_split: str = field(
        default=None, metadata={
            'help': 'The split of val dataset',
        })

    train_dataset_namespace: str = field(
        default=DEFAULT_DATASET_NAMESPACE,
        metadata={
            'help': 'The dataset namespace used for training',
        })

    val_dataset_namespace: str = field(
        default=DEFAULT_DATASET_NAMESPACE,
        metadata={
            'help': 'The dataset namespace used for evaluating',
        })

    dataset_json_file: str = field(
        default=None,
        metadata={
            'help':
            'The json file to parse all datasets from, used in a complex dataset scenario,'
            'the json format should be like:'
            '''
                    [
                        {
                            "dataset": {
                                # All args used in the MsDataset.load function
                                "dataset_name": "xxx",
                                ...
                            },
                            # All columns used, mapping the column names in each dataset in same names.
                            "column_mapping": {
                                "text1": "sequence1",
                                "text2": "sequence2",
                                "label": "label",
                            },
                            # float or str, float means to split the dataset into train/val,
                            # or just str(train/val)
                            "split": 0.8,
                        }
                    ]
                    ''',
        })


@dataclass
class ModelArgs:
    task: str = field(
        default=None,
        metadata={
            'help': 'The task code to be used',
            'cfg_node': 'task'
        })

    model: str = field(
        default=None, metadata={
            'help': 'A model id or model dir',
        })

    model_revision: str = field(
        default=None, metadata={
            'help': 'the revision of model',
        })

    model_type: str = field(
        default=None,
        metadata={
            'help':
            'The mode type, if load_model_config is False, user need to fill this field',
            'cfg_node': 'model.type'
        })


@dataclass
class TrainArgs:

    seed: int = field(
        default=42, metadata={
            'help': 'The random seed',
        })

    per_device_train_batch_size: int = field(
        default=16,
        metadata={
            'cfg_node': 'train.dataloader.batch_size_per_gpu',
            'help':
            'The `batch_size_per_gpu` argument for the train dataloader',
        })

    train_data_worker: int = field(
        default=0,
        metadata={
            'cfg_node': 'train.dataloader.workers_per_gpu',
            'help': 'The `workers_per_gpu` argument for the train dataloader',
        })

    train_shuffle: bool = field(
        default=False,
        metadata={
            'cfg_node': 'train.dataloader.shuffle',
            'help': 'The `shuffle` argument for the train dataloader',
        })

    train_drop_last: bool = field(
        default=False,
        metadata={
            'cfg_node': 'train.dataloader.drop_last',
            'help': 'The `drop_last` argument for the train dataloader',
        })

    per_device_eval_batch_size: int = field(
        default=16,
        metadata={
            'cfg_node': 'evaluation.dataloader.batch_size_per_gpu',
            'help':
            'The `batch_size_per_gpu` argument for the eval dataloader',
        })

    eval_data_worker: int = field(
        default=0,
        metadata={
            'cfg_node': 'evaluation.dataloader.workers_per_gpu',
            'help': 'The `workers_per_gpu` argument for the eval dataloader',
        })

    eval_shuffle: bool = field(
        default=False,
        metadata={
            'cfg_node': 'evaluation.dataloader.shuffle',
            'help': 'The `shuffle` argument for the eval dataloader',
        })

    eval_drop_last: bool = field(
        default=False,
        metadata={
            'cfg_node': 'evaluation.dataloader.drop_last',
            'help': 'The `drop_last` argument for the eval dataloader',
        })

    max_epochs: int = field(
        default=5,
        metadata={
            'cfg_node': 'train.max_epochs',
            'help': 'The training epochs',
        })

    work_dir: str = field(
        default='./train_target',
        metadata={
            'cfg_node': 'train.work_dir',
            'help': 'The directory to save models and logs',
        })

    lr: float = field(
        default=5e-5,
        metadata={
            'cfg_node': 'train.optimizer.lr',
            'help': 'The learning rate of the optimizer',
        })

    lr_scheduler: str = field(
        default='LinearLR',
        metadata={
            'cfg_node': 'train.lr_scheduler.type',
            'help': 'The lr_scheduler type in torch',
        })

    optimizer: str = field(
        default='AdamW',
        metadata={
            'cfg_node': 'train.optimizer.type',
            'help': 'The optimizer type in PyTorch, like `AdamW`',
        })

    optimizer_params: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer',
            'help': 'The optimizer params',
            'cfg_setter': set_flatten_value,
        })

    lr_scheduler_params: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.lr_scheduler',
            'help': 'The lr scheduler params',
            'cfg_setter': set_flatten_value,
        })

    lr_strategy: str = field(
        default='by_epoch',
        metadata={
            'cfg_node': 'train.lr_scheduler.options.lr_strategy',
            'help': 'The lr decay strategy',
            'choices': ['by_epoch', 'by_step', 'no'],
        })

    local_rank: int = field(
        default=0, metadata={
            'help': 'The local rank',
        })

    logging_interval: int = field(
        default=5,
        metadata={
            'help': 'The interval of iter of logging information',
            'cfg_node': 'train.logging.interval',
        })

    eval_strategy: str = field(
        default='by_epoch',
        metadata={
            'help': 'Eval strategy, can be `by_epoch` or `by_step` or `no`',
            'cfg_node': 'evaluation.period.eval_strategy',
            'choices': ['by_epoch', 'by_step', 'no'],
        })

    eval_interval: int = field(
        default=1,
        metadata={
            'help': 'Eval interval',
            'cfg_node': 'evaluation.period.interval',
        })

    eval_metrics: str = field(
        default=None,
        metadata={
            'help': 'The metric name for evaluation',
            'cfg_node': 'evaluation.metrics'
        })

    save_strategy: str = field(
        default='by_epoch',
        metadata={
            'help':
            'Checkpointing strategy, can be `by_epoch` or `by_step` or `no`',
            'cfg_node': 'train.checkpoint.period.save_strategy',
            'choices': ['by_epoch', 'by_step', 'no'],
        })

    save_interval: int = field(
        default=1,
        metadata={
            'help':
            'The interval of epoch or iter of saving checkpoint period',
            'cfg_node': 'train.checkpoint.period.interval',
        })

    save_best_checkpoint: bool = field(
        default=False,
        metadata={
            'help':
            'Save the checkpoint(if it\'s the best) after the evaluation.',
            'cfg_node': 'train.checkpoint.best.save_best',
        })

    metric_for_best_model: str = field(
        default=None,
        metadata={
            'help': 'The metric used to measure the model.',
            'cfg_node': 'train.checkpoint.best.metric_key',
        })

    metric_rule_for_best_model: str = field(
        default='max',
        metadata={
            'help':
            'The rule to measure the model with the metric, can be `max` or `min`',
            'cfg_node': 'train.checkpoint.best.rule',
        })

    max_checkpoint_num: int = field(
        default=None,
        metadata={
            'help':
            'The max number of checkpoints to keep, older ones will be deleted.',
            'cfg_node': 'train.checkpoint.period.max_checkpoint_num',
        })

    max_checkpoint_num_best: int = field(
        default=1,
        metadata={
            'help':
            'The max number of best checkpoints to keep, worse ones will be deleted.',
            'cfg_node': 'train.checkpoint.best.max_checkpoint_num',
        })

    push_to_hub: bool = field(
        default=False,
        metadata={
            'help': 'Push to hub after each checkpointing',
            'cfg_node': 'train.checkpoint.period.push_to_hub',
        })

    repo_id: str = field(
        default=None,
        metadata={
            'help':
            'The repo id in modelhub, usually the format is "group/model"',
            'cfg_node': 'train.checkpoint.period.hub_repo_id',
        })

    hub_token: str = field(
        default=None,
        metadata={
            'help':
            'The modelhub token, you can also set the token to the env variable `MODELSCOPE_API_TOKEN`',
            'cfg_node': 'train.checkpoint.period.hub_token',
        })

    private_hub: bool = field(
        default=True,
        metadata={
            'help': 'Upload to a private hub',
            'cfg_node': 'train.checkpoint.period.private_hub',
        })

    hub_revision: str = field(
        default='master',
        metadata={
            'help': 'Which branch to commit to',
            'cfg_node': 'train.checkpoint.period.hub_revision',
        })

    push_to_hub_best: bool = field(
        default=False,
        metadata={
            'help': 'Push to hub after each checkpointing',
            'cfg_node': 'train.checkpoint.best.push_to_hub',
        })

    repo_id_best: str = field(
        default=None,
        metadata={
            'help':
            'The repo id in modelhub, usually the format is "group/model"',
            'cfg_node': 'train.checkpoint.best.hub_repo_id',
        })

    hub_token_best: str = field(
        default=None,
        metadata={
            'help':
            'The modelhub token, you can also set the token to the env variable `MODELSCOPE_API_TOKEN`',
            'cfg_node': 'train.checkpoint.best.hub_token',
        })

    private_hub_best: bool = field(
        default=True,
        metadata={
            'help': 'Upload to a private hub',
            'cfg_node': 'train.checkpoint.best.private_hub',
        })

    hub_revision_best: str = field(
        default='master',
        metadata={
            'help': 'Which branch to commit to',
            'cfg_node': 'train.checkpoint.best.hub_revision',
        })


@dataclass(init=False)
class TrainingArgs(DatasetArgs, TrainArgs, ModelArgs):

    use_model_config: bool = field(
        default=False,
        metadata={
            'help':
            'Use the configuration of the model, '
            'default will only use the parameters in the CLI and the dataclass',
        })

    def __init__(self, **kwargs):
        self.manual_args = list(kwargs.keys())
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])
        self._unknown_args = {}

    def parse_cli(self, parser_args=None):
        """Construct a TrainingArg class by the parameters of CLI.

        Returns:
            Self
        """
        parser = CliArgumentParser(self)
        args, unknown = parser.parse_known_args(parser_args)
        unknown = [
            item for item in unknown
            if item not in ('\\', '\n') and '--local-rank=' not in item
        ]
        _unknown = {}
        for i in range(0, len(unknown), 2):
            _unknown[unknown[i].replace('-', '')] = parse_value(unknown[i + 1])
        args_dict = vars(args)
        self.manual_args += parser.manual_args
        self._unknown_args.update(_unknown)
        for key, value in deepcopy(args_dict).items():
            if key is not None and hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_config(self, ignore_default_config=None):
        """Convert the TrainingArgs to the `Config`

        Returns:
            The Config, and extra parameters in dict.
        """
        cfg = Config()
        args_dict = addict.Dict()

        if ignore_default_config is None:
            ignore_default_config = self.use_model_config

        for f in fields(self):
            cfg_node = f.metadata.get('cfg_node')
            cfg_setter = f.metadata.get('cfg_setter') or (lambda x: x)
            if cfg_node is not None:
                if f.name in self.manual_args or not ignore_default_config:
                    if isinstance(cfg_node, str):
                        cfg_node = [cfg_node]
                    for _node in cfg_node:
                        cfg.merge_from_dict(
                            {_node: cfg_setter(getattr(self, f.name))})
            else:
                args_dict[f.name] = getattr(self, f.name)

        cfg.merge_from_dict(self._unknown_args)
        return cfg, args_dict

    def get_metadata(self, key):
        _fields = fields(self)
        for f in _fields:
            if f.name == key:
                return f
        return None


def build_dataset_from_file(filename):
    """
    The filename format:
    [
        {
            "dataset": {
                "dataset_name": "xxx",
                ...
            },
            "column_mapping": {
                "text1": "sequence1",
                "text2": "sequence2",
                "label": "label",
            }
            "usage": 0.8,
        }
    ]
    """
    from modelscope import MsDataset
    train_set = []
    eval_set = []

    with open(filename, 'r') as f:
        ds_json = json.load(f)
        for ds in ds_json:
            dataset = MsDataset.load(**ds['dataset']).to_hf_dataset()
            all_columns = dataset.column_names
            keep_columns = ds['column_mapping'].keys()
            remove_columns = [
                column for column in all_columns if column not in keep_columns
            ]
            from datasets import Features
            from datasets import Value
            from datasets import ClassLabel
            features = [
                f for f in dataset.features.items() if f[0] in keep_columns
            ]
            new_features = {}
            for f in features:
                if isinstance(f[1], ClassLabel):
                    new_features[f[0]] = Value(f[1].dtype)
                else:
                    new_features[f[0]] = f[1]
            new_features = Features(new_features)
            dataset = dataset.map(
                lambda x: x,
                remove_columns=remove_columns,
                features=new_features).rename_columns(ds['column_mapping'])
            usage = ds['usage']
            if isinstance(usage, str):
                assert usage in ('train', 'val')
                if usage == 'train':
                    train_set.append(dataset)
                else:
                    eval_set.append(dataset)
            else:
                assert isinstance(usage, float) and 0 < usage < 1
                ds_dict = dataset.train_test_split(train_size=usage)
                train_set.append(ds_dict['train'])
                eval_set.append(ds_dict['test'])

    from datasets import concatenate_datasets
    return concatenate_datasets(train_set), concatenate_datasets(eval_set)


def parse_value(value: str) -> Union[str, float, bool, None]:
    const_map = {
        'True': True,
        'true': True,
        'False': False,
        'false': False,
        'None': None,
        'none': None,
        'null': None
    }
    if value in const_map:
        return const_map[value]
    elif '"' in value or "'" in value:
        return value.replace('"', '').replace("'", '')
    elif re.match(r'^\d+$', value):
        return int(value)
    elif re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                  value):
        return float(value)
    else:
        return value
