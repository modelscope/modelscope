import os
from dataclasses import dataclass, field

from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.data.preprocessors.nlp_preprocessor import build_preprocessor
from adaseq.training.default_trainer import DefaultTrainer as AdaSeqTrainer

from modelscope import MsDataset, TrainingArgs, build_dataset_from_file


@dataclass(init=False)
class NamedEntityRecognitionArguments(TrainingArgs):
    preprocessor: str = field(
        default='sequence-labeling-preprocessor',
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })

    sequence_length: int = field(
        default=150,
        metadata={
            'cfg_node': 'preprocessor.max_length',
            'help': 'The parameters for train dataset',
        })

    data_collator: str = field(
        default='SequenceLabelingDataCollatorWithPadding',
        metadata={
            'cfg_node': 'data_collator',
            'help': 'The type of data collator',
        })

    dropout: float = field(
        default=0.0,
        metadata={
            'cfg_node': 'model.dropout',
            'help': 'Dropout rate',
        })

    use_crf: bool = field(
        default=True,
        metadata={
            'cfg_node': 'model.use_crf',
            'help': 'Whether to add a CRF decoder layer',
        })

    crf_lr: float = field(
        default=5.0e-1, metadata={
            'help': 'Learning rate for CRF layer',
        })


training_args = NamedEntityRecognitionArguments().parse_cli()
config, args = training_args.to_config()
print(args)

if args.dataset_json_file is None:
    train_dataset = MsDataset.load(
        args.train_dataset_name,
        subset_name=args.train_subset_name,
        split=args.train_split,
        namespace=args.train_dataset_namespace).to_hf_dataset()
    validation_dataset = MsDataset.load(
        args.val_dataset_name,
        subset_name=args.val_subset_name,
        split=args.val_split,
        namespace=args.val_dataset_namespace).to_hf_dataset()
else:
    train_dataset, validation_dataset = build_dataset_from_file(
        args.dataset_json_file)
dm = DatasetManager({
    'train': train_dataset,
    'valid': validation_dataset
}, labels={'type': 'count_span_labels'})  # yapf: disable

config.preprocessor.model_dir = args.model
config.model.embedder = {'model_name_or_path': args.model}
preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)
config.model.id_to_label = preprocessor.id_to_label
data_collator = build_data_collator(preprocessor.tokenizer,
                                    dict(type=config.data_collator))
config.train.optimizer.param_groups = [{'regex': 'crf', 'lr': args.crf_lr}]

cfg_file = os.path.join(config.train.work_dir, 'config.yaml')
config.dump(cfg_file)

kwargs = dict(
    cfg_file=cfg_file,
    work_dir=config.train.work_dir,
    dataset_manager=dm,
    data_collator=data_collator,
    preprocessor=preprocessor)

trainer = AdaSeqTrainer(**kwargs)
trainer.train()
