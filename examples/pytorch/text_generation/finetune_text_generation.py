from dataclasses import dataclass, field

from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass
class TextGenerationArguments(TrainingArgs):
    
    src_txt: str = field(
        default=None,
        metadata={
            'help': 'The source text key of preprocessor',
            'cfg_node': 'preprocessor.src_txt'
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })
    
    lr_scheduler: str = field(
        default=None,
        metadata={
            'help': 'The lr scheduler type',
            'cfg_node': 'train.lr_scheduler.type'
        })
    
    def __call__(self, config):
        config = super().__call__(config)
        if config.train.lr_scheduler.type == 'noam':
            config.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': noam_lambda,
                'options': {
                    'by_epoch': False
                }
            }
        return config

def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * 100**(-1.5))

args = TextGenerationArguments.from_cli(task='text-generation')

print(args)

dataset = MsDataset.load(args.dataset_name)
train_dataset = dataset['train']
eval_dataset = dataset['test']

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    seed=args.seed,
    cfg_modify_fn=args)

trainer: EpochBasedTrainer = build_trainer(name='nlp-gpt3-trainer', default_args=kwargs)
trainer.train()
