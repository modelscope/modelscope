from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.trainers.training_args import TrainingArgs
from modelscope.trainers import EpochBasedTrainer, build_trainer

training_args = TrainingArgs(task='text-to-image-synthesis').parse_cli()
config, args = training_args.to_config()
print(args)

train_dataset = MsDataset.load(args.train_dataset_name, 
                               split='train',
                               download_mode=DownloadMode.FORCE_REDOWNLOAD)
validation_dataset = MsDataset.load(args.train_dataset_name, 
                                    split='validation',
                                    download_mode=DownloadMode.FORCE_REDOWNLOAD)

def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': lambda _: 1,
        'last_epoch': -1
    }
    cfg.train.optimizer.lr = 1e-4
    return cfg

kwargs = dict(
    model=training_args.model,
    model_revision='v1.0.6',
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    cfg_modify_fn=cfg_modify_fn)

trainer: EpochBasedTrainer = build_trainer(name=Trainers.lora_diffusion, default_args=kwargs)
trainer.train()
