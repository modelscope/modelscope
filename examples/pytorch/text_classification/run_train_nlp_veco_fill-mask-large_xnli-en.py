from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import os

def test_finetune_nlp_veco_fill_mask_large():
    langs = ['en']
    langs_eval = ['en']
    train_datasets = []
    for lang in langs:
        train_datasets.append(
            MsDataset.load('xnli', language=lang, split='train'))
    eval_datasets = []
    for lang in langs_eval:
        eval_datasets.append(
            MsDataset.load(
                'xnli', language=lang, split='validation'))
    train_len = sum([len(dataset) for dataset in train_datasets])
    labels = [0, 1, 2]

    def cfg_modify_fn(cfg):
        cfg.task = 'nli'
        cfg['preprocessor'] = {'type': 'nli-tokenizer'}
        cfg['dataset'] = {
            'train': {
                'first_sequence': 'premise',
                'second_sequence': 'hypothesis',
                'labels': labels,
                'label': 'label',
            }
        }
        cfg['train'] = {
            'work_dir':
            '/tmp',
            'max_epochs':
            2,
            'dataloader': {
                'batch_size_per_gpu': 16,
                'workers_per_gpu': 1
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 2e-5,
                'options': {
                    'cumulative_iters': 8,
                }
            },
            'lr_scheduler': {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters': int(train_len / 16) * 2,
                'options': {
                    'by_epoch': False
                }
            },
            'hooks': [{
                'type': 'CheckpointHook',
                'interval': 1,
                'save_dir': '/tmp'
            },
                {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 500
            }]
        }
        return cfg

    kwargs = dict(
        model='damo/nlp_veco_fill-mask-large',
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        work_dir='/tmp',
        cfg_modify_fn=cfg_modify_fn)

    os.environ['LOCAL_RANK'] = '0'
    trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
    trainer.train()

if __name__ == '__main__':
    test_finetune_nlp_veco_fill_mask_large()

