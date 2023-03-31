import os
import time
import shutil
from datasets import load_dataset
from functools import reduce
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.preprocessors.nlp import WordSegmentationBlankSetToLabelPreprocessor

# MAX_EPOCH
MAX_EPOCH=10
# WORK_DIR
WORK_DIR='/tmp'

def test_finetune_cluener(MAX_EPOCH=MAX_EPOCH,WORK_DIR=WORK_DIR):
    # download cluener
    os.system(
        f'curl https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip > {WORK_DIR}/cluener_public.zip'
    )
    time.sleep(100)
    shutil.unpack_archive(f'{WORK_DIR}/cluener_public.zip', f'{WORK_DIR}/cluener_public')
    data_files={
        'train':'train.json',
        'eval':'dev.json'
    }
    train_dataset = load_dataset(
        'json',
        data_files=f'{WORK_DIR}/cluener_public/{data_files["train"]}',
        split='train'
    )
    eval_dataset = load_dataset(
        'json',
        data_files=f'{WORK_DIR}/cluener_public/{data_files["eval"]}',
        split='train'
    )

    # convert data format
    def convert_json2conll(example):
        text_str=''
        text=example['text']
        label_list = ['0'] * len(text)
        for i,word in enumerate(text):
            if i==0:
                text_str+=word
            else:
                text_str+=" "+word
        label_str=''
        label = example['label']
        for entity_name, info in label.items():
            if info == None:
                continue
            else:
                for entity_text, index in info.items():
                    if index == None:
                        continue
                    else:
                        for index_num in range(len(index)):
                            if len(entity_text) == 1:
                                label_list[index[index_num][0]] = f"B-{entity_name}"
                            else:
                                for i, idx in enumerate(range(int(index[index_num][0]), int(index[index_num][1]) + 1)):
                                    if i == 0:
                                        label_list[idx] = f"B-{entity_name}"
                                    else:
                                        label_list[idx] = f"I-{entity_name}"
        for i, t in enumerate(label_list):
            if i == 0:
                label_str+=t
            else:
                label_str+=f" {t}"
        example['text'] = text_str
        example['label']=label_str
        return example
    train_dataset=train_dataset.map(convert_json2conll,batched=False)
    eval_dataset=eval_dataset.map(convert_json2conll,batched=False)

    preprocessor = WordSegmentationBlankSetToLabelPreprocessor()
    def split_to_dict(example):
        data = example['label'].split(' ')
        data = list(filter(lambda x: len(x) > 0, data))
        example['label'] = data
        return preprocessor(example['text'])
    train_dataset = train_dataset.map(split_to_dict, batched=False)
    eval_dataset = eval_dataset.map(split_to_dict, batched=False)

    def reducer(x, y):
        x = x.split(' ') if isinstance(x, str) else x
        y = y.split(' ') if isinstance(y, str) else y
        return x + y
    label_enumerate_values = list(
        set(reduce(reducer, eval_dataset['label'])))
    label_enumerate_values.sort()

    def cfg_modify_fn(cfg):
        cfg.task = 'token-classification'
        cfg['dataset'] = {
            'train': {
                'labels': label_enumerate_values,
                'first_sequence': 'tokens',
                'label': 'label',
            },
            'val': {
                'labels': label_enumerate_values,
                'first_sequence': 'tokens',
                'label': 'label',
            }
        }
        cfg['preprocessor'] = {
            'type': 'token-cls-tokenizer',
            'padding': 'max_length'
        }

        cfg['train'] = {
            'work_dir':
                WORK_DIR,
            'max_epochs':
                MAX_EPOCH,
            'dataloader': {
                'batch_size_per_gpu': 32,
                'workers_per_gpu': 1
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 5e-5,
                'options': {}
            },
            'lr_scheduler': {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters': 3328,
                'options': {
                    'by_epoch': False
                }
            },
            'hooks': [
                {
                    'type': 'TextLoggerHook',
                    'interval': 1
                }, {
                    'type': 'IterTimerHook'
                }, {
                    'type': 'EvaluationHook',
                    'by_epoch': False,
                    'interval': 20
                }]
        }
        return cfg

    kwargs = dict(
        model='damo/nlp_structbert_backbone_base_std',
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn)

    os.environ['LOCAL_RANK'] = '0'
    trainer = build_trainer(name=Trainers.nlp_base_trainer,default_args=kwargs)
    trainer.train()

if __name__ == '__main__':
    test_finetune_cluener()



