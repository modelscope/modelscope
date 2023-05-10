import os
import shutil
import tempfile
import unittest

from modelscope.msdatasets.dataset_cls.custom_datasets.torch_custom_dataset import TorchCustomDataset
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import DistributedTestCase, test_level
from stanford_alpaca.train import *
from modelscope.models.nlp.llama import LlamaForTextGeneration, LlamaTokenizerFast


class SupervisedDataset(TorchCustomDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])





if __name__ == '__main__':

    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'CosineAnnealingLR',
            'T_max': 1,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            'type': 'AdamW',
            'lr': 2e-5,
            "weight_decay": 0.0,
            "options": {
                "warmup": {
                    "type": "LinearWarmup",
                    "warmup_ratio": 0.03
                }
            }
        }
        cfg.train["bf16"] = True
        cfg.train["gradient_accumulation_steps"] = 8
        cfg.train.dataloader = {
            'batch_size_per_gpu': 4,
            'workers_per_gpu': 2
        }
        cfg.train.hooks.append({
            "type": "DeepspeedHook",
            "config": "/root/work/stanford_alpaca/configs/default_offload_opt_param.json",
            "with_mpu": False,
        })
        
        
        cfg.preprocessor.sequence_length = 512
        return cfg

    model_name_or_path="/run/model/llama-7b"
    model = LlamaForTextGeneration.from_pretrained(
        model_name_or_path,
        cache_dir="/run/model/ms_out",
    )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_name_or_path,
        cache_dir="/run/model/ms_out",
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path='/root/work/stanford_alpaca/alpaca_data.json')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    kwargs = dict(
        model=model,
        cfg_file=os.path.join(model_name_or_path, "configuration.json"),
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        max_epochs=1,
        launcher='pytorch',
        work_dir="/run/model/ms_out",
        cfg_modify_fn=cfg_modify_fn)

    # Construct trainer and train
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs)
    trainer.train()

