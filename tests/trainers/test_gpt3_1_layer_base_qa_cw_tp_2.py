# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import torch
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import DistributedTestCase, test_level
from modelscope.utils.hub import snapshot_download, read_config
from modelscope.utils.hub import read_config, Config
from modelscope.utils.constant import Tasks


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class TestGPT3OneLayerBaseQAandCWtp2(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_dir = snapshot_download('damo/nlp_gpt3_text-generation_1.3B')
        config:Config = read_config(os.path.join(self.model_dir, 'configuration.json'))
        config.megatron.world_size = 2
        config.megatron.tensor_model_parallel_size = 2
        config.dump(os.path.join(self.model_dir, 'configuration.json'))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_poetry(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            finetune_poetry_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_poetry(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            evaluate_poetry_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    # TODO: add gpt3 trainer predict unittest

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_poetry(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            predict_poetry_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_poetry(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            pipeline_poetry_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_dureader(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            finetune_dureader_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_dureader(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            evaluate_dureader_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    # TODO: add gpt3 trainer predict unittest

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_dureader(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            predict_dureader_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_dureader(self):
        dist_start_cmd = 'torchrun --nproc_per_node 2'
        self.start(
            pipeline_dureader_tp_2, num_gpus=2, dist_start_cmd=dist_start_cmd)

class GPT3OneLayerQAandCW():

    def __init__(self,subtask,tp,work_dir):
        self.subtask=subtask
        self.tp=tp
        dataset_name={'qa':'DuReader_robust-QG','cw':'chinese-poetry-collection'}
        dataset_dict=MsDataset.load(dataset_name[subtask])
        if subtask=='qa':
            self.train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
                .map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'}).select(range(20))
            self.eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
                .map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'}).select(range(20))
        if subtask=='cw':
            self.train_dataset = dataset_dict['train'].remap_columns(
                {'text1': 'src_txt'}).select(range(20))
            self.eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'}).select(range(20))
        self.tp=tp
        self.work_dir=work_dir

    def finetune(self,max_epochs,cfg_modify_fn):
        kwargs = dict(
            model='damo/nlp_gpt3_text-generation_1.3B',
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            max_epochs=max_epochs,
            work_dir=self.work_dir,
            cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(
            name=Trainers.gpt3_trainer, default_args=kwargs)
        trainer.train()

    def evaluate(self):
        kwargs = dict(
            model=fr'{self.work_dir}/output',
            eval_dataset=self.eval_dataset,
            work_dir=self.work_dir
        )
        trainer = build_trainer(default_args=kwargs)
        trainer.evaluate()

    def predict(self):
        kwargs = dict(
            model=fr'{self.work_dir}/output',
            predict_datasets=self.eval_dataset,
            work_dir=self.work_dir
        )
        trainer = build_trainer(default_args=kwargs)
        trainer.predict()

    def pipeline(self):
        pipeline_ins = pipeline(Tasks.text_generation, model=fr'{self.work_dir}/output',work_dir=self.work_dir)
        return pipeline_ins


gpt3_one_layer_cw_tp_2=GPT3OneLayerQAandCW(subtask='cw',tp=2,work_dir='./gpt3_poetry_tp_2')
def finetune_poetry_tp_2():

    max_epochs = 2
    num_warmup_steps = 100

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
        cfg.train.dataloader = {
            'batch_size_per_gpu': 2,
            'workers_per_gpu': 1
        }
        cfg.train.hooks.append({
            'type': 'MegatronHook'
        })
        cfg.evaluation.dataloader = {
            'batch_size_per_gpu': 2,
            'workers_per_gpu': 1
        }
        cfg.evaluation.metrics = 'ppl'
        cfg.num_hidden_layers = 1
        cfg.model.strict = False
        return cfg

    gpt3_one_layer_cw_tp_2.finetune(max_epochs=max_epochs,cfg_modify_fn=cfg_modify_fn)


def evaluate_poetry_tp_2():
    gpt3_one_layer_cw_tp_2.evaluate()

def predict_poetry_tp_2():
    gpt3_one_layer_cw_tp_2.predict()

def pipeline_poetry_tp_2():
    pipe=gpt3_one_layer_cw_tp_2.pipeline()
    input='窗含西岭千秋雪'
    gen_content = pipe(input, max_length=128)
    with open(fr'{gpt3_one_layer_cw_tp_2.work_dir}/gpt3_1_layer_base_tp2_cw_pipeline_gen_text.txt', 'w', encoding='utf-8') as f:
        f.write(gen_content)

gpt3_one_layer_qa_tp_2=GPT3OneLayerQAandCW(subtask='qa',tp=2,work_dir='./dureader_tp_2')

def finetune_dureader_tp_2():
    max_epochs = 2
    num_warmup_steps = 100

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
        cfg.train.dataloader = {
            'batch_size_per_gpu': 16,
            'workers_per_gpu': 1
        }
        cfg.train.hooks.append({
            'type': 'EvaluationHook',
            'by_epoch': True,
            'interval': 1
        })
        cfg.train.hooks.append({
            'type': 'MegatronHook'
        })
        cfg.num_hidden_layers = 1
        cfg.preprocessor.sequence_length = 512
        cfg.model.checkpoint_model_parallel_size = 1
        return cfg

    gpt3_one_layer_qa_tp_2.finetune(max_epochs=max_epochs,cfg_modify_fn=cfg_modify_fn)

def evaluate_dureader_tp_2():
    gpt3_one_layer_qa_tp_2.evaluate()

def predict_dureader_tp_2():
    gpt3_one_layer_qa_tp_2.predict()

def pipeline_dureader_tp_2():
    pipe = gpt3_one_layer_qa_tp_2.pipeline()
    input1 = '推荐一下防脱发的洗发水'
    input2='重新推荐一次'
    gen_content1 = pipe(input1, max_length=128)
    with open(fr'{gpt3_one_layer_qa_tp_2.work_dir}/gpt3_1_layer_base_tp2_qa_pipeline_gen_text.txt', 'aw', encoding='utf-8') as f:
        f.write(gen_content1)
    gen_content2 = pipe(input2, max_length=128)
    with open(fr'{gpt3_one_layer_qa_tp_2.work_dir}/gpt3_1_layer_base_tp2_qa_pipeline_gen_text.txt', 'aw',encoding='utf-8') as f:
        f.write(gen_content2)

if __name__ == '__main__':
    unittest.main()


















