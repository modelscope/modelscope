import argparse
import copy
import csv
import datetime
import itertools as it
import multiprocessing
import os
import pickle
import random
import subprocess
import sys

import json
from termcolor import colored

CONFIG = [{
    'lr': [1e-5, 2e-5],
    'batch-size': [16, 32],
    'epochs': [20, 40],
    'warmup': [0.1],
    'weight-decay': [0.1],
    # "adam-beta2": [0.98],
    # "adam-eps": [1e-8],
    'seed': [1, 2, 3]
}]

TASK_CONFIG = {
    'rte': ('--task rte '
            '--data-dir /root/data/superglue/RTE '
            '--seq-length 256 '),
    'cb': ('--task cb '
           '--data-dir /root/data/superglue/CB '
           '--seq-length 256 '),
    'multirc': ('--task multirc '
                '--data-dir /root/data/superglue/MultiRC '
                '--seq-length 430 '),
}

MODEL_CONFIG = {
    'blocklm-roberta-large':
    ('--block-lm '
     '--cloze-eval '
     '--num-layers 24 '
     '--hidden-size 1024 '
     '--num-attention-heads 16 '
     '--max-position-embeddings 512 '
     '--tokenizer-model-type roberta '
     '--tokenizer-type GPT2BPETokenizer '
     '--load-pretrained /root/data/checkpoints/blocklm-roberta-large/250000 '),
    'blocklm-base-na':
    ('--block-lm '
     '--cloze-eval '
     '--num-layers 12 '
     '--hidden-size 768 '
     '--num-attention-heads 12 '
     '--max-position-embeddings 512 '
     '--tokenizer-model-type bert-base-uncased '
     '--tokenizer-type BertWordPieceTokenizer '
     '--load-pretrained /root/data/checkpoints/blocklm-base-len6-na03-12-21-21'
     ),
}

CHECKPOINT_PATH = '/root/data/finetune_checkpoints'
RESULT_PATH = 'runs/{EXPERIMENT_NAME}/results.json'
LOG_PATH = 'logs/'

DISTRIBUTED_ARGS = '--nproc_per_node {N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port {MASTER_PORT}'  # noqa

COMMON_ARGS = ('--save-interval 10000 '
               '--log-interval 50 '
               '--eval-interval 1000 '
               '--eval-iters 100 ')


def get_command(model, task, n_gpu, config, overwrite=True):

    distributed_args = DISTRIBUTED_ARGS.format(
        N_GPU=n_gpu, MASTER_PORT=random.randint(10000, 65535))

    config = copy.deepcopy(config)
    hyper = '-'.join([f'{k}-{v}' for k, v in config.items()])
    experiment_name = f'{model}-{task}/{hyper}'

    command = (
        f'python -m torch.distributed.launch {distributed_args} finetune_gpt2.py '
        f'--finetune {MODEL_CONFIG[model]} {TASK_CONFIG[task]} {COMMON_ARGS} '
        f'--experiment-name {experiment_name} '
        f'--save {CHECKPOINT_PATH} '
        f'--checkpoint-activations '
        f'--eval-batch-size 16 ')

    config['batch-size'] = config['batch-size'] // n_gpu
    command = update_cmd(command, config)
    if overwrite:
        command += '--overwrite '

    result_path = RESULT_PATH.format(EXPERIMENT_NAME=experiment_name)
    log_path = LOG_PATH + f'{model}-{task}-{hyper}.txt'

    return command, result_path, log_path


def chain_configs(configs):
    '''
        @param configs list of configurations
    '''
    all_configs = []
    for config in configs:
        # preserve order of configs
        keys = sorted(config)
        all_args = it.product(*(config[k] for k in keys))
        all_args_dict = [dict(zip(keys, c)) for c in all_args]

        all_configs.append(all_args_dict)

    return it.chain(*all_configs)  # flatten result


def update_cmd(cmd, config):
    '''
        @param cmd str
        @param configs list of dicts
    '''
    for k, v in config.items():
        if v is None:
            continue
        if type(v) == bool:
            if v:
                cmd += '--{} '.format(k)
        else:
            cmd += '--{} {} '.format(k, v)

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dispatcher to run all experiments')

    parser.add_argument(
        '--gpu', type=str, default='0,1,2,3', help='list of available gpus')
    parser.add_argument(
        '--n_gpu', type=int, default=1, help='number of gpus per job')
    parser.add_argument('--model', type=str, default='blocklm-roberta-large')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='whether to rerun experiments with the same result '
        'file location')
    parser.add_argument('--debug', action='store_true', default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    assert args.model in MODEL_CONFIG
    assert args.task in TASK_CONFIG

    # compute cartesian product for each set of configurations
    configs = chain_configs(CONFIG)
    all_configs = configs

    # queues
    gpu_list = args.gpu.split(',')
    total_gpu = len(gpu_list)

    gpu_queues = []
    for i in range(0, total_gpu, args.n_gpu):
        gpu = ','.join(gpu_list[i:i + args.n_gpu])
        gpu_queues.append((multiprocessing.Queue(), gpu))
    done_queue = multiprocessing.Queue()

    indx = 0
    num_jobs = 0

    for config in all_configs:
        gpu_queues[indx][0].put(config)
        indx = (indx + 1) % len(gpu_queues)
        num_jobs += 1

    for job_queue, gpu in gpu_queues:
        print('Start GPU worker {} with {} jobs'.format(
            gpu, job_queue.qsize()))
        multiprocessing.Process(
            target=_worker, args=(gpu, job_queue, done_queue, args)).start()

    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    summary_path = LOG_PATH + f'grid_{args.model}-{args.task}_{timestamp}.txt'

    print('Summary path:', summary_path)

    for _ in range(num_jobs):
        result_path, config = done_queue.get()

        try:
            res = json.load(open(result_path))
        except Exception as e:
            print('Experiment at {} failed'.format(
                colored(result_path, 'red')))
            print(e)
            continue

        with open(summary_path, 'a') as f:
            f.write('Config: ' + json.dumps(config) + '\n')
            f.write(json.dumps(res) + '\n')

    print('Done')


def _worker(gpu, queue, done_queue, args):
    while not queue.empty():
        config = queue.get()
        if config is None:
            return
        done_queue.put(_launch_experiment(gpu, config, args))


def _launch_experiment(gpu, config, args):

    command, result_path, log_path = get_command(args.model, args.task,
                                                 args.n_gpu, config,
                                                 args.overwrite)

    shell_cmd = f'CUDA_VISIBLE_DEVICES={gpu} ' + command
    if not args.debug:
        shell_cmd += f' > {log_path} 2>&1; '

    print('Time {}, launched exp: {}'.format(
        str(datetime.datetime.now()), log_path))

    # if experiment has already been run, skip
    if not os.path.exists(result_path) or args.overwrite:
        return_code = subprocess.call(shell_cmd, shell=True)  # noqa

    if not os.path.exists(result_path):
        # running this process failed, alert me
        print(
            'Dispatcher, Alert! Job has crashed! Check logfile at:[{}]'.format(
                log_path))

    return result_path, config


if __name__ == '__main__':
    main()
