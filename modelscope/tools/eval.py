# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse

from modelscope.trainers import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate a model')
    parser.add_argument('config', help='config file path', type=str)
    parser.add_argument(
        '--trainer_name', help='name for trainer', type=str, default=None)
    parser.add_argument(
        '--checkpoint_path',
        help='checkpoint to be evaluated',
        type=str,
        default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kwargs = dict(cfg_file=args.config)
    trainer = build_trainer(args.trainer_name, kwargs)
    trainer.evaluate(args.checkpoint_path)


if __name__ == '__main__':
    main()
