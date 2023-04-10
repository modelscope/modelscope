# coding = utf-8

import os
from argparse import ArgumentParser

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='The model id or model dir')
    parser.add_argument('--train_scp', type=str, help='The train scp file')
    parser.add_argument('--cv_scp', type=str, help='The cv scp file')
    parser.add_argument('--merge_trans', type=str, help='The merge trans file')
    parser.add_argument('--keywords', type=str, help='The key words')
    parser.add_argument('--work_dir', type=str, help='The work dir')
    parser.add_argument('--test_scp', type=str, help='The test scp file')
    parser.add_argument('--test_trans', type=str, help='The test trains file')
    args = parser.parse_args()
    print(args)

    # s1
    work_dir = args.work_dir

    # s2
    model_id = args.model
    configs = read_config(model_id)
    config_file = os.path.join(work_dir, 'config.json')
    configs.dump(config_file)

    # s3
    kwargs = dict(
        model=model_id,
        work_dir=work_dir,
        cfg_file=config_file,
    )
    trainer = build_trainer(
        Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

    # s4
    train_scp = args.train_scp
    cv_scp = args.cv_scp
    trans_file = args.merge_trans
    kwargs = dict(train_data=train_scp, cv_data=cv_scp, trans_data=trans_file)
    trainer.train(**kwargs)

    # s5
    keywords = args.keywords
    test_dir = os.path.join(work_dir, 'test_dir')
    test_scp = args.test_scp
    trans_file = args.test_trans
    rank = int(os.environ['RANK'])
    if rank == 0:
        kwargs = dict(
            test_dir=test_dir,
            test_data=test_scp,
            trans_data=trans_file,
            gpu=0,
            keywords=keywords,
            batch_size=args.batch_size,
        )
        trainer.evaluate(None, None, **kwargs)


if __name__ == '__main__':
    main()
