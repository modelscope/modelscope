# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os

from modelscope.preprocessors.nlp.space.fields.intent_field import \
    IntentBPETextField

FILE_NAME = 'train.json'


def intent_preprocess(path, cfg):

    bpe = IntentBPETextField(path, cfg)
    args = cfg.Dataset
    build_examples_fn = bpe.build_examples_multi_turn if args.trigger_role == 'system' \
        else bpe.build_examples_single_turn
    build_score_matrix_fn = bpe.build_score_matrix
    build_score_matrix_multiprocessing_fn = bpe.build_score_matrix_multiprocessing
    data_paths = list(
        os.path.dirname(c) for c in sorted(
            glob.glob(args.data_dir + '/**/' + FILE_NAME, recursive=True)))
    data_paths = bpe.filter_data_path(data_paths=data_paths)

    for mode in ['train', 'valid', 'test']:
        for data_path in data_paths:
            input_file = os.path.join(data_path, f'{mode}.json')
            output_file = os.path.join(data_path,
                                       f'{mode}.{bpe.tokenizer_type}.jsonl')
            output_score_file = os.path.join(data_path, f'{mode}.Score.npy')
            if os.path.exists(input_file) and not os.path.exists(output_file):
                examples = build_examples_fn(input_file, data_type=mode)
                if examples:
                    bpe.save_examples(examples, output_file)
                else:
                    continue
            if os.path.exists(output_file) and not os.path.exists(output_score_file) and \
                    not args.dynamic_score and 'AnPreDial' in data_path:
                examples = bpe.load_examples(output_file)
                if args.num_process >= 2:
                    score_matrix = build_score_matrix_multiprocessing_fn(
                        examples)
                else:
                    score_matrix = build_score_matrix_fn(examples)
                bpe.save_examples(score_matrix, output_score_file)
