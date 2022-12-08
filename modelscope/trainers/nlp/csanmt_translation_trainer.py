# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import time
from typing import Dict, Optional

import tensorflow as tf

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.nlp import CsanmtForTranslation
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()


@TRAINERS.register_module(module_name=r'csanmt-translation')
class CsanmtTranslationTrainer(BaseTrainer):

    def __init__(self, model: str, cfg_file: str = None, *args, **kwargs):
        model = self.get_or_download_model_dir(model)
        tf.reset_default_graph()

        self.model_dir = model
        self.model_path = osp.join(model, ModelFile.TF_CHECKPOINT_FOLDER)
        if cfg_file is None:
            cfg_file = osp.join(model, ModelFile.CONFIGURATION)

        super().__init__(cfg_file)

        self.params = {}
        self._override_params_from_file()

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)

        self.source_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='source_wids')
        self.target_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='target_wids')
        self.output = {}

        self.global_step = tf.train.create_global_step()

        self.model = CsanmtForTranslation(self.model_path, **self.params)
        output = self.model(input=self.source_wids, label=self.target_wids)
        self.output.update(output)

        self.model_saver = tf.train.Saver(
            tf.global_variables(),
            max_to_keep=self.params['keep_checkpoint_max'])
        with self._session.as_default() as sess:
            logger.info(f'loading model from {self.model_path}')

            pretrained_variables_map = get_pretrained_variables_map(
                self.model_path)

            tf.train.init_from_checkpoint(self.model_path,
                                          pretrained_variables_map)
            sess.run(tf.global_variables_initializer())

    def _override_params_from_file(self):

        self.params['hidden_size'] = self.cfg['model']['hidden_size']
        self.params['filter_size'] = self.cfg['model']['filter_size']
        self.params['num_heads'] = self.cfg['model']['num_heads']
        self.params['num_encoder_layers'] = self.cfg['model'][
            'num_encoder_layers']
        self.params['num_decoder_layers'] = self.cfg['model'][
            'num_decoder_layers']
        self.params['layer_preproc'] = self.cfg['model']['layer_preproc']
        self.params['layer_postproc'] = self.cfg['model']['layer_postproc']
        self.params['shared_embedding_and_softmax_weights'] = self.cfg[
            'model']['shared_embedding_and_softmax_weights']
        self.params['shared_source_target_embedding'] = self.cfg['model'][
            'shared_source_target_embedding']
        self.params['initializer_scale'] = self.cfg['model'][
            'initializer_scale']
        self.params['position_info_type'] = self.cfg['model'][
            'position_info_type']
        self.params['max_relative_dis'] = self.cfg['model']['max_relative_dis']
        self.params['num_semantic_encoder_layers'] = self.cfg['model'][
            'num_semantic_encoder_layers']
        self.params['src_vocab_size'] = self.cfg['model']['src_vocab_size']
        self.params['trg_vocab_size'] = self.cfg['model']['trg_vocab_size']
        self.params['attention_dropout'] = 0.0
        self.params['residual_dropout'] = 0.0
        self.params['relu_dropout'] = 0.0

        self.params['train_src'] = self.cfg['dataset']['train_src']
        self.params['train_trg'] = self.cfg['dataset']['train_trg']
        self.params['vocab_src'] = self.cfg['dataset']['src_vocab']['file']
        self.params['vocab_trg'] = self.cfg['dataset']['trg_vocab']['file']

        self.params['num_gpus'] = self.cfg['train']['num_gpus']
        self.params['warmup_steps'] = self.cfg['train']['warmup_steps']
        self.params['update_cycle'] = self.cfg['train']['update_cycle']
        self.params['keep_checkpoint_max'] = self.cfg['train'][
            'keep_checkpoint_max']
        self.params['confidence'] = self.cfg['train']['confidence']
        self.params['optimizer'] = self.cfg['train']['optimizer']
        self.params['adam_beta1'] = self.cfg['train']['adam_beta1']
        self.params['adam_beta2'] = self.cfg['train']['adam_beta2']
        self.params['adam_epsilon'] = self.cfg['train']['adam_epsilon']
        self.params['gradient_clip_norm'] = self.cfg['train'][
            'gradient_clip_norm']
        self.params['learning_rate_decay'] = self.cfg['train'][
            'learning_rate_decay']
        self.params['initializer'] = self.cfg['train']['initializer']
        self.params['initializer_scale'] = self.cfg['train'][
            'initializer_scale']
        self.params['learning_rate'] = self.cfg['train']['learning_rate']
        self.params['train_batch_size_words'] = self.cfg['train'][
            'train_batch_size_words']
        self.params['scale_l1'] = self.cfg['train']['scale_l1']
        self.params['scale_l2'] = self.cfg['train']['scale_l2']
        self.params['train_max_len'] = self.cfg['train']['train_max_len']
        self.params['num_of_epochs'] = self.cfg['train']['num_of_epochs']
        self.params['save_checkpoints_steps'] = self.cfg['train'][
            'save_checkpoints_steps']
        self.params['num_of_samples'] = self.cfg['train']['num_of_samples']
        self.params['eta'] = self.cfg['train']['eta']

        self.params['beam_size'] = self.cfg['evaluation']['beam_size']
        self.params['lp_rate'] = self.cfg['evaluation']['lp_rate']
        self.params['max_decoded_trg_len'] = self.cfg['evaluation'][
            'max_decoded_trg_len']

        self.params['seed'] = self.cfg['model']['seed']

    def train(self, *args, **kwargs):
        logger.info('Begin csanmt training')

        train_src = osp.join(self.model_dir, self.params['train_src'])
        train_trg = osp.join(self.model_dir, self.params['train_trg'])
        vocab_src = osp.join(self.model_dir, self.params['vocab_src'])
        vocab_trg = osp.join(self.model_dir, self.params['vocab_trg'])

        epoch = 0
        iteration = 0

        with self._session.as_default() as tf_session:
            while True:
                epoch += 1
                if epoch >= self.params['num_of_epochs']:
                    break
                tf.logging.info('%s: Epoch %i' % (__name__, epoch))
                train_input_fn = input_fn(
                    train_src,
                    train_trg,
                    vocab_src,
                    vocab_trg,
                    batch_size_words=self.params['train_batch_size_words'],
                    max_len=self.params['train_max_len'],
                    num_gpus=self.params['num_gpus']
                    if self.params['num_gpus'] > 1 else 1,
                    is_train=True,
                    session=tf_session,
                    epoch=epoch)

                features, labels = train_input_fn

                try:
                    while True:
                        features_batch, labels_batch = tf_session.run(
                            [features, labels])
                        iteration += 1
                        feed_dict = {
                            self.source_wids: features_batch,
                            self.target_wids: labels_batch
                        }
                        sess_outputs = self._session.run(
                            self.output, feed_dict=feed_dict)
                        loss_step = sess_outputs['loss']
                        logger.info('Iteration: {}, step loss: {:.6f}'.format(
                            iteration, loss_step))

                        if iteration % self.params[
                                'save_checkpoints_steps'] == 0:
                            tf.logging.info('%s: Saving model on step: %d.' %
                                            (__name__, iteration))
                            ck_path = self.model_dir + 'model.ckpt'
                            self.model_saver.save(
                                tf_session,
                                ck_path,
                                global_step=tf.train.get_global_step())

                except tf.errors.OutOfRangeError:
                    tf.logging.info('epoch %d end!' % (epoch))

            tf.logging.info(
                '%s: NMT training completed at time: %s.' %
                (__name__, time.asctime(time.localtime(time.time()))))

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """evaluate a dataset

        evaluate a dataset via a specific model from the `checkpoint_path` path, if the `checkpoint_path`
        does not exist, read from the config file.

        Args:
            checkpoint_path (Optional[str], optional): the model path. Defaults to None.

        Returns:
            Dict[str, float]: the results about the evaluation
            Example:
            {"accuracy": 0.5091743119266054, "f1": 0.673780487804878}
        """
        pass


def input_fn(src_file,
             trg_file,
             src_vocab_file,
             trg_vocab_file,
             num_buckets=20,
             max_len=100,
             batch_size=200,
             batch_size_words=4096,
             num_gpus=1,
             is_train=True,
             session=None,
             epoch=None):
    src_vocab = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            src_vocab_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
        num_oov_buckets=1)  # NOTE unk-> vocab_size
    trg_vocab = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            trg_vocab_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
        num_oov_buckets=1)  # NOTE unk-> vocab_size
    src_dataset = tf.data.TextLineDataset(src_file)
    trg_dataset = tf.data.TextLineDataset(trg_file)
    src_trg_dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    src_trg_dataset = src_trg_dataset.map(
        lambda src, trg: (tf.string_split([src]), tf.string_split([trg])),
        num_parallel_calls=10).prefetch(1000000)
    src_trg_dataset = src_trg_dataset.map(
        lambda src, trg: (src.values, trg.values),
        num_parallel_calls=10).prefetch(1000000)
    src_trg_dataset = src_trg_dataset.map(
        lambda src, trg: (src_vocab.lookup(src), trg_vocab.lookup(trg)),
        num_parallel_calls=10).prefetch(1000000)

    if is_train:

        def key_func(src_data, trg_data):
            bucket_width = (max_len + num_buckets - 1) // num_buckets
            bucket_id = tf.maximum(
                tf.size(input=src_data) // bucket_width,
                tf.size(input=trg_data) // bucket_width)
            return tf.cast(tf.minimum(num_buckets, bucket_id), dtype=tf.int64)

        def reduce_func(unused_key, windowed_data):
            return windowed_data.padded_batch(
                batch_size_words, padded_shapes=([None], [None]))

        def window_size_func(key):
            bucket_width = (max_len + num_buckets - 1) // num_buckets
            key += 1
            size = (num_gpus * batch_size_words // (key * bucket_width))
            return tf.cast(size, dtype=tf.int64)

        src_trg_dataset = src_trg_dataset.filter(
            lambda src, trg: tf.logical_and(
                tf.size(input=src) <= max_len,
                tf.size(input=trg) <= max_len))
        src_trg_dataset = src_trg_dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=key_func,
                reduce_func=reduce_func,
                window_size_func=window_size_func))

    else:
        src_trg_dataset = src_trg_dataset.padded_batch(
            batch_size * num_gpus, padded_shapes=([None], [None]))

    iterator = tf.data.make_initializable_iterator(src_trg_dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()

    if is_train:
        session.run(iterator.initializer)
        if epoch == 1:
            session.run(tf.tables_initializer())
    return features, labels


def get_pretrained_variables_map(checkpoint_file_path, ignore_scope=None):
    reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_file_path))
    saved_shapes = reader.get_variable_to_shape_map()
    if ignore_scope is None:
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
    else:
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes and all(
                                scope not in var.name
                                for scope in ignore_scope)])
    restore_vars = []
    name2var = dict(
        zip(
            map(lambda x: x.name.split(':')[0], tf.global_variables()),
            tf.global_variables()))
    restore_map = {}
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                restore_map[saved_var_name] = curr_var
    return restore_map
