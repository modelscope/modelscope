# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import re
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()

__all__ = ['LanguageIdentificationPipeline']


@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.language_identification)
class LanguageIdentificationPipeline(Pipeline):
    r""" Language Identification Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> pipeline_ins = pipeline(Tasks.text_classification, 'damo/nlp_language_identification-classification-base')
    >>> pipeline_ins('Elon Musk, co-founder and chief executive officer of Tesla Motors.\n' \
    >>>              'Gleichzeitig nahm die Legion an der Befriedung Algeriens teil, die von.\n' \
    >>>              '使用pipeline推理及在线体验功能的时候，尽量输入单句文本，如果是多句长文本建议人工分句。'

    >>> {
    >>>    "labels":[
    >>>        "en",
    >>>        "de",
    >>>        "zh"
    >>>    ],
    >>>    "scores":[
    >>>        [('en', 0.99)],
    >>>        [('de', 1.0)],
    >>>        [('zh', 1.0)]
    >>>    ]
    >>> }
    """

    def __init__(self, model: str, **kwargs):
        """Build a language identification pipeline with a model dir or a model id in the model hub.

        Args:
            model: A Model instance.
        """
        super().__init__(model=model, **kwargs)
        export_dir = model
        self.debug = False

        self.cfg = Config.from_file(
            os.path.join(export_dir, ModelFile.CONFIGURATION))

        joint_vocab_file = os.path.join(
            export_dir, self.cfg[ConfigFields.preprocessor]['vocab'])
        vocabfiles = []
        vocabfiles_reverse = []
        for i, w in enumerate(open(joint_vocab_file, 'rb')):
            w = w.strip()
            try:
                w = w.decode('utf-8')
                vocabfiles.append((w, i))
                vocabfiles_reverse.append((i, w))
            except UnicodeDecodeError:
                # [debug] print error info
                if self.debug:
                    print('error vocab:', w, i)
                pass
        self.vocab = dict(vocabfiles)
        self.vocab_reverse = dict(vocabfiles_reverse)
        self.unk_id = self.vocab.get('<UNK>', 1)
        self.pad_id = self.vocab.get('</S>', 0)

        joint_label_file = os.path.join(
            export_dir, self.cfg[ConfigFields.preprocessor]['label'])
        self.label = dict([(i, w.strip()) for i, w in enumerate(
            open(joint_label_file, 'r', encoding='utf8'))])
        self.unk_label = 'unk'

        tf.reset_default_graph()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)
        tf.saved_model.loader.load(self._session,
                                   [tf.saved_model.tag_constants.SERVING],
                                   export_dir)
        default_graph = tf.get_default_graph()
        # [debug] print graph ops
        if self.debug:
            for op in default_graph.get_operations():
                print(op.name, op.values())

        self.input_ids = default_graph.get_tensor_by_name('src_cid:0')
        output_label = default_graph.get_tensor_by_name('output_label:0')
        output_score = default_graph.get_tensor_by_name('predict_score:0')

        self.output = {
            'output_ids': output_label,
            'output_score': output_score
        }
        init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        self._session.run([init, local_init])
        tf.saved_model.loader.load(self._session,
                                   [tf.saved_model.tag_constants.SERVING],
                                   export_dir)

    def _lid_preprocess(self, input: str) -> list:
        sentence = input.lower()
        # HtmlToText
        CLEANR = r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'
        sentence = re.sub(CLEANR, '', sentence)
        # RemoveLinks
        URLRE = r'\S+[./]\S+\s?'
        sentence = re.sub(URLRE, '', sentence)
        EMAILRE = r'\S*@\S*\s?'
        sentence = re.sub(EMAILRE, '', sentence)

        # SBC2DBC
        def stringpartQ2B(uchar):
            inside_code = ord(uchar)
            if 0xFF00 < inside_code or inside_code > 0xFF5F:
                inside_code -= 0xFEE0
            elif inside_code == 0x3000:
                inside_code = 0x0020
            elif inside_code in [
                    0x301D, 0x301E, 0x201C, 0x201D, 0x201E, 0x201F
            ]:
                inside_code = 0x0022
            elif inside_code in [0x2018, 0x2019, 0x201A, 0x201B]:
                inside_code = 0x0027
            return chr(inside_code)

        # RemoveNoisyChars
        m_noisyChars = ",-+\"\'\\&.!=:;°·$«»|±[]{}_?<>~^*/%#@()，。！《》？、`\xc2\xa0…‼️"
        sentence = ''.join([
            stringpartQ2B(c) if c not in m_noisyChars else ' '
            for c in sentence
        ])
        EMOJIRE = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U0001f926-\U0001f937'  # emoji
            u'\U00010000-\U0010ffff'  # char emoji
            u'\U00002702-\U000027B0'  # char emoji
            u'\u2640-\u2642\u2600-\u2B55'
            u'\u200d\u23cf\u23e9\u231a\ufe0f\u3030'  # dingbats
            ']+',
            re.UNICODE)
        sentence = re.sub(EMOJIRE, '', sentence)
        # RemoveDigitalWords
        sentence = ' '.join([
            item for item in sentence.split()
            if (not bool(re.search(r'\d', item))
                or not bool(re.match(r'^[a-z0-9+-_]+$', item)))
        ])
        # replaceBrandWords
        # wordCorrection
        # removeSpaces
        outids = []
        for w in sentence.strip():
            tmp = self.vocab.get(w, self.unk_id)
            if len(outids
                   ) > 0 and tmp == self.unk_id and outids[-1] == self.unk_id:
                continue
            outids.append(tmp)
        if len(outids) > 0 and outids[0] == self.unk_id:
            outids = outids[1:]
        if len(outids) > 0 and outids[-1] == self.unk_id:
            outids = outids[:-1]
        return outids

    def preprocess(self, input: str) -> Dict[str, Any]:
        sentencelt = input.split('\n')
        input_ids_lt = [
            self._lid_preprocess(sentence) for sentence in sentencelt
            if sentence.strip() != ''
        ]

        # [debug] print info example:
        if self.debug:
            for sentence, input_ids in zip(sentencelt, input_ids_lt):
                print('raw:', sentence)
                print(
                    'res:', ''.join([
                        self.vocab_reverse.get(wid, self.unk_id).replace(
                            '<UNK>', ' ') for wid in input_ids
                    ]))
        maxlen = max([len(ids) for ids in input_ids_lt])
        for ids in input_ids_lt:
            ids.extend([self.pad_id] * (maxlen - len(ids)))
        input_ids = np.array(input_ids_lt)

        result = {'input_ids': input_ids}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_ids: input['input_ids']}
            sess_outputs = self._session.run(self.output, feed_dict=feed_dict)
            return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_scores_raw = inputs['output_score']

        supported_104_lang = set([
            'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ce', 'co',
            'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa',
            'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi',
            'hmn', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv',
            'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lo', 'lt', 'lv',
            'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl',
            'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk',
            'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta',
            'te', 'tg', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh',
            'yi', 'yo', 'zh', 'zh-tw', 'zu'
        ])
        labels_scores_lt = []
        output_labels = []
        for output_score in output_scores_raw:
            tmplt = []
            for s, l in zip(output_score, self.label.values()):
                if l not in supported_104_lang:
                    continue
                tmplt.append((l, s))
            tmplt = sorted(tmplt, key=lambda i: i[1], reverse=True)[:3]
            if len(tmplt) == 0:
                tmplt = [(0, 1.00)]
            labels_scores_lt.append(tmplt)
            output_labels.append(tmplt[0][0])
        output_scores = [[(label, round(score, 2))
                          for label, score in labels_scores if score > 0.01]
                         for labels_scores in labels_scores_lt]

        result = {
            OutputKeys.LABELS: output_labels,
            OutputKeys.SCORES: output_scores
        }
        return result
