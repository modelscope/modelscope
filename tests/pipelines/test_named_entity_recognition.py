# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import (LSTMForTokenClassificationWithCRF,
                                   ModelForTokenClassificationWithCRF)
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import NamedEntityRecognitionPipeline
from modelscope.preprocessors import \
    TokenClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class NamedEntityRecognitionTest(unittest.TestCase):
    language_examples = {
        'zh':
        '新华社北京二月十一日电（记者唐虹）',
        'en':
        'Italy recalled Marcello Cuttitta',
        'ru':
        'важным традиционным промыслом является производство пальмового масла .',
        'fr':
        'fer à souder électronique',
        'es':
        'el primer avistamiento por europeos de esta zona fue en 1606 , '
        'en la expedición española mandada por luis váez de torres .',
        'nl':
        'in het vorige seizoen promoveerden sc cambuur , dat kampioen werd en go ahead eagles via de play offs .',
        'tr':
        'köyün pırasa kavurması ve içi yağlama ve akıtma adındaki hamur işleri meşhurdur . ; çörek ekmeği ; '
        'diye adlandırdıkları mayasız ekmeği unutmamaklazım .',
        'ko':
        '국립진주박물관은 1984년 11월 2일 개관하였으며 한국 전통목조탑을 석조 건물로 형상화한 것으로 건축가 김수근 선생의 대표적 작품이다 .',
        'fa':
        'ﺞﻤﻋیﺕ ﺍیﻥ ﺎﺴﺗﺎﻧ ۳۰ ﻩﺯﺍﺭ ﻦﻓﺭ ﺎﺴﺗ ﻭ ﻢﻧﺎﺒﻋ ﻢﻬﻣی ﺍﺯ ﺲﻧگ ﺂﻬﻧ ﺩﺍﺭﺩ .',
        'de':
        'die szene beinhaltete lenny baker und christopher walken .',
        'hi':
        '१४९२ में एक चार्टर के आधार पर, उसके पिता ने उसे वाडोविस के उत्तराधिकारी के रूप में छोड़ दिया।',
        'bn':
        'যদিও গির্জার সবসময় রাজকীয় পিউ থাকত, তবে গির্জায় রাজকীয়ভাবে এটিই ছিল প্রথম দেখা।',
        'multi':
        '新华社北京二月十一日电（记者唐虹）',
    }

    all_modelcards_info = [
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-news',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-social_media',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-generic',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-resume',
            'language': 'zh'
        },
        {
            'model_id': 'damo/nlp_lstm_named-entity-recognition_chinese-news',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_lstm_named-entity-recognition_chinese-social_media',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_lstm_named-entity-recognition_chinese-generic',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_lstm_named-entity-recognition_chinese-resume',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-book',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-finance',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-game',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-bank',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-literature',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-cmeee',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-news',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-social_media',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-literature',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-politics',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-music',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-science',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-ai',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-wiki',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-large-generic',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-generic',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_multilingual-large-generic',
            'language': 'multi'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_russian-large-generic',
            'language': 'ru'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_spanish-large-generic',
            'language': 'es'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_dutch-large-generic',
            'language': 'nl'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_turkish-large-generic',
            'language': 'tr'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_korean-large-generic',
            'language': 'ko'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_farsi-large-generic',
            'language': 'fa'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_german-large-generic',
            'language': 'de'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_hindi-large-generic',
            'language': 'hi'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_bangla-large-generic',
            'language': 'bn'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-ecom',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_chinese-base-ecom-50cls',
            'language': 'zh'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_english-large-ecom',
            'language': 'en'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_russian-large-ecom',
            'language': 'ru'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_french-large-ecom',
            'language': 'fr'
        },
        {
            'model_id':
            'damo/nlp_raner_named-entity-recognition_spanish-large-ecom',
            'language': 'es'
        },
        {
            'model_id':
            'damo/nlp_structbert_keyphrase-extraction_base-icassp2023-mug-track4-baseline',
            'language': 'zh'
        },
        {
            'model_id': 'damo/nlp_raner_chunking_english-large',
            'language': 'en'
        },
    ]

    def setUp(self) -> None:
        self.task = Tasks.named_entity_recognition
        self.model_id = 'damo/nlp_raner_named-entity-recognition_chinese-base-news'
        self.english_model_id = 'damo/nlp_raner_named-entity-recognition_english-large-ecom'
        self.chinese_model_id = 'damo/nlp_raner_named-entity-recognition_chinese-large-generic'
        self.tcrf_model_id = 'damo/nlp_raner_named-entity-recognition_chinese-base-news'
        self.lcrf_model_id = 'damo/nlp_lstm_named-entity-recognition_chinese-news'
        self.addr_model_id = 'damo/nlp_structbert_address-parsing_chinese_base'
        self.lstm_model_id = 'damo/nlp_lstm_named-entity-recognition_chinese-generic'
        self.sentence = '这与温岭市新河镇的一个神秘的传说有关。[SEP]地名'
        self.sentence_en = 'pizza shovel'
        self.sentence_zh = '他 继 续 与 貝 塞 斯 達 遊 戲 工 作 室 在 接 下 来 辐 射 4 游 戏 。'
        self.addr = '浙江省杭州市余杭区文一西路969号亲橙里'
        self.addr1 = '浙江省西湖区灵隐隧道'
        self.addr2 = '内蒙古自治区巴彦淖尔市'
        self.ecom = '欧美单 秋季女装时尚百搭休闲修身 亚麻混纺短款 外套西装'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_tcrf_by_direct_model_download(self):
        cache_path = snapshot_download(self.tcrf_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = ModelForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = NamedEntityRecognitionPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lcrf_by_direct_model_download(self):
        cache_path = snapshot_download(self.lcrf_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = LSTMForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = NamedEntityRecognitionPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_tcrf_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.tcrf_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_addrst_with_model_from_modelhub(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_address-parsing_chinese_base')
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.addr))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_addrst_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.addr_model_id)
        print(pipeline_ins(input=self.addr))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_addrst_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.addr_model_id)
        print(
            pipeline_ins(
                input=[self.addr, self.addr1, self.addr2], batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_addrst_with_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=self.addr_model_id,
            padding=False)
        print(pipeline_ins(input=[self.addr, self.addr1, self.addr2]))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lcrf_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.lcrf_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_tcrf_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.tcrf_model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lcrf_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.lcrf_model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lcrf_with_chinese_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.chinese_model_id)
        print(pipeline_ins(input=self.sentence_zh))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lcrf_with_chinese_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=self.chinese_model_id,
            padding=False)
        print(
            pipeline_ins(input=[
                self.sentence_zh, self.sentence_zh[:20], self.sentence_zh[10:]
            ]))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lcrf_with_chinese_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.chinese_model_id)
        print(
            pipeline_ins(
                input=[
                    self.sentence_zh, self.sentence_zh[:20],
                    self.sentence_zh[10:]
                ],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstm_with_chinese_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.lstm_model_id)
        print(pipeline_ins(input=self.sentence_zh))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstm_with_chinese_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=self.lstm_model_id,
            padding=False)
        print(
            pipeline_ins(input=[
                self.sentence_zh, self.sentence_zh[:20], self.sentence_zh[10:]
            ]))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstm_with_chinese_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.lstm_model_id)
        print(
            pipeline_ins(
                input=[
                    self.sentence_zh, self.sentence_zh[:20],
                    self.sentence_zh[10:]
                ],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_english_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.english_model_id)
        print(pipeline_ins(input=self.sentence_en))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_english_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.english_model_id)
        print(
            pipeline_ins(
                input=[self.ecom, self.sentence_zh, self.sentence],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_english_with_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=self.english_model_id,
            padding=False)
        print(pipeline_ins(input=[self.ecom, self.sentence_zh, self.sentence]))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.named_entity_recognition)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_all_modelcards(self):
        for item in self.all_modelcards_info:
            model_id = item['model_id']
            sentence = self.language_examples[item['language']]
            with self.subTest(model_id=model_id):
                pipeline_ins = pipeline(Tasks.named_entity_recognition,
                                        model_id)
                print(pipeline_ins(input=sentence))


if __name__ == '__main__':
    unittest.main()
