# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from typing import Any, Dict

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class DocumentSegmentationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.document_segmentation

    bert_ds_model_id = 'damo/nlp_bert_document-segmentation_chinese-base'
    bert_ds_eng_model_id = 'damo/nlp_bert_document-segmentation_english-base'
    ponet_ts_model_id = 'damo/nlp_ponet_document-segmentation_topic-level_chinese-base'

    sentences = '近年来，随着端到端语音识别的流行，基于Transformer结构的语音识别系统逐渐成为了主流。然而，由于Transformer是一种自回归模型，需要逐个生成目标文字，计算复杂度随着目标文字数量线性增加，限制了其在工业生产中的应用。针对Transoformer模型自回归生成文字的低计算效率缺陷，学术界提出了非自回归模型来并行的输出目标文字。根据生成目标文字时，迭代轮数，非自回归模型分为：多轮迭代式与单轮迭代非自回归模型。其中实用的是基于单轮迭代的非自回归模型。对于单轮非自回归模型，现有工作往往聚焦于如何更加准确的预测目标文字个数，如CTC-enhanced采用CTC预测输出文字个数，尽管如此，考虑到现实应用中，语速、口音、静音以及噪声等因素的影响，如何准确的预测目标文字个数以及抽取目标文字对应的声学隐变量仍然是一个比较大的挑战；另外一方面，我们通过对比自回归模型与单轮非自回归模型在工业大数据上的错误类型（如下图所示，AR与vanilla NAR），发现，相比于自回归模型，非自回归模型，在预测目标文字个数方面差距较小，但是替换错误显著的增加，我们认为这是由于单轮非自回归模型中条件独立假设导致的语义信息丢失。于此同时，目前非自回归模型主要停留在学术验证阶段，还没有工业大数据上的相关实验与结论。'  # noqa *
    sentences_1 = '移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。'  # noqa *
    eng_sentences = 'The Saint Alexander Nevsky Church was established in 1936 by Archbishop Vitaly (Maximenko) () on a tract of land donated by Yulia Martinovna Plavskaya.The initial chapel, dedicated to the memory of the great prince St. Alexander Nevsky (1220–1263), was blessed in May, 1936.The church building was subsequently expanded three times.In 1987, ground was cleared for the construction of the new church and on September 12, 1989, on the Feast Day of St. Alexander Nevsky, the cornerstone was laid and the relics of St. Herman of Alaska placed in the foundation.The imposing edifice, completed in 1997, is the work of Nikolaus Karsanov, architect and Protopresbyter Valery Lukianov, engineer.Funds were raised through donations.The Great blessing of the cathedral took place on October 18, 1997 with seven bishops, headed by Metropolitan Vitaly Ustinov, and 36 priests and deacons officiating, some 800 faithful attended the festivity.The old church was rededicated to Our Lady of Tikhvin.Metropolitan Hilarion (Kapral) announced, that cathedral will officially become the episcopal See of the Ruling Bishop of the Eastern American Diocese and the administrative center of the Diocese on September 12, 2014.At present the parish serves the spiritual needs of 300 members.The parochial school instructs over 90 boys and girls in religion, Russian language and history.The school meets every Saturday.The choir is directed by Andrew Burbelo.The sisterhood attends to the needs of the church and a church council acts in the administration of the community.The cathedral is decorated by frescoes in the Byzantine style.The iconography project was fulfilled by Father Andrew Erastov and his students from 1995 until 2001.'  # noqa *

    def run_pipeline(self, model_id: str, documents: str) -> Dict[str, Any]:
        p = pipeline(task=self.task, model=model_id)
        result = p(documents=documents)
        return result

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_document_segmentation(self):
        logger.info('Run document segmentation (Bert) with one document ...')

        result = self.run_pipeline(
            model_id=self.bert_ds_model_id, documents=self.sentences)
        print(result[OutputKeys.TEXT])

        result = self.run_pipeline(
            model_id=self.bert_ds_eng_model_id, documents=self.eng_sentences)
        print(result[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_topic_segmentation(self):
        logger.info('Run topic segmentation (PoNet) with one document ...')

        result = self.run_pipeline(
            model_id=self.ponet_ts_model_id, documents=self.sentences)
        # print("return:")
        print(result[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_documents_segmentation(self):
        logger.info('Run document segmentation (Bert) with many documents ...')

        result = self.run_pipeline(
            model_id=self.bert_ds_model_id,
            documents=[self.sentences, self.sentences_1])

        documents_list = result[OutputKeys.TEXT]
        for document in documents_list:
            print(document)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
