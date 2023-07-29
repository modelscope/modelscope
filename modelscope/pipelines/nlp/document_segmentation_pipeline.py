# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import Dataset

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import \
    DocumentSegmentationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['DocumentSegmentationPipeline']


@PIPELINES.register_module(
    Tasks.document_segmentation, module_name=Pipelines.document_segmentation)
class DocumentSegmentationPipeline(Pipeline):

    def __init__(
            self,
            model: Union[Model, str],
            preprocessor: DocumentSegmentationTransformersPreprocessor = None,
            config_file: str = None,
            device: str = 'gpu',
            auto_collate=True,
            **kwargs):
        """The document segmentation pipeline.

        Args:
            model (str or Model): Supply either a local model dir or a model id from the model hub
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            **kwargs)

        kwargs.pop('compile', None)
        kwargs.pop('compile_options', None)

        self.model_dir = self.model.model_dir
        self.model_cfg = self.model.model_cfg
        if preprocessor is None:
            self.preprocessor = DocumentSegmentationTransformersPreprocessor(
                self.model_dir, self.model.config.max_position_embeddings,
                **kwargs)

    def __call__(
            self, documents: Union[List[List[str]], List[str],
                                   str]) -> Dict[str, Any]:
        output = self.predict(documents)
        output = self.postprocess(output)
        return output

    def predict(
            self, documents: Union[List[List[str]], List[str],
                                   str]) -> Dict[str, Any]:
        pred_samples = self.cut_documents(documents)

        if self.model_cfg['level'] == 'topic':
            paragraphs = pred_samples.pop('paragraphs')

        predict_examples = Dataset.from_dict(pred_samples)

        # Predict Feature Creation
        predict_dataset = self.preprocessor(predict_examples, self.model_cfg)
        num_examples = len(
            predict_examples[self.preprocessor.context_column_name])
        num_samples = len(
            predict_dataset[self.preprocessor.context_column_name])

        if self.model_cfg['type'] == 'bert':
            predict_dataset.pop('segment_ids')

        labels = predict_dataset.pop('labels')
        sentences = predict_dataset.pop('sentences')
        example_ids = predict_dataset.pop(
            self.preprocessor.example_id_column_name)

        if (self.model or (self.has_multiple_models and self.models[0])):
            if not self._model_prepare:
                self.prepare_model()

        with torch.no_grad():
            input = {
                key: torch.tensor(val).to(self.device)
                for key, val in predict_dataset.items()
            }
            predictions = self.model.forward(**input).logits.cpu()

        predictions = np.argmax(predictions, axis=2)
        assert len(sentences) == len(
            predictions), 'sample {}  infer_sample {} prediction {}'.format(
                num_samples, len(sentences), len(predictions))
        # Remove ignored index (special tokens)

        true_predictions = [
            [
                self.preprocessor.label_list[p]
                for (p, l) in zip(prediction, label) if l != -100  # noqa *
            ] for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [
                self.preprocessor.label_list[l]
                for (p, l) in zip(prediction, label) if l != -100  # noqa *
            ] for prediction, label in zip(predictions, labels)
        ]

        # Save predictions
        out = []
        for i in range(num_examples):
            if self.model_cfg['level'] == 'topic':
                out.append({
                    'sentences': [],
                    'labels': [],
                    'predictions': [],
                    'paragraphs': paragraphs[i]
                })
            else:
                out.append({'sentences': [], 'labels': [], 'predictions': []})

        for prediction, sentence_list, label, example_id in zip(
                true_predictions, sentences, true_labels, example_ids):
            if self.model_cfg['level'] == 'doc':
                if len(label) < len(sentence_list):
                    label.append('B-EOP')
                    prediction.append('B-EOP')
                assert len(sentence_list) == len(prediction), '{} {}'.format(
                    len(sentence_list), len(prediction))
                assert len(sentence_list) == len(label), '{} {}'.format(
                    len(sentence_list), len(label))

            out[example_id]['sentences'].extend(sentence_list)
            out[example_id]['labels'].extend(label)
            out[example_id]['predictions'].extend(prediction)

        if self.model_cfg['level'] == 'topic':
            for i in range(num_examples):
                assert len(out[i]['predictions']) + 1 == len(
                    out[i]['paragraphs'])
                out[i]['predictions'].append('B-EOP')
                out[i]['labels'].append('B-EOP')

        return out

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        result = []
        res_preds = []
        list_count = len(inputs)

        if self.model_cfg['level'] == 'topic':
            for num in range(list_count):
                res = []
                pred = []
                for s, p, l in zip(inputs[num]['paragraphs'],
                                   inputs[num]['predictions'],
                                   inputs[num]['labels']):
                    s = s.strip()
                    if p == 'B-EOP':
                        s = ''.join([s, '\n\n\t'])
                        pred.append(1)
                    else:
                        s = ''.join([s, '\n\t'])
                        pred.append(0)
                    res.append(s)
                res_preds.append(pred)
                document = ('\t' + ''.join(res).strip())
                result.append(document)
        else:
            for num in range(list_count):
                res = []
                for s, p in zip(inputs[num]['sentences'],
                                inputs[num]['predictions']):
                    s = s.strip()
                    if p == 'B-EOP':
                        s = ''.join([s, '\n\t'])
                    res.append(s)

                document = ('\t' + ''.join(res))
                result.append(document)

        if list_count == 1:
            return {OutputKeys.TEXT: result[0]}
        else:
            return {OutputKeys.TEXT: result}

    def cut_documents(self, para: Union[List[List[str]], List[str], str]):
        document_list = para
        paragraphs = []
        sentences = []
        labels = []
        example_id = []
        id = 0

        if self.model_cfg['level'] == 'topic':
            if isinstance(para, str):
                document_list = [[para]]
            elif isinstance(para[0], str):
                document_list = [para]

            for document in document_list:
                sentence = []
                label = []
                for item in document:
                    sentence_of_current_paragraph = self.cut_sentence(item)
                    sentence.extend(sentence_of_current_paragraph)
                    label.extend(['-100']
                                 * (len(sentence_of_current_paragraph) - 1)
                                 + ['B-EOP'])
                paragraphs.append(document)
                sentences.append(sentence)
                labels.append(label)
                example_id.append(id)
                id += 1

            return {
                'example_id': example_id,
                'sentences': sentences,
                'paragraphs': paragraphs,
                'labels': labels
            }
        else:
            if isinstance(para, str):
                document_list = [para]

            for document in document_list:
                sentence = self.cut_sentence(document)
                label = ['O'] * (len(sentence) - 1) + ['B-EOP']
                sentences.append(sentence)
                labels.append(label)
                example_id.append(id)
                id += 1

            return {
                'example_id': example_id,
                'sentences': sentences,
                'labels': labels
            }

    def cut_sentence(self, para):
        para = re.sub(r'([。！.!？\?])([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)  # noqa *
        para = para.rstrip()
        return [_ for _ in para.split('\n') if _]
