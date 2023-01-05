# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.logger import get_logger

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_segmentation)
class DocumentSegmentationTransformersPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 model_max_length: int,
                 mode: str = ModeKeys.INFERENCE,
                 question_column_name='labels',
                 context_column_name='sentences',
                 example_id_column_name='example_id',
                 label_list=['B-EOP', 'O']):
        """The preprocessor for document segmentation task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
            model_max_length: The max length the model supported.
            mode: The mode for this preprocessor.
            question_column_name: The key for the question column, default `labels`.
            context_column_name: The key for the context column, default `sentences`.
            example_id_column_name: The key for the example id column, default `example_id`.
            label_list: The label list, default `['B-EOP', 'O']`
        """

        super().__init__(mode)
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir, )
        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.example_id_column_name = example_id_column_name
        self.label_list = label_list
        self.label_to_id = {
            label: id
            for id, label in enumerate(self.label_list)
        }
        self.target_specical_ids = set()
        self.target_specical_ids.add(self.tokenizer.eos_token_id)
        self.max_seq_length = model_max_length

    def __call__(self, examples, model_cfg=None) -> Dict[str, Any]:
        questions = examples[self.question_column_name]
        contexts = examples[self.context_column_name]
        example_ids = examples[self.example_id_column_name]
        num_examples = len(questions)

        sentences = []
        for sentence_list in contexts:
            sentence_list = [_ + '[EOS]' for _ in sentence_list]
            sentences.append(sentence_list)

        try:
            tokenized_examples = self.tokenizer(
                sentences,
                is_split_into_words=True,
                add_special_tokens=False,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
        except Exception as e:
            logger.error(e)
            return {}

        segment_ids = []
        token_seq_labels = []
        for example_index in range(num_examples):
            example_input_ids = tokenized_examples['input_ids'][example_index]
            example_labels = questions[example_index]
            example_labels = [
                self.label_to_id[_] if _ in self.label_to_id else -100
                for _ in example_labels
            ]
            example_token_labels = []
            segment_id = []
            cur_seg_id = 1
            para_segment_id = []
            cut_para_seg_id = 1
            for token_index in range(len(example_input_ids)):
                if example_input_ids[token_index] in self.target_specical_ids:
                    example_token_labels.append(example_labels[cur_seg_id - 1])
                    segment_id.append(cur_seg_id)
                    cur_seg_id += 1
                else:
                    example_token_labels.append(-100)
                    segment_id.append(cur_seg_id)

                if example_token_labels[token_index] != -100:
                    para_segment_id.append(cut_para_seg_id)
                    cut_para_seg_id += 1
                else:
                    para_segment_id.append(cut_para_seg_id)

            if model_cfg is not None and model_cfg[
                    'type'] == 'ponet' and model_cfg['level'] == 'topic':
                segment_ids.append(para_segment_id)
            else:
                segment_ids.append(segment_id)
            token_seq_labels.append(example_token_labels)

        tokenized_examples['segment_ids'] = segment_ids
        tokenized_examples['token_seq_labels'] = token_seq_labels

        new_segment_ids = []
        new_token_seq_labels = []
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []
        new_example_ids = []
        new_sentences = []

        for example_index in range(num_examples):
            example_input_ids = tokenized_examples['input_ids'][example_index]
            example_token_type_ids = tokenized_examples['token_type_ids'][
                example_index]
            example_attention_mask = tokenized_examples['attention_mask'][
                example_index]
            example_segment_ids = tokenized_examples['segment_ids'][
                example_index]
            example_token_seq_labels = tokenized_examples['token_seq_labels'][
                example_index]
            example_sentences = contexts[example_index]
            example_id = example_ids[example_index]
            example_total_num_sentences = len(questions[example_index])
            example_total_num_tokens = len(
                tokenized_examples['input_ids'][example_index])
            accumulate_length = [
                i for i, x in enumerate(tokenized_examples['input_ids']
                                        [example_index])
                if x == self.tokenizer.eos_token_id
            ]
            samples_boundary = []
            left_index = 0
            sent_left_index = 0
            sent_i = 0

            # for sent_i, length in enumerate(accumulate_length):
            while sent_i < len(accumulate_length):
                length = accumulate_length[sent_i]
                right_index = length + 1
                sent_right_index = sent_i + 1
                if right_index - left_index >= self.max_seq_length - 1 or right_index == example_total_num_tokens:
                    samples_boundary.append([left_index, right_index])

                    sample_input_ids = [
                        self.tokenizer.cls_token_id
                    ] + example_input_ids[left_index:right_index]
                    sample_input_ids = sample_input_ids[:self.max_seq_length]

                    sample_token_type_ids = [
                        0
                    ] + example_token_type_ids[left_index:right_index]
                    sample_token_type_ids = sample_token_type_ids[:self.
                                                                  max_seq_length]

                    sample_attention_mask = [
                        1
                    ] + example_attention_mask[left_index:right_index]
                    sample_attention_mask = sample_attention_mask[:self.
                                                                  max_seq_length]

                    sample_segment_ids = [
                        0
                    ] + example_segment_ids[left_index:right_index]
                    sample_segment_ids = sample_segment_ids[:self.
                                                            max_seq_length]

                    sample_token_seq_labels = [
                        -100
                    ] + example_token_seq_labels[left_index:right_index]
                    sample_token_seq_labels = sample_token_seq_labels[:self.
                                                                      max_seq_length]

                    if sent_right_index - 1 == sent_left_index:
                        left_index = right_index
                        sample_input_ids[-1] = self.tokenizer.eos_token_id
                        sample_token_seq_labels[-1] = -100
                    else:
                        left_index = accumulate_length[sent_i - 1] + 1
                        if sample_token_seq_labels[-1] != -100:
                            sample_token_seq_labels[-1] = -100

                    if sent_right_index - 1 == sent_left_index or right_index == example_total_num_tokens:
                        sample_sentences = example_sentences[
                            sent_left_index:sent_right_index]
                        sent_left_index = sent_right_index
                        sent_i += 1
                    else:
                        sample_sentences = example_sentences[
                            sent_left_index:sent_right_index - 1]
                        sent_left_index = sent_right_index - 1

                    if (len([_ for _ in sample_token_seq_labels if _ != -100
                             ])) != len(sample_sentences) - 1 and (len([
                                 _
                                 for _ in sample_token_seq_labels if _ != -100
                             ])) != len(sample_sentences):
                        tmp = []
                        for w_i, w, l in zip(
                                sample_input_ids,
                                self.tokenizer.decode(sample_input_ids).split(
                                    ' '), sample_token_seq_labels):
                            tmp.append((w_i, w, l))
                    while len(sample_input_ids) < self.max_seq_length:
                        sample_input_ids.append(self.tokenizer.pad_token_id)
                        sample_token_type_ids.append(0)
                        sample_attention_mask.append(0)
                        sample_segment_ids.append(example_total_num_sentences
                                                  + 1)
                        sample_token_seq_labels.append(-100)

                    new_input_ids.append(sample_input_ids)
                    new_token_type_ids.append(sample_token_type_ids)
                    new_attention_mask.append(sample_attention_mask)
                    new_segment_ids.append(sample_segment_ids)
                    new_token_seq_labels.append(sample_token_seq_labels)
                    new_example_ids.append(example_id)
                    new_sentences.append(sample_sentences)
                else:
                    sent_i += 1
                    continue

        output_samples = {}
        output_samples['input_ids'] = new_input_ids
        output_samples['token_type_ids'] = new_token_type_ids
        output_samples['attention_mask'] = new_attention_mask

        output_samples['segment_ids'] = new_segment_ids
        output_samples['example_id'] = new_example_ids
        output_samples['labels'] = new_token_seq_labels
        output_samples['sentences'] = new_sentences

        return output_samples
