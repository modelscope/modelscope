# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.builder import MODELS
from modelscope.outputs import SentencEmbeddingModelOutput
from modelscope.utils.constant import Tasks
from .backbone import BertModel, BertPreTrainedModel


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            'cls', 'avg', 'avg_top2', 'avg_first_last'
        ], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1)
                    / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0
                             * attention_mask.unsqueeze(-1)
                             ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0
                             * attention_mask.unsqueeze(-1)
                             ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


@MODELS.register_module(Tasks.sentence_embedding, module_name=Models.bert)
class BertForSentenceEmbedding(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.pooler_type = kwargs.get('emb_pooler_type', 'cls')
        self.pooler = Pooler(self.pooler_type)
        self.normalize = kwargs.get('normalize', False)
        setattr(self, self.base_model_prefix,
                BertModel(config, add_pooling_layer=False))

    def forward(self, query=None, docs=None, labels=None):
        r"""
        Args:
            query (:obj: `dict`): Dict of pretrained models's input for the query sequence. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
            docs (:obj: `dict`): Dict of pretrained models's input for the query sequence. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
        Returns:
            Returns `modelscope.outputs.SentencEmbeddingModelOutput
        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_corom_sentence-embedding_chinese-base')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_corom_sentence-embedding_chinese-base')
            >>> print(model(**preprocessor('source_sentence':['This is a test'])))
        """
        query_embeddings, doc_embeddings = None, None
        if query is not None:
            query_embeddings = self.encode(**query)
        if docs is not None:
            doc_embeddings = self.encode(**docs)
        outputs = SentencEmbeddingModelOutput(
            query_embeddings=query_embeddings, doc_embeddings=doc_embeddings)
        if query_embeddings is None or doc_embeddings is None:
            return outputs
        if self.base_model.training:
            loss_fct = nn.CrossEntropyLoss()
            scores = torch.matmul(query_embeddings, doc_embeddings.T)
            if labels is None:
                labels = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long)
                labels = labels * (
                    doc_embeddings.size(0) // query_embeddings.size(0))
            loss = loss_fct(scores, labels)
            outputs.loss = loss
        return outputs

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.base_model.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        outputs = self.pooler(outputs, attention_mask)
        if self.normalize:
            outputs = F.normalize(outputs, p=2, dim=-1)
        return outputs

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        model_dir = kwargs.get('model_dir')
        model_kwargs = {
            'emb_pooler_type': kwargs.get('emb_pooler_type', 'cls'),
            'normalize': kwargs.get('normalize', False)
        }
        model = super(Model, cls).from_pretrained(
            pretrained_model_name_or_path=model_dir, **model_kwargs)
        model.model_dir = model_dir
        return model
