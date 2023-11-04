# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from transformers import BloomConfig
from transformers import BloomModel as BloomModelTransform

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.outputs import SentencEmbeddingModelOutput
from modelscope.utils.constant import Tasks


class DecoderPooler(torch.nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'last': the last token state.
    'weighted_mean': position weighted average of all token states.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            'last', 'weighted_mean'
        ], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state

        if self.pooler_type in ['last']:
            n, l, h = last_hidden.shape

            # Get shape [n] indices of the last token (i.e. the last token for each batch item)
            # Any sequence where min == 1, we use the entire sequence lenth since argmin = 0
            values, indices = torch.min(attention_mask, 1, keepdim=False)
            gather_indices = torch.where(values == 0, indices,
                                         l) - 1  # Shape [n]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [n] --> [n, 1, h]
            gather_indices = gather_indices.unsqueeze(1).unsqueeze(1).expand(
                n, 1, h)

            # Gather along the 1st dim (l) (n, l, h -> n, h)
            pooled_output = torch.gather(last_hidden, 1,
                                         gather_indices).squeeze(dim=1)

        elif self.pooler_type == 'weighted_mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                last_hidden.size()).float()
            # last_hidden shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=last_hidden.shape[1]
                             + 1).unsqueeze(0).unsqueeze(-1).expand(
                                 last_hidden.size()).float().to(
                                     last_hidden.device))
            assert weights.shape == last_hidden.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        else:
            raise NotImplementedError

        return pooled_output


@MODELS.register_module(
    group_key=Tasks.sentence_embedding, module_name=Models.bloom)
class BloomForSentenceEmbedding(BloomModelTransform, TorchModel):
    r"""
    This model represent a text to a dense vector by the last token state or weighted mean of all token states.
    See `Language Models are Universal Embedders
    <https://arxiv.org/pdf/2310.08232.pdf>`_ for details.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.pooler_type = kwargs.get('emb_pooler_type', 'weighted_mean')
        self.pooler = DecoderPooler(self.pooler_type)
        self.normalize = kwargs.get('normalize', False)
        setattr(self, self.base_model_prefix, BloomModelTransform(config))

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
            >>> model = Model.from_pretrained('damo/nlp_udever_bloom_560m')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_udever_bloom_560m')
            >>> inputs = preprocessor({'source_sentence': ['This is a test']})
            >>> outputs = model(**inputs)
            >>> print(outputs)
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
            loss_fct = torch.nn.CrossEntropyLoss()
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
    ):
        outputs = self.base_model.forward(
            input_ids, attention_mask=attention_mask)
        embeddings = self.pooler(outputs, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

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
            'emb_pooler_type': kwargs.get('emb_pooler_type', 'weighted_mean'),
            'normalize': kwargs.get('normalize', False)
        }
        if model_dir is None:
            config = BloomConfig(**kwargs)
            model = cls(config)
        else:
            model = super(BloomModelTransform, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_kwargs)
        model.model_dir = model_dir
        return model
