import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM


class TextTransformer(nn.Module):

    def __init__(self, config_dict, feat_dim=768, use_grad_ckp=True):
        super(TextTransformer, self).__init__()
        bert_config = BertConfig.from_dict(config_dict)
        if use_grad_ckp:
            bert_config.gradient_checkpointing = True

        self.bert = BertForMaskedLM(bert_config).bert

        self.projector = nn.Linear(
            bert_config.hidden_size, feat_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        trans_features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        output_states = self.bert(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]

        return self.projector(cls_tokens)
