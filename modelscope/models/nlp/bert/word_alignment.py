# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import WordAlignmentOutput 

from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import BertModel, BertPreTrainedModel

logger = logging.get_logger()

# mbert model
@MODELS.register_module(Tasks.word_alignment, module_name=Models.bert) 
class MBertForWordAlignment(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        config.num_hidden_layers = kwargs.get("encoder_layers", 8)
        
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        src_input_ids=None,
        src_attention_mask=None,
        src_b2w_map=None,
        tgt_input_ids=None,
        tgt_attention_mask=None,
        tgt_b2w_map=None,
        threshold=0.001, 
        bpe_level=False,
    ):
        with torch.no_grad():
            src_encoder_out = self.bert(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )
            tgt_encoder_out = self.bert(
                input_ids=tgt_input_ids,
                attention_mask=tgt_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )
            
            atten_mask_src = (1 - ((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask)[:, None, None, :].float()) * -10000
            atten_mask_tgt = (1 - ((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask)[:, None, None, :].float()) * -10000

            src_align_out = src_encoder_out[0]
            tgt_align_out = tgt_encoder_out[0]

            bpe_sim = torch.bmm(src_align_out, tgt_align_out.transpose(1,2))
        
        attention_scores_src = bpe_sim.unsqueeze(1) + atten_mask_tgt
        attention_scores_tgt = bpe_sim.unsqueeze(1) + atten_mask_src.transpose(-1, -2)
        
        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src)
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)

        align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)
        align_matrix = align_matrix.squeeze(1)

        len_src = (atten_mask_src==0).sum(dim=-1).unsqueeze(-1)
        len_tgt = (atten_mask_tgt==0).sum(dim=-1).unsqueeze(-1)
           
        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src / torch.sqrt(len_src.float()))
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt / torch.sqrt(len_tgt.float()))

        word_aligns = []
        
        for idx, (line_align, b2w_src, b2w_tgt) in enumerate(zip(align_matrix, src_b2w_map, tgt_b2w_map)):
            aligns = dict()
            non_specials = torch.where(line_align)
            for i, j in zip(*non_specials):
                if not bpe_level:
                    word_pair = (src_b2w_map[idx][i-1].item(), tgt_b2w_map[idx][j-1].item())
                    if not word_pair in aligns:
                        aligns[word_pair] = bpe_sim[idx][i, j].item()
                    else:
                        aligns[word_pair] = max(aligns[word_pair], bpe_sim[idx][i, j].item())
                else:
                    aligns[(i.item()-1, j.item()-1)] = bpe_sim[idx][i, j].item()
            word_aligns.append(aligns)
            
        return WordAlignmentOutput(predictions=word_aligns)
