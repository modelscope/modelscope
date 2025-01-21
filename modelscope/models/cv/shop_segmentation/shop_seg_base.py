# Base modules are adapted from https://github.com/open-mmlab/mmcv/,
# originally Apache 2.0 License, Copyright (c) 2018-2022 OpenMMLab,
# https://github.com/open-mmlab/mmsegmentation/,
# originally Apache 2.0 License, Copyright (c) 2020-2021 OpenMMLab,
# and adapted from https://github.com/raoyongming/DenseCLIP/,
# originally MIT License, Copyright (c) 2022 Rao, Yongming.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_fpn import FPNHead
from .models import (CLIPTextContextEncoder, CLIPVisionTransformer,
                     ContextDecoder)
from .neck_fpn import FPN
from .utils import SimpleTokenizer, tokenize


class SHOPSEG(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 model_dir,
                 context_length=22,
                 context_feature='attention',
                 score_concat_index=2,
                 tau=0.07,
                 token_embed_dim=512,
                 text_dim=512,
                 **args):
        super(SHOPSEG, self).__init__()

        self.model_dir = model_dir
        self.tokenizer = SimpleTokenizer(model_dir
                                         + '/bpe_simple_vocab_16e6.txt.gz')

        backbone = CLIPVisionTransformer(
            input_resolution=1024,
            patch_size=16,
            width=768,
            layers=12,
            output_dim=512,
            drop_path_rate=0.1,
            pretrained=False,
            get_embeddings=True)

        text_encoder = CLIPTextContextEncoder(
            context_length=30,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            embed_dim=512,
            pretrained=False)

        context_decoder = ContextDecoder(
            transformer_width=256,
            transformer_heads=4,
            transformer_layers=3,
            visual_dim=512,
            dropout=0.1)
        neck = FPN(
            in_channels=[768, 768, 768 + 2, 768], out_channels=256, num_outs=4)
        head_fpd = FPNHead(channels=256, num_classes=2)

        self.backbone = backbone
        self.text_encoder = text_encoder
        self.context_decoder = context_decoder
        self.context_length = context_length
        self.score_concat_index = score_concat_index

        self.context_feature = context_feature
        self.tau = tau
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(
            torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        self.neck = neck
        self.head_fpn = head_fpd

        self.tau = 0.07

    def encode_text(self, text, context_length):
        output = tokenize(self.tokenizer, text, context_length, True)
        return output

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def after_extract_feat(self, x, name_list):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            x1 = global_feat.reshape(B, C, 1)
            x2 = visual_embeddings.reshape(B, C, H * W)
            visual_context = torch.cat([x1, x2], dim=2).permute(0, 2, 1)
        texts = torch.cat([
            self.encode_text(c, context_length=self.context_length)
            for c in name_list
        ])
        x1 = texts.to(global_feat.device)
        x1 = self.text_encoder(x1, self.contexts)
        text_embeddings = x1.expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map_list = []
        bsz = B
        for i in range(bsz):
            ind = 2 * i
            sub_text = torch.cat(
                [text[i:i + 1, ind:ind + 1], text[i:i + 1, ind + 1:ind + 2]],
                dim=1)  # 1 * 2 * h * w

            sub_score_map = torch.einsum('bchw,bkc->bkhw',
                                         visual_embeddings[i:i + 1],
                                         sub_text)  # 1 * 2 * h * w
            score_map_list.append(sub_score_map)
        score_map = torch.cat(score_map_list, dim=0)  # b * 2 * h * w
        x_orig[self.score_concat_index] = torch.cat(
            [x_orig[self.score_concat_index], score_map], dim=1)
        return x_orig, score_map

    def forward(self, img, text_list=None):
        if text_list is None:
            bsz = img.size()[0]
            text_list = ['foregeound'] * bsz
        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(4)]
        name_list = []
        for name in text_list:
            name_list.append('others')
            name_list.append(name[0:20])
        x_orig, score_map = self.after_extract_feat(x, name_list)
        x_orig = list(self.neck(x_orig))
        _x_orig = x_orig
        pred = self.head_fpn(_x_orig)
        return pred
