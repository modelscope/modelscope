# Copyright (c) 2022 Zhipu.AI
"""Sample Generate GPT2"""

import argparse
import copy
import os
import random
import time
from datetime import datetime

import deepspeed
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from pypinyin import FINALS, FINALS_TONE, TONE3, pinyin

from .arguments import get_args

from .gpt2 import mpu
from .gpt2.configure_data import configure_data
from .gpt2.data_utils import make_tokenizer
from .gpt2.fp16 import FP16_Module
from .gpt2.model import DistributedDataParallel as DDP
from .gpt2.model import GPT2Model
from .gpt2.utils import (Timers, get_checkpoint_iteration, load_checkpoint,
                         print_rank_0)

open_old_pronounce = 1

class APIException(Exception):

    def __init__(self, message):
        super().__init__(message)
        

class CanNotReturnException(APIException):

    def __init__(self, message, payload=None):
        self.payload = payload
        super().__init__(message)

class InputTooLongException(APIException):

    def __init__(self, message, payload=None):
        self.payload = payload
        super().__init__(message)

def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        max_memory_length=args.mem_length,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers,
        parallel_output=True,
        relative_encoding=args.transformer_xl)

    if mpu.get_data_parallel_rank() == 0:
        print(
            ' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])),
            flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if hasattr(args, 'deepspeed') and args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if not args.deepspeed:
        if USE_TORCH_DDP:
            i = torch.cuda.current_device()
            model = DDP(
                model,
                device_ids=[i],
                output_device=i,
                process_group=mpu.get_data_parallel_group())
        else:
            model = DDP(model)

    return model


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               transformer_xl=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if transformer_xl:
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(
            torch.triu(attention_mask, 1 - seq_length + mem_length),
            mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones(
                (att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(
            data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    # master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '6001')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(
            args, 'deepspeed'
    ) and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            print(args.load)
            path = os.path.join(args.load, 'mp_rank_00_model_states.pt')
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['module'])
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


rus = set([
    '八', '搭', '塌', '邋', '插', '察', '杀', '煞', '夹', '俠', '瞎', '辖', '狹', '匣', '黠',
    '鸭', '押', '压', '刷', '刮', '滑', '猾', '挖', '蜇', '舌', '鸽', '割', '胳', '搁', '瞌',
    '喝', '合', '盒', '盍', '曷', '貉', '涸', '劾', '核', '钵', '剝', '泼', '摸', '脱', '托',
    '捋', '撮', '缩', '豁', '活', '切', '噎', '汁', '织', '隻', '掷', '湿', '虱', '失', '十',
    '什', '拾', '实', '食', '蝕', '识', '石', '劈', '霹', '滴', '踢', '剔', '屐', '积', '激',
    '击', '漆', '吸', '息', '媳', '昔', '席', '锡', '檄', '觋', '揖', '一', '壹', '扑', '匍',
    '仆', '弗', '紱', '拂', '福', '蝠', '幅', '辐', '服', '伏', '茯', '督', '突', '秃', '俗',
    '出', '蜀', '窟', '哭', '忽', '惚', '斛', '鹄', '屋', '屈', '诎', '曲', '戌', '拍', '塞',
    '摘', '拆', '黑', '勺', '芍', '嚼', '粥', '妯', '熟', '白', '柏', '伯', '薄', '剥', '摸',
    '粥', '轴', '舳', '妯', '熟', '角', '削', '学'
])
ss = set([
    'de', 'te', 'le', 'ze', 'ce', 'se', 'fa', 'fo', 'dei', 'zei', 'gei', 'hei',
    'sei', 'bie', 'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'kuo', 'zhuo',
    'chuo', 'shuo', 'ruo'
])


def checkpz(st, wd):

    if not (st[-1] in ['1', '2', '3', '4']):
        return 0

    if open_old_pronounce == 1:
        if wd in rus:
            return 2
        if wd in ['嗟', '瘸', '靴', '爹']:
            return 1
        if st[:-1] in ss:
            return 2

        if (st[-1] == '2' and st[0] in ['b', 'd', 'g', 'j', 'z']):
            return 2
        if 'ue' in st:
            return 2

    if st[-1] in ['1', '2']:
        return 1

    return 2


# inner rhy, must obey
def checkrhyself(sentence):
    if len(sentence) == 0:
        return 0
    st = sentence
    fullst = False
    while (len(st) > 0 and st[-1] in [',', '。', '，', '?', '？', '!', '！']):
        st = st[:-1]
        fullst = True

    l1 = pinyin(st, style=TONE3)
    if len(l1) < len(st):
        return 1
    for i in l1:
        if len(i[0]) < 2:
            return 1
    if len(st) <= 3:
        return 2

    pz1 = checkpz(l1[1][0], sentence[1])

    if len(st) >= 4:
        pz2 = checkpz(l1[3][0], sentence[3])
        if pz2 + pz1 != 3:
            return 1
    if len(st) >= 6:
        pz3 = checkpz(l1[5][0], sentence[5])
        if pz2 + pz3 != 3:
            return 1
    if fullst:
        if len(sentence) < 6:
            return 1
        pz11 = checkpz(l1[-3][0], st[-3])
        pz12 = checkpz(l1[-2][0], st[-2])
        pz13 = checkpz(l1[-1][0], st[-1])
        if (pz11 == pz12) and (pz12 == pz13):
            return 1

    return 2


def checkrhy(sentence, last, imp, req=0):

    while (len(sentence) > 0
           and (sentence[-1] in [',', '。', '，', '?', '？', '!', '！'])):
        sentence = sentence[:-1]
    if len(sentence) == 0:
        return 0

    while last[-1] in [',', '。', '，', '?', '？', '!', '！']:
        last = last[:-1]
    l1 = pinyin(sentence, style=TONE3)
    l2 = pinyin(last, style=TONE3)
    disobey = 0
    if len(l1) != len(sentence):
        return -1000
    for i in range(len(sentence)):
        if (i < len(l1)) and (i < len(l2)):
            st1 = checkpz(l1[i][0], sentence[i])

            sr1 = checkpz(l2[i][0], last[i])
            if (req == 1 and i % 2 == 1):
                st1 = 3 - st1

            if st1 + sr1 != 3:
                if req == 0:
                    disobey += 0.35
                if i % 2 == 1:
                    disobey += 0.35
                    if req == 1:
                        disobey += 0.2
                if i == len(l2) - 1:
                    disobey += 0.65
                    if req == 1:
                        disobey += 0.35

    disobey *= imp
    disobey = -5 * disobey / len(l2)
    for i in range(len(l1)):
        for j in range(i + 2, len(l1)):
            if l1[i][0][:-1] == l1[j][0][:-1]:
                disobey -= 7 / len(l1)
    return disobey


def checksentence(sentence,
                  original_context,
                  min_length,
                  max_length,
                  endnote,
                  curvote=0,
                  yayun=None):

    if '<|end' in sentence:
        return 1

    if '的' in sentence:
        return 1
    if len(sentence) == 0:
        return 1
    if ((len(sentence) > max_length and not (sentence[-1] in endnote))
            or len(sentence) == 0) or len(sentence) > max_length + 1:
        return 1
    if (sentence[-1] in endnote) and ((len(sentence) <= min_length) or  # noqa
                                      (len(sentence) == 7)):  # noqa
        return 1

    if (sentence[-1] in endnote) and (sentence[:-1] in original_context):
        return 1

    mdisobey = 0  # noqa
    illegal_notes = [
        ' ', ':', '《', '》', '‘', '“', '-', '——', '⁇', '[', '【', '】', ']', '.',
        '、', '(', '（', ')', '）', '·'
    ]
    if '。' in endnote:
        illegal_notes.extend([',', '，'])
    else:
        illegal_notes.append('。')
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64, 123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
    last = getlastsentence(original_context)
    if min_length == max_length:
        imp = 1
        if (',' in last) or ('，' in last):
            imp = 1.5

        if curvote == 0:
            rt = checkrhy(sentence, last, imp, req=1)
        else:
            rt = checkrhy(sentence, last, imp)
        if rt < -0.75:
            return 1

    for i in range(len(sentence)):
        if min_length == max_length:
            if (i < len(last) - 1) and (sentence[i] == last[i]):
                return 1

        if i < len(sentence) - 1:
            if sentence[i:i + 2] in original_context:
                return 1
            if sentence[i:i + 2] in sentence[:i]:
                return 1

    if checkrhyself(sentence) == 1:
        return 1
    cc = curvote
    if yayun is None:
        cc = 0
    if (cc == 1 and len(sentence) >= max_length):

        final1 = pinyin(sentence, style=FINALS)
        if len(final1) < max_length:
            return 1
        final1 = final1[max_length - 1][0]
        final2 = pinyin(yayun, style=FINALS)[-1][0]
        group = [['a', 'ia', 'ua'], ['ai', 'uai', 'ei', 'ui', 'uei'],
                 ['an', 'uan', 'ian'], ['ie', 'ue', 've'], ['ou', 'iu', 'iou'],
                 ['ang', 'iang', 'uang'], ['ao', 'iao'], ['e', 'o', 'uo'],
                 ['en', 'un', 'uen', 'ong', 'iong', 'in', 'ing', 'er']]
        doc = 0
        if final1 == final2:
            doc = 1
        for i in group:
            if (final1 in i) and (final2 in i):
                doc = 1
        if doc == 0:
            return 1

    if (sentence[-1] in endnote):
        return 0

    return 2


def generate_sentence(model,
                      tokenizer,
                      args,
                      device,
                      current_tokens,
                      mems,
                      endnote=[',', '，', '?', '？'],
                      num_candidates=1,
                      min_length=5,
                      max_length=7,
                      yayun=None):
    model.eval()
    with torch.no_grad():
        mct_tree = []
        if mems == []:
            mems = []
            tokens, attention_mask, position_ids = get_batch(
                current_tokens, device, args)
            logits, *rts = model(tokens, position_ids, attention_mask, *mems)
        else:
            tokens = current_tokens
            index = len(tokens[0])
            logits, *rts = model(
                tokens[:, index - 1:index],
                tokens.new_ones((1, 1)) * (index - 1),
                tokens.new_ones(
                    1,
                    1,
                    1,
                    args.mem_length + 1,
                    device=tokens.device,
                    dtype=torch.float), *mems)

        output_tokens_list = tokens.view(-1).contiguous()
        original_context = tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length = len(tokens[0])
        logits = logits[0, -1]
        mct_tree.append([
            logits, rts, tokens, -np.ones(len(logits)),
            torch.ones(len(logits)).cuda(), 0
        ])
        final_result = []
        nextid = 0
        tries = 0
        max_tries = num_candidates * 30
        curvote = 1
        if ',' in endnote:
            curvote = 0
        if ',' in endnote:
            endid = 43359
        else:
            endid = 43361
        dpcount = 0

        tmp = args.temperature

        while ((len(final_result) < num_candidates) and (tries < max_tries)
               and (tries < 1000)):
            currentid = nextid
            tries += 1
            while currentid != -1:
                tc = torch.log(mct_tree[currentid][4])
                tc = tc + F.relu(tc - 10) * 1000
                logits = mct_tree[currentid][0].view(-1) - tc * 0.5
                logits = logits[:50001]
                log_probs = F.softmax(logits, dim=-1)

                pr = torch.multinomial(log_probs, num_samples=1)[0]
                prev = pr.item()
                mct_tree[currentid][4][prev] += 1
                lastid = currentid
                currentid = int(mct_tree[currentid][3][prev])
            # start from lastid & currentid

            cqs = mct_tree[lastid][2]
            tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            sentence = tokenizer.DecodeIds(
                output_tokens_list[context_length:].tolist())
            logit = mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs = torch.log(log_probs)
            score = log_pbs[prev].item()
            nextid = 0
            ip = checksentence(
                sentence,
                original_context,
                min_length,
                max_length,
                endnote,
                curvote=curvote,
                yayun=yayun)
            for j in final_result:
                if j[0] == sentence:
                    ip = 1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip = 1

            score = mct_tree[lastid][5] + score
            if (ip == 1):
                nextid = lastid
                dpcount += 1
                max_tries += 1
                if (dpcount >= 50) or (dpcount >= 8
                                       and len(sentence) < max_length):
                    nextid = 0
                    dpcount = 0
                mct_tree[lastid][4][prev] = 100000
                continue
            dpcount = 0
            if (ip == 0):
                mct_tree[lastid][4][prev] = 100000
                yay = yayun
                if curvote == 1:
                    yay = sentence[-2]

                final_result.append([
                    copy.deepcopy(sentence),
                    copy.deepcopy(score),
                    copy.deepcopy(tokens),
                    copy.deepcopy(mct_tree[lastid][1]), yay
                ])
                continue

            mct_tree[lastid][3][prev] = len(mct_tree)
            tmp = args.temperature
            if (len(sentence) >= 4
                    or (len(sentence) == 3 and max_length == 5)):
                tmp = tmp * 0.6
            rts = mct_tree[lastid][1]
            index = len(tokens[0])

            logits, *rts = model(
                tokens[:, index - 1:index],
                tokens.new_ones((1, 1)) * (index - 1),
                tokens.new_ones(
                    1,
                    1,
                    1,
                    args.mem_length + 1,
                    device=tokens.device,
                    dtype=torch.float), *rts)
            logits = logits[0, -1] / tmp
            if len(sentence) == max_length:
                logits[endid] += 10
            mct_tree.append([
                logits, rts, tokens, -np.ones(len(logits)),
                torch.ones(len(logits)).cuda(), score
            ])
            nextid = len(mct_tree) - 1
        del mct_tree
        torch.cuda.empty_cache()
        res = {}
        res['output_tokens_length'] = len(output_tokens_list)
        res['result'] = final_result
        return res


def getlength(str):
    w = str.replace('。', ',').replace('，', ',').replace('？', ',').replace(
        '?', ',').replace(' ', ',').replace('！',
                                            ',').replace('!', ',').replace(
                                                ':', ',').replace(' ', '')
    sp = w.split(',')

    return len(sp[-2])


def getlastsentence(str):
    w = str.replace('。', ',').replace('，', ',').replace('？', ',').replace(
        '?', ',').replace(' ', ',').replace('！',
                                            ',').replace('!', ',').replace(
                                                ':', ',').replace(' ', '')
    sp = w.split(',')
    fom = sp[-1]
    if len(fom) == 0:
        fom = sp[-2]
    return fom + str[-1]


def generate_string(model,
                    tokenizer,
                    args,
                    device,
                    title,
                    author,
                    desc=None,
                    length=None,
                    st=None,
                    lycr=5,
                    senlength=4):
    lycr_str = ''
    senlength_str = ''
    if lycr == 5:
        lycr_str = '诗体：五言'
    else:
        lycr_str = '诗体：七言'
    if senlength == 4:
        senlength_str = '格律：绝句'
    else:
        senlength_str = '格律：律诗'
    input_str = title + ' 作者:' + author + ' 体裁:诗歌' + lycr_str + senlength_str + '题名:' + title + ' 正文: '  # noqa
    if desc is not None:
        input_str = title + ' 作者:' + author + ' 体裁:诗歌' + lycr_str + senlength_str + '描述:' + desc + ' 题名:' + title + ' 正文: '  # noqa
    input_len = len(input_str)  # noqa
    context_count = 0  # noqa
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens = tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length >= args.seq_length:
            res = {}
            res['prompt_token_num'] = 0
            res['completion_token_num'] = 0
            res['text'] = ''
            res['errmsg'] = 'the text you entered is too long, please reduce the number of characters'
            raise InputTooLongException(
                'the text you entered is too long, please reduce the number of characters',
                res)

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor = torch.cuda.LongTensor(eo_tokens)  # noqa
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()

        start_time = time.time()  # noqa

        counter, mems = 0, []  # noqa
        org_context_length = context_length  # noqa
        completion_token_length = context_length
        beam_size = 1
        beam_candidate = 1
        beam_max = 1  # noqa
        max_headings = 4  # noqa
        final_storage = []  # noqa
        final_storage_score = []  # noqa
        step = senlength + 1
        if st is None:
            st = 8
        overall_score = []
        past_beam_id = []

        if length is not None:
            res = generate_sentence(
                model,
                tokenizer,
                args,
                device,
                context_tokens_tensor, [],
                min_length=lycr - 1,
                max_length=lycr,
                num_candidates=beam_size)
            beam_sentences = res.get('result', [])
            completion_token_length = res.get('output_tokens_length', 0)
        else:
            res = generate_sentence(
                model,
                tokenizer,
                args,
                device,
                context_tokens_tensor, [],
                min_length=lycr - 1,
                max_length=lycr,
                num_candidates=beam_size)
            beam_sentences = res.get('result', [])
            completion_token_length = res.get('output_tokens_length', 0)

        if len(beam_sentences) == 0:
            res = {}
            res['prompt_token_num'] = context_length
            res['completion_token_num'] = 0
            res['text'] = ''
            res['errmsg'] = '太难了，写不出来。'
            raise CanNotReturnException('太难了，写不出来。', res)

        for i in range(step):
            beam_new_sentences = []

            endnote = [',', '，', '?', '？']
            if i % 2 == 0:
                endnote = ['。', '?', '？', '！', '!']
            overall_score = []  # noqa
            past_beam_id = []  # noqa
            id = 0
            current_sentence = input_str + beam_sentences[0][0]

            ini_score = beam_sentences[id][1]  # noqa
            token_tensor = beam_sentences[id][2]
            mems = beam_sentences[id][3]

            len_sentence = getlength(beam_sentences[id][0])  # noqa

            res = generate_sentence(
                model,
                tokenizer,
                args,
                device,
                token_tensor,
                mems,
                num_candidates=beam_candidate,
                endnote=endnote,
                min_length=lycr - 1,
                max_length=lycr,
                yayun=beam_sentences[id][-1])
            gen = res.get('result', [])
            completion_token_length = res.get('output_tokens_length', 0)
            if len(gen) == 0:
                res = {}
                res['prompt_token_num'] = context_length
                res['completion_token_num'] = context_length
                res['text'] = ''
                res['errmsg'] = '太难了，写不出来。'
                raise CanNotReturnException('太难了，写不出来。', res)
            jj = gen[0]
            if ('<|end' in jj[0] or i == senlength - 1):
                if (i % 2 == 1 and i > -3):
                    del beam_sentences
                    del beam_new_sentences
                    torch.cuda.empty_cache()
                    res = {}
                    res['prompt_token_num'] = context_length
                    res['completion_token_num'] = completion_token_length
                    res['text'] = current_sentence
                    return res
                else:
                    res = generate_sentence(
                        model,
                        tokenizer,
                        args,
                        device,
                        token_tensor,
                        mems,
                        num_candidates=beam_candidate,
                        endnote=endnote,
                        min_length=lycr - 1,
                        max_length=lycr,
                        yayun=beam_sentences[id][-1])
                    gen = res.get('result', [])
                    completion_token_length = res.get('output_tokens_length',
                                                      0)

            if len(gen) == 0:
                res = {}
                res['prompt_token_num'] = context_length
                res['completion_token_num'] = 0
                res['text'] = ''
                res['errmsg'] = '太难了，写不出来。'
                raise CanNotReturnException('太难了，写不出来。', res)
            st = jj[0]
            # experiment shows that this is better universal,

            jj[0] = beam_sentences[id][0] + jj[0]
            jj[1] = 0
            beam_new_sentences.append(jj)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences = beam_new_sentences

            # parallel ends

        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()
        res = {}
        res['prompt_token_num'] = context_length
        res['completion_token_num'] = 0
        res['text'] = ''
        res['errmsg'] = '太难了，写不出来。'
        raise CanNotReturnException('太难了，写不出来。', res)


def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir
    }
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size() # noqa
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    args.vocab_size = after
    print('prepare tokenizer done', flush=True)

    return tokenizer


def set_args():
    args = get_args()
    args.deepspeed = True
    args.num_nodes = 1
    args.num_gpus = 1
    args.model_parallel_size = 1
    args.num_layers = 32
    args.hidden_size = 2560
    args.load = 'modelscope-txl/'
    args.num_attention_heads = 32
    args.max_position_embeddings = 1024
    args.tokenizer_type = 'ChineseSPTokenizer'
    args.cache_dir = 'cache'
    args.fp16 = True
    args.out_seq_length = 180
    args.seq_length = 200
    args.mem_length = 256
    args.transformer_xl = True
    args.temperature = 1.2
    args.top_k = 0
    args.top_p = 0

    return args


def prepare_model(model_dir):
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()  # noqa

    # Arguments.
    args = set_args()
    args.load = model_dir
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    # get the tokenizer
    args.tokenizer_path = model_dir
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    return model, tokenizer, args


def fast_poem(content, model, tokenizer, args):
    title = content['title']
    author = content['author']
    desc = content['desc']
    lycr = content['lycr']
    senlength = content['senlength']

    res = generate_string(
        model,
        tokenizer,
        args,
        torch.cuda.current_device(),
        title,
        author,
        desc=desc,
        lycr=lycr,
        senlength=senlength)

    return res
