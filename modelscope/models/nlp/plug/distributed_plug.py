import random
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict

from . import PlugModel
from modelscope.models.base import Tensor
from modelscope.utils.nlp import mpu
from modelscope.utils.nlp.utils import print_rank_0
from modelscope.utils.nlp.fp16 import FP16_Module
from modelscope.utils.nlp.distributed import DistributedDataParallel as DDP

import os
from modelscope.utils.torch_utils import init_dist
def initialize_distributed(rank):
    """Initialize torch.distributed."""
    # Manually set the device ids.
    #torch.multiprocessing.set_start_method("spawn")
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '12345')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=8, rank=rank,
        init_method=init_method)
    # Set the model-parallel communicators.
    mpu.initialize_model_parallel(8)

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits



class DistributedPlug:
    @classmethod
    def init(cls, rank, model_dir, model_config, args):
    #def init(cls, rank):
        #torch.backends.cudnn.enabled = False
        #
        cls.rank = rank
        cls.args = args
        cls.config = model_config
        cls.model_dir = model_dir
        initialize_distributed(rank)
        cls.set_random_seed(cls, args.seed)
        cls.setup_model(cls, path_load_tag='model')

    def set_random_seed(cls, seed):
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mpu.model_parallel_cuda_manual_seed(seed)

    def get_model(cls):
        """Build the model."""

        print_rank_0('Building Plug model. It will take a few minutes ...')
        model = PlugModel(cls.config)

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)

        if cls.args.deepspeed and cls.args.fp16:
            model.half()   

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if cls.args.fp16:
            model = FP16_Module(model)
            if cls.args.fp32_embedding:
                model.module.model.bert.embeddings.word_embeddings.float()
                model.module.model.bert.embeddings.position_embeddings.float()
                model.module.model.bert.embeddings.token_type_embeddings.float()
            if cls.args.fp32_tokentypes:
                model.module.model.bert.embeddings.token_type_embeddings.float()
            if cls.args.fp32_layernorm:
                for name, _module in model.named_modules():
                    if 'LayerNorm' in name:
                        _module.float()

        # model = DDP(model)

        return model

    def setup_model(cls, path_load_tag='model'):
        dist_model = cls.get_model(cls)
        if cls.model_dir is not None:
            from modelscope.utils.nlp.load_checkpoint import pre_load
            load_model = pre_load(mpu, cls.model_dir, tag=path_load_tag)
            # model_dict = dist_model.module.module.model.state_dict()
            model_dict = dist_model.module.model.state_dict()
            for key in load_model:
                if key not in model_dict.keys():
                    print_rank_0('Skip key: '+key)
                else:
                    print_rank_0('Loading key: '+key)
            # dist_model.module.module.model.load_state_dict(load_model, strict=False)
            dist_model.module.model.load_state_dict(load_model, strict=False)
        cls.args.iteration = 0
        cls.dist_model = dist_model

    @classmethod
    def forward(cls, input:Dict[str, Tensor]):
        device = torch.cuda.current_device()
        tokens = input["input_ids"].to(device)
        dec_input_ids = input["dec_input_ids"].to(device)
        attention_mask = input["attention_mask"].to(device)
        cls.dist_model.eval()
        seq_length = 128
        with torch.no_grad():
            all_generate_tokens = []
            generate_tokens = []
            counter = 0
            sequence_output = None
            vocab_size = 21128
            #tokens, attention_mask, types, dec_input_ids = get_batch(context_tokens_tensor, device, args)
            while counter < seq_length:
                # if counter % 128 == 0 and counter != 0:
                #    generate_tokens.append(tokenizer.vocab[args.sep_token])
                #    start = (context_tokens_tensor == 102).nonzero(as_tuple=True)[-1]
                #    if start + len(generate_tokens) >= 512:
                #        context_tokens_tensor = torch.cat([context_tokens_tensor[:start], torch.cuda.LongTensor(generate_tokens)], -1)[-512:]
                #    else:
                #        context_tokens_tensor[start:start+len(generate_tokens)] = torch.cuda.LongTensor(generate_tokens)
                #    tokens, attention_mask, types, dec_input_ids = get_batch(context_tokens_tensor, device, args)
                #    generate_tokens = []
                #    sequence_output = None

                position_ids = torch.full([cls.args.batch_size, 1], len(generate_tokens), dtype=torch.long, device=device)
                _, logits, sequence_output = cls.dist_model(tokens, None, attention_mask, dec_input_ids, attention_mask, position_ids, is_infer=True, sequence_output=sequence_output, parallel_output=False)

                partition_vocab_size = logits.size()[-1]

                logits = logits[:, -1, :]
                logits = logits / cls.args.temperature
                logits = top_k_logits(logits, top_k=cls.args.top_k, top_p=cls.args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                prev_token = prev[0].item()
                if prev_token >= vocab_size: #or prev_token == 102:
                    prev_token = 100
                    prev[0] = 100
                # if prev_token == 102 and len(all_generate_tokens) > int(max(1, length) * 0.8):
                if prev_token == 102:
                    break
                #if prev_token == 102:
                #    counter += 1
                #    continue
                #if prev_token == 100:
                #    counter += 1
                #    continue
                dec_input_ids = torch.cat([dec_input_ids, prev], dim=1)
                generate_tokens.append(prev_token)
                all_generate_tokens.append(prev_token)
                counter += 1

            generate_context = []
            for token in all_generate_tokens:
                if generate_context and generate_context[-1] == 100 and token == 100:
                    continue
                else:
                    generate_context.append(token)
            return {"generate_context": generate_context}

