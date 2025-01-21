# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
from megatron_util import mpu, print_rank_0
from megatron_util.fp16 import FP16_Module
from torch.nn import functional as F

from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.utils.logger import get_logger
from modelscope.utils.megatron_utils import init_megatron_util
from modelscope.utils.nlp.load_checkpoint import pre_load
from . import PlugModel
from .configuration import PlugNLGConfig

logger = get_logger()


class DistributedPlug(TorchModel):
    """
    The wapper class of PLUG Model to initialize parallel environment, load model weights, generate sentences.
    Parameters:
        model_dir (`str`, *required*):
            Path to model damo/nlp_plug_text-generation_27B.
        The model structure in model_dir should be like this:
        model_dir
            |_ config.json
            |_ configuration.json
            |_ ds_zero-offload_10B_config.json
            |_ vocab.txt
            |_ model <-- an empty directory

        Model binaries shall be downloaded separately to populate the model directory, so that
        the model directory would contain the following binaries:
            |_ model
                |_ mp_rank_00_model_states.pt
                |_ mp_rank_01_model_states.pt
                |_ mp_rank_02_model_states.pt
                |_ mp_rank_03_model_states.pt
                |_ mp_rank_04_model_states.pt
                |_ mp_rank_05_model_states.pt
                |_ mp_rank_06_model_states.pt
                |_ mp_rank_07_model_states.pt
        rank (`int`, *required*):
            Used to identify different GPUs in a tensor parallel environment. eg. The rank of GPU #0 is 0, and the
            model file `mp_rank_00_model_states.pt` will be loaded on this GPU.
        world_size (`int`, *required*, defaults to 8):
            The parallel size in total.
        model_parallel_size (`int`, *required*, defaults to 8):
            The parallel size of model(tensor parallel).
        master_ip (`str`, *required*):
            The master IP, can usually be set to `"127.0.0.1"`, used as part of
            [`~torch.distributed.init_process_group`] method parameter `init_method`.
            `init_method` = `"tcp://{master_ip}:{master_port}"`
        master_port (`str`, *required*):
            The master port, can usually be set to `"29500"`, used as part of
            [`~torch.distributed.init_process_group`] method parameter `init_method`.
            `init_method` = `"tcp://{master_ip}:{master_port}"`
        seed (`int`, *optional*, defaults to 42):
            Random seed to control sampling.
    """

    def __init__(self, model_dir, rank, **kwargs):
        super().__init__(model_dir, **kwargs)
        self.rank = rank
        self.model_cfg = kwargs
        self.config = PlugNLGConfig.from_pretrained(model_dir)

        init_megatron_util(model_dir=model_dir, rank=rank)

        self.iteration = 0
        self.model = self.initialize_model(path_load_tag='model')

    def initialize_model(self, path_load_tag='model'):
        """Build the model."""
        print_rank_0('Building Plug model. It will take a few minutes ...')
        model = PlugModel(self.config)

        if mpu.get_data_parallel_rank() == 0:
            logger.info(
                ' > number of parameters on model parallel rank {}: {}'.format(
                    mpu.get_tensor_model_parallel_rank(),
                    sum([p.nelement() for p in model.parameters()])))

        if self.config.deepspeed and self.config.fp16:
            model.half()

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if self.config.fp16:
            model = FP16_Module(model)
            if self.config.fp32_embedding:
                model.module.model.bert.embeddings.word_embeddings.float()
                model.module.model.bert.embeddings.position_embeddings.float()
                model.module.model.bert.embeddings.token_type_embeddings.float(
                )
            if self.config.fp32_tokentypes:
                model.module.model.bert.embeddings.token_type_embeddings.float(
                )
            if self.config.fp32_layernorm:
                for name, _module in model.named_modules():
                    if 'LayerNorm' in name:
                        _module.float()

        load_model = pre_load(
            mpu.get_tensor_model_parallel_rank(),
            self.model_dir,
            tag=path_load_tag)
        model_dict = model.module.model.state_dict()
        for key in load_model:
            if key not in model_dict.keys():
                print_rank_0('Skip key: ' + key)
            else:
                print_rank_0('Loading key: ' + key)
        model.module.model.load_state_dict(load_model, strict=False)
        return model

    @staticmethod
    def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        # This function has been mostly taken from huggingface conversational ai code at
        # https://medium.com/huggingface/how-to-build-a-state-of-the-art-
        # conversational-ai-with-transfer-learning-2d818ac26313

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

    def forward(self,
                input_tokens,
                token_type_ids=None,
                attention_mask=None,
                target_tokens=None,
                position_ids=None,
                decode_attention_mask=None,
                checkpoint_activations=False,
                is_infer=False,
                sequence_output=None,
                parallel_output=True):
        return self.model(
            input_tokens,
            token_type_ids,
            attention_mask,
            target_tokens,
            position_ids,
            decode_attention_mask,
            checkpoint_activations=checkpoint_activations,
            is_infer=is_infer,
            sequence_output=sequence_output,
            parallel_output=parallel_output)

    def generate(self, input: Dict[str, Tensor], out_length=128, *kwargs):
        device = torch.cuda.current_device()
        batch_size = input['input_ids'].shape[0]
        tokens = input['input_ids'].view(1, -1).contiguous().to(device)
        dec_input_ids = input['dec_input_ids'].to(device)
        attention_mask = input['attention_mask'].to(device)
        self.model.eval()
        with torch.no_grad():
            # Only supports batch_size=1
            all_generate_tokens = []
            generate_tokens = []
            counter = 0
            sequence_output = None
            vocab_size = self.config.original_vocab_size
            sep_token_idx = 102  # index of [SEP] token in BertTokenizer
            while counter < out_length:
                if counter % 128 == 0 and counter != 0:
                    # Sliding window
                    generate_tokens.append(sep_token_idx)
                    start = (tokens == sep_token_idx).nonzero(
                        as_tuple=True)[-1]
                    if start + len(generate_tokens) >= 512:
                        tokens = torch.cat([
                            tokens[:start],
                            torch.cuda.LongTensor(generate_tokens)
                        ], -1)[-512:]
                    else:
                        tokens[0][start:start + len(generate_tokens
                                                    )] = torch.cuda.LongTensor(
                                                        generate_tokens)

                    attention_mask = (tokens != 0)
                    dec_input_ids = input['dec_input_ids'].to(device)
                    generate_tokens = []
                    sequence_output = None

                position_ids = torch.full([batch_size, 1],
                                          len(generate_tokens),
                                          dtype=torch.long,
                                          device=device)
                _, logits, sequence_output = self.model(
                    tokens,
                    None,
                    attention_mask,
                    dec_input_ids,
                    attention_mask,
                    position_ids,
                    is_infer=True,
                    sequence_output=sequence_output,
                    parallel_output=False)
                logits = logits[:, -1, :]
                logits = logits / self.model_cfg['temperature']
                logits = self.top_k_logits(
                    logits,
                    top_k=self.model_cfg['top_k'],
                    top_p=self.model_cfg['top_p'])
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                prev_token = prev[0].item()
                if prev_token >= vocab_size:
                    prev_token = 100
                    prev[0] = 100
                if prev_token == 102 and len(all_generate_tokens) > int(
                        max(1, out_length) * 0.8):
                    break
                if prev_token == 102:
                    counter += 1
                    continue
                dec_input_ids = torch.cat([dec_input_ids, prev], dim=1)
                generate_tokens.append(prev_token)
                all_generate_tokens.append(prev_token)
                counter += 1

            generate_context = []
            for token in all_generate_tokens:
                if generate_context and generate_context[
                        -1] == 100 and token == 100:
                    continue
                else:
                    generate_context.append(token)
            return {'generate_context': generate_context}
