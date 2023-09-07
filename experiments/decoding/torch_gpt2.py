# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import os
import math
import transformers


class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.max_position_embeddings = 1024
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.intermediate_size = 3072
        self.layer_norm_epsilon = 1e-5
        self.num_heads = 12


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config: GPT2Config = config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.c_attn = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer(
            'casual_mask', 
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool), diagonal=0).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings), 
            persistent=False
        )

    def forward(self, hidden_states: Tensor, last_key, last_value):
        # params:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        #   last_key: [batch_size, num_heads, prev_seq_length, head_dim]
        #   last_value: [batch_size, num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        #   key: [batch_size, num_heads, seq_length, head_dim]
        #   value: [batch_size, num_heads, seq_length, head_dim]
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        prev_seq_length = last_key.shape[2]
        qkv = self.c_attn(hidden_states)  # [batch_size, seq_length, hidden_size * 3]
        q, k, v = torch.split(qkv, hidden_states.shape[-1], dim=-1)  # [batch_size, seq_length, hidden_size] * 3
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_dim]
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_dim]

        kk = torch.cat([last_key, k], dim=2)    # [batch_size, num_heads, prev_seq_length + seq_length, head_dim]
        vv = torch.cat([last_value, v], dim=2)  # [batch_size, num_heads, prev_seq_length + seq_length, head_dim]

        # [num_heads, seq_length, prev_seq_length + seq_length]
        # like (seq_length = 3, prev_seq_length = 2)
        # 1 1 1
        # 1 1 1 1
        # 1 1 1 1 1

        # [batch_size, num_heads, seq_length, prev_seq_length + seq_length]
        new_seq_len = prev_seq_length + seq_length
        attn_weights = torch.matmul(q, kk.transpose(-1, -2)) / math.sqrt(self.head_dim)
        casual_mask = self.casual_mask[:, :, new_seq_len - seq_length:new_seq_len, :new_seq_len]

        mask_value = torch.finfo(q.dtype).min
        mask_value = torch.full([], mask_value, dtype=q.dtype, device=q.device)
        casual_mask = torch.where(casual_mask, attn_weights, mask_value)

        qk = torch.softmax(attn_weights, dim=-1)  # [batch_size, num_heads, seq_length, seq_length + prev_seq_length]

        hidden_states = torch.matmul(qk, vv)  # [batch_size, num_heads, seq_length, head_dim]
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size
        )
        hidden_states = self.c_proj(hidden_states)  # [batch_size, seq_length, hidden_size]
        return hidden_states, kk, vv


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        # params:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        # return:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate='tanh')
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states, last_key, last_value):
        # params:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        #   last_key: [batch_size, num_heads, prev_seq_length, head_dim]
        #   last_value: [batch_size, num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [batch_size, seq_length, hidden_size]
        #   last_key: [batch_size, num_heads, seq_length, head_dim]
        #   last_value: [batch_size, num_heads, seq_length, head_dim]
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states, key, value = self.attn(hidden_states, last_key, last_value)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, key, value


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config: GPT2Config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids, past_keys, past_values):
        # params:
        #   input_ids: [batch_size, seq_length]
        #   position_ids: int32[batch_size, seq_length]
        #   past_keys: [layers, batch_size, num_heads, prev_seq_length, head_dim]
        #   past_values: [layers, batch_size, num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [batch_size, 1, hidden_size]
        #   position_ids: int32[batch_size, 1]
        #   updated_keys: [batch_size, num_heads, prev_seq_length + seq_length, head_dim]
        #   updated_values: [batch_size, num_heads, prev_seq_length + seq_length, head_dim]

        inputs_embeds = self.wte(input_ids)  # [batch_size, seq_length, hidden_size]
        position_embeds = self.wpe(position_ids)  # [batch_size, seq_length, hidden_size]
        hidden_states = inputs_embeds + position_embeds  # [batch_size, seq_length, hidden_size]
        cur_keys = []  # layers of [batch_size, num_heads, seq_length, head_dim]
        cur_values = []  # layers of [batch_size, num_heads, seq_length, head_dim]
        for i, block in enumerate(self.h):
            hidden_states, cur_key, cur_value = block(hidden_states, past_keys[i], past_values[i])
            cur_keys.append(cur_key)
            cur_values.append(cur_value)

        cur_keys = torch.stack(cur_keys, dim=0)  # [layers, batch_size, num_heads, seq_length, head_dim]
        cur_values = torch.stack(cur_values, dim=0)  # [layers, batch_size, num_heads, seq_length, head_dim]

        hidden_states = self.ln_f(hidden_states)  # [batch_size, seq_length, hidden_size]

        # # [batch_size, layers, num_heads, prev_seq_length + seq_length, head_dim]]
        # updated_cur_keys = torch.concat([past_keys, cur_keys], dim=3)
        # updated_cur_values = torch.concat([past_values, cur_values], dim=3)
        
        position_ids = position_ids[:, -1:] + 1  # [batch_size, 1]

        return hidden_states, position_ids, cur_keys, cur_values
        # return hidden_states[-1:], position_ids, None, None


class GPT2LMHead(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

    @classmethod
    def from_transformers(cls, model_name: str):
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

        # load from transformers
        hf_gpt2: torch.nn.Module = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        hf_config = transformers.GPT2Config.from_pretrained(model_name)

        # create config
        config = GPT2Config()
        config.vocab_size = hf_config.vocab_size
        config.hidden_size = hf_config.n_embd
        config.num_hidden_layers = hf_config.n_layer
        config.num_heads = hf_config.n_head
        config.intermediate_size = hf_config.n_inner if hf_config.n_inner else 4 * hf_config.n_embd
        config.max_position_embeddings = hf_config.n_positions
        config.layer_norm_epsilon = hf_config.layer_norm_epsilon

        # create model
        module = cls(config)
        allow_missing = ['lm_head.weight']
        found_tensors = []
        with torch.no_grad():
            for name, tensor in hf_gpt2.named_parameters():
                pointer = module
                for m_name in name.split('.')[:-1]:
                    pointer = getattr(pointer, m_name)
                is_linear = isinstance(pointer, nn.Linear)
                pointer = getattr(pointer, name.split('.')[-1])

                if not isinstance(pointer, Tensor):
                    raise ValueError('{} is not a tensor'.format(name))
                found_tensors.append(pointer)
                if is_linear:
                    tensor = tensor.t()
                try:
                    pointer.copy_(tensor)
                except Exception as e:
                    raise ValueError('copying {} failed: {}'.format(name, e))
        module.lm_head.weight = module.transformer.wte.weight

        return module

    def forward(self, input_ids, position_ids, past_keys=None, past_values=None):
        # params:
        #   input_ids: int32[batch_size, seq_length]
        #   position_ids: int32[batch_size, seq_length]
        #   past_keys: [layers, batch_size, prev_seq_length, hidden_size]
        #   past_values: [layers, batch_size, prev_seq_length, hidden_size]
        # return:
        #   input_ids: int32[batch_size, 1]
        #   position_ids: int32[batch_size, 1]
        #   updated_keys: [layers, batch_size, prev_seq_length + seq_length, hidden_size]
        #   updated_values: [layers, batch_size, prev_seq_length + seq_length, hidden_size]
        batch_sz, seq_len = input_ids.shape
        if past_keys is None:
            past_keys = torch.zeros([self.num_hidden_layers ,batch_sz, self.num_heads, 0, self.head_dim], dtype=input_ids.dtype, device=input_ids.device)
        if past_values is None:
            past_values = torch.zeros([self.num_hidden_layers ,batch_sz, self.num_heads, 0, self.head_dim], dtype=input_ids.dtype, device=input_ids.device)
        
        hidden_states, position_ids, past_keys, past_values = self.transformer(
            input_ids, position_ids, past_keys, past_values
        )
        logits = self.lm_head(hidden_states[:, -1:])  # [batch_size, 1, vocab_size]
        updated_input_ids = torch.argmax(logits, dim=-1)  # [1]
        # we want to keep types consistent, since in the autoregressive case,
        #   the output is fed back into the input of the compiled model
        updated_input_ids = updated_input_ids.to(input_ids.dtype)
        return updated_input_ids, position_ids, past_keys, past_values

from decoding_config import ModelConfig, DecodingSampleConfig
import onnx

def get_model(name: str = 'gpt2'):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']
    from transformers import GPT2Config

    hf_gpt2 = GPT2LMHead.from_transformers(name)
    hf_gpt2.eval()
    hf_gpt2.to('cuda')
    hf_gpt2_config = GPT2Config.from_pretrained(name)
    model_config = ModelConfig(n_layers=hf_gpt2_config.n_layer, n_heads=hf_gpt2_config.n_head, head_dim=hf_gpt2_config.n_embd // hf_gpt2_config.n_head)
    return hf_gpt2, model_config

def save_onnx(model, model_config: ModelConfig, decoding_config: DecodingSampleConfig, path: str = 'gpt2.onnx'):
    x = torch.randint(0, 50257, (decoding_config.batch_size, decoding_config.q_seq_len), dtype=torch.int32).cuda()
    ps = torch.arange(0, decoding_config.q_seq_len, dtype=torch.int32).unsqueeze(0).repeat(decoding_config.batch_size, 1).cuda()

    kc = torch.randn(model_config.n_layers, decoding_config.batch_size, model_config.n_heads, decoding_config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
    vc = torch.randn(model_config.n_layers, decoding_config.batch_size, model_config.n_heads, decoding_config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')

    torch.onnx.export(
        model, 
        (x, ps, kc, vc),
        path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input_ids', 'position_ids', 'past_keys', 'past_values'],
        output_names=['updated_input_ids', 'updated_position_ids', 'updated_keys', 'updated_values'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'},
            'position_ids': {0: 'batch_size', 1: 'seq_length'},
            'past_keys': {1: 'batch_size', 3: 'prev_seq_length'},
            'past_values': {1: 'batch_size', 3: 'prev_seq_length'},
            'updated_input_ids': {0: 'batch_size'},
            'updated_position_ids': {0: 'batch_size'},
            'updated_keys': {1: 'batch_size', 3: 'prev_seq_length+seq_length'},
            'updated_values': {1: 'batch_size', 3: 'prev_seq_length+seq_length'},
        }
    )

    onnx.checker.check_model(path)

import onnxruntime
import numpy as np
from hidet.utils.benchmark import benchmark_func

from decoding_config import ModelConfig, DecodingSampleConfig


def benchmark_onnx(ort_session: onnxruntime.InferenceSession, model_config: ModelConfig, config: DecodingSampleConfig):
    binding = ort_session.io_binding()
    x = torch.randint(0, 50257, (config.batch_size, config.q_seq_len), dtype=torch.int32).cuda()
    ps = torch.arange(0, config.q_seq_len, dtype=torch.int32).unsqueeze(0).repeat(config.batch_size, 1).cuda()

    kc = torch.randn(model_config.n_layers, config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
    vc = torch.randn(model_config.n_layers, config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')

    def bind_input(name, tensor, dtype=np.float32):
        binding.bind_input(
            name=name,
            device_type='cuda',
            device_id=0,
            element_type=dtype,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr()
        )
    bind_input('input_ids', x, np.int32)
    bind_input('position_ids', ps, np.int32)
    bind_input('past_keys', kc, np.float16)
    bind_input('past_values', vc, np.float16)

    def bind_output(name, tensor, dtype=np.float32):
        binding.bind_output(
            name=name,
            device_type='cuda',
            device_id=0,
            element_type=dtype,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr()
        )
    output_ids = torch.empty([config.batch_size, 1], dtype=torch.int32, device='cuda')
    output_position_ids = torch.empty([config.batch_size, 1], dtype=torch.int32, device='cuda')
    output_keys   = torch.empty([model_config.n_layers, config.batch_size, model_config.n_heads, kc.shape[3] + x.shape[1], model_config.head_dim], dtype=torch.float16, device='cuda')
    output_values = torch.empty([model_config.n_layers, config.batch_size, model_config.n_heads, kc.shape[3] + x.shape[1], model_config.head_dim], dtype=torch.float16, device='cuda')

    bind_output('updated_input_ids', output_ids, np.int32)
    bind_output('updated_position_ids', output_position_ids, np.int32)
    bind_output('updated_keys', output_keys, np.float16)
    bind_output('updated_values', output_values, np.float16)
    return benchmark_func(lambda: ort_session.run_with_iobinding(binding), repeat=config.repeat, median=False, warmup=config.warmup)

def benchmark_torch(model, model_config: ModelConfig, config: DecodingSampleConfig):
    x = torch.randint(0, 50257, (config.batch_size, config.q_seq_len), dtype=torch.int32).cuda()
    ps = torch.arange(0, config.q_seq_len, dtype=torch.int32).unsqueeze(0).repeat(config.batch_size, 1).cuda()

    kc = torch.randn(model_config.n_layers, config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
    vc = torch.randn(model_config.n_layers, config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
    
    return benchmark_func(lambda: model(x, ps, kc, vc), repeat=config.repeat, median=False, warmup=config.warmup)


model, config = get_model()
ort_session = onnxruntime.InferenceSession("gpt2.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(benchmark_onnx(ort_session, config, DecodingSampleConfig(124, 128, 2)))
print(benchmark_torch(model, config, DecodingSampleConfig(124, 128, 2)))
