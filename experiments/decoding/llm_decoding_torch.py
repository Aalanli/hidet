# %%
from hidet.utils.benchmark import benchmark_func

from decoding_config import DecodingSampleConfig, ModelConfig

import torch
import hidet
import hidet.testing
from transformers import GPT2Config, GPT2LMHeadModel, LlamaForCausalLM, LlamaConfig
from optimum.onnxruntime import ORTModelForCausalLM

# LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float16).save_pretrained('meta-llama/llama-2-7b-chat-hf-fp16')
# model = ORTModelForCausalLM.from_pretrained('nenkoru/llama-7b-onnx-merged-fp16', export=True, provider='CUDAExecutionProvider')


def get_model_llama_torch(name: str = 'decapoda-research/llama-7b-hf'):
    with torch.device('cuda'):
        model = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model_config = ModelConfig(n_layers=model.config.num_hidden_layers, n_heads=model.config.num_attention_heads, head_dim=model.config.hidden_size // model.config.num_key_value_heads)
    return model, model_config


def bench_llama_torch(model, model_config: ModelConfig, config: DecodingSampleConfig):
    # ids = torch.randint(0, 32000, (config.batch_size, config.q_seq_len)).to('cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
        for _ in range(2)) for _ in range(model_config.n_layers))
    attn_mask = torch.ones((config.batch_size, 1024), device='cuda', dtype=torch.bool)
    pos_ids = torch.arange(0, 1024, device='cuda').unsqueeze(0).repeat(config.batch_size, 1)
    with torch.no_grad():
        def bench():
            ids = [torch.randint(0, 32000, (config.batch_size, 1), device='cuda') for _ in range(config.q_seq_len)]
            cache = kv_cache
            for i in range(config.q_seq_len):
                cache = model(ids[i], attention_mask=attn_mask[:, :config.kv_seq_len+i+1], position_ids=pos_ids[:, i+config.kv_seq_len:i+config.kv_seq_len+1], use_cache=True, past_key_values=cache).past_key_values
                # print(cache[0][0].shape)
        
        return benchmark_func(bench, median=False, warmup=3)

def bench_llama_hidet(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 32000, (config.batch_size, config.q_seq_len), dtype=torch.int32, device='cuda')
    ids = hidet.from_torch(ids)
    device = 'cuda'
    position_ids = torch.arange(0, 1024, dtype=torch.int32, device=device).unsqueeze(0)
    position_ids = hidet.from_torch(position_ids)

    make_past = lambda: hidet.zeros(
        [config.batch_size, model_config.n_heads, 0, model_config.head_dim], device=device, dtype=hidet.float16
    )
    past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]

    def bench():
        model(ids, position_ids, *past_keys_values)
    
    return benchmark_func(bench, repeat=config.repeat, median=False, warmup=config.warmup)


def get_model_gpt2_torch(name: str):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

    hf_gpt2 = GPT2LMHeadModel.from_pretrained(name)
    hf_gpt2.eval()
    hf_gpt2.to('cuda')
    hf_gpt2_config = GPT2Config.from_pretrained(name)
    model_config = ModelConfig(n_layers=hf_gpt2_config.n_layer, n_heads=hf_gpt2_config.n_head, head_dim=hf_gpt2_config.n_embd // hf_gpt2_config.n_head)
    return hf_gpt2, model_config


def get_model_gpt2_onnx(name: str):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']
    model = ORTModelForCausalLM.from_pretrained(name, export=True, provider='CUDAExecutionProvider')
    model_config = ModelConfig(n_layers=model.config.n_layer, n_heads=model.config.n_head, head_dim=model.config.n_embd // model.config.n_head)
    return model, model_config

def bench_gpt2_torch(model: GPT2LMHeadModel, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 32000, (config.batch_size, config.q_seq_len)).to('cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim).to('cuda') 
        for _ in range(2)) for _ in range(model_config.n_layers))
    
    def bench():
        model(ids, use_cache=True, past_key_values=kv_cache)
    
    return benchmark_func(bench, repeat=config.repeat, median=False, warmup=config.warmup)


def bench_gpt2_onnx(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 50257, (config.batch_size, config.q_seq_len)).to('cuda')
    attn_mask = torch.ones(config.batch_size, config.kv_seq_len + 1, dtype=torch.int32, device='cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim).to('cuda') 
        for _ in range(2)) for _ in range(model_config.n_layers))

    bench = lambda: model(ids, attn_mask, kv_cache)
    return benchmark_func(bench, repeat=config.repeat, median=False, warmup=config.warmup)

# currently do not have enough gpu memory to run this: (for exporting llama to onnx)
# optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation --framework pt --opset 16 --sequence_length 1024 --batch_size 1 --device cuda --fp16 llama-2-7b-optimum/


# config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=512, batch_size=32)
# #####################
# model, model_config = get_model_gpt2_torch('gpt2')
# model = model.to('cuda')
# with torch.autocast('cuda'):
#     times = bench_gpt2_torch(model, model_config, config)

# print('torch-gpt2:', sum(times) / len(times))
# # torch-gpt2: 18.939590454101562

# del model
# # print(torch.cuda.memory_summary())
# torch.cuda.empty_cache()

# #####################
# model, model_config = get_model_gpt2_torch('gpt2')
# model = model.to('cuda')
# model = torch.compile(model) # for some reason max-autotune is slower than default
# with torch.autocast('cuda'):
#     times = bench_gpt2_torch(model, model_config, config)

# print('torch-compile-gpt2:', sum(times) / len(times))
# # torch-compile-gpt2: 17.110588550567627

# del model
# torch.cuda.empty_cache()

# #####################
# model, model_config = get_model_gpt2_onnx('gpt2')
# timse = bench_gpt2_onnx(model, model_config, config)
# print('onnx gpt2:', sum(times) / len(times))
# # onnx gpt2: 17.110588550567627

# del model
# torch.cuda.empty_cache()

# don't have enough memory for larger batch/kv_seq_len
config = DecodingSampleConfig(q_seq_len=128, kv_seq_len=128, batch_size=1)
# %%
#####################
model, model_config = get_model_llama_torch()
times = bench_llama_torch(model, model_config, config)
print('torch-llama:', sum(times) / len(times) / 1000)
# torch-llama: 26.811532974243164

del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_llama_torch()
model = torch.compile(model)
times = bench_llama_torch(model, model_config, config)
print('torch-compile-llama:', sum(times) / len(times) / 1000)
# torch-compile-llama: 23.289554119110107

del model
torch.cuda.empty_cache()
torch._dynamo.reset()
torch.cuda.empty_cache()
import gc
gc.collect()

# %%
model, model_config = get_model_llama_torch()
model = torch.compile(model, mode='reduce-overhead')
times = bench_llama_torch(model, model_config, config)
print('torch-compile-llama:', sum(times) / len(times) / 1000)
# torch-compile-llama: 23.289554119110107

del model
torch.cuda.empty_cache()

# decode 128 tokens with 0 prefill, batch size 1
# torch-llama: 3.195237979888916
# torch-compile-llama: 2.881284008026123

# 128 prefill, decode 128, batch size 1
# torch-llama: 3.2373869705200193
# torch-compile-llama: 2.905135974884033
