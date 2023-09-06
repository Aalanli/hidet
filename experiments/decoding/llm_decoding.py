# %%
from dataclasses import dataclass
from hidet.utils.benchmark import benchmark_func

@dataclass
class DecodingSampleConfig:
    q_seq_len: int # between 1, 32 (inclusive)
    kv_seq_len: int # between 512, 1024 (inclusive)
    batch_size: int # between 1, 32 (inclusive)
    repeat: int = 20

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    head_dim: int


import torch
import hidet
import hidet.testing
from transformers import GPT2Config, GPT2LMHeadModel, LlamaForCausalLM, LlamaConfig
from hidet.testing.models.llama import get_compiled_model, generate, convert_model
from hidet.runtime.storage import current_memory_pool

from optimum.onnxruntime import ORTModelForCausalLM

# LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float16).save_pretrained('meta-llama/llama-2-7b-chat-hf-fp16')
# # %%
# model = ORTModelForCausalLM.from_pretrained('nenkoru/llama-7b-onnx-merged-fp16', export=True, provider='CUDAExecutionProvider')


def get_model_llama_torch(name: str = 'decapoda-research/llama-7b-hf'):
    with torch.device('cuda'):
        model = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model_config = ModelConfig(n_layers=model.config.num_hidden_layers, n_heads=model.config.num_attention_heads, head_dim=model.config.hidden_size // model.config.num_key_value_heads)
    return model, model_config

def get_model_llama_hidet(name: str = 'decapoda-research/llama-7b-hf'):
    model, config, tokenizer = get_compiled_model(name, device='cuda', opt=True, batch_size='batch_size')
    model_config = ModelConfig(n_layers=config.num_hidden_layers, n_heads=config.num_attention_heads, head_dim=config.hidden_size // config.num_key_value_heads)
    return model, model_config

def bench_llama_torch(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 32000, (config.batch_size, config.q_seq_len)).to('cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=torch.float16, device='cuda')
        for _ in range(2)) for _ in range(model_config.n_layers))
    
    def bench():
        model(ids, use_cache=True, past_key_values=kv_cache)
    
    return benchmark_func(bench, repeat=config.repeat, median=False)

def bench_llama_hidet(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = hidet.randint(0, 32000, (config.batch_size, config.q_seq_len), dtype=hidet.int32, device='cuda')
    device = 'cuda'
    position_ids = hidet.arange(0, 1024, dtype=hidet.int32, device=device).unsqueeze(0)

    make_past = lambda: hidet.zeros(
        [config.batch_size, model_config.n_heads, 0, model_config.head_dim], device=device, dtype=hidet.float16
    )
    past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]

    def bench():
        model(ids, position_ids, *past_keys_values)
    
    return benchmark_func(bench, repeat=config.repeat, median=False)


def get_model_gpt2_torch(name: str):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

    hf_gpt2 = GPT2LMHeadModel.from_pretrained(name)
    hf_gpt2.eval()
    hf_gpt2.to('cuda')
    hf_gpt2_config = GPT2Config.from_pretrained(name)
    model_config = ModelConfig(n_layers=hf_gpt2_config.n_layer, n_heads=hf_gpt2_config.n_head, head_dim=hf_gpt2_config.n_embd // hf_gpt2_config.n_head)
    return hf_gpt2, model_config

def get_model_gpt2_hidet(name: str):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']
    device = 'cuda'
    gpt2_module = hidet.testing.models.gpt2.model(disable_cache=True)
    gpt2_module.cuda()

    input_ids = hidet.symbol(['batch_size', 'seq_length'], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol(['batch_size', 'seq_length'], dtype=hidet.int32, device=device)
    cache_shape = ['batch_size', gpt2_module.num_hidden_layers, gpt2_module.num_heads, 'prev_seq_length', gpt2_module.head_dim]
    past_keys = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)
    past_values = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)

    outputs = gpt2_module(input_ids, position_ids, past_keys, past_values)
    graph = hidet.trace_from(outputs, inputs=[input_ids, position_ids, past_keys, past_values])

    with hidet.graph.PassContext() as ctx:
        # ctx.set_precision('float16')
        graph = hidet.graph.optimize(graph)
    compiled_model = graph.build(space=2)
    model_config = ModelConfig(n_layers=gpt2_module.num_hidden_layers, n_heads=gpt2_module.num_heads, head_dim=gpt2_module.head_dim)
    return compiled_model, model_config

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
    
    return benchmark_func(bench, repeat=config.repeat, median=False)

def bench_gpt2_hidet(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = hidet.randint(0, 50257, (config.batch_size, config.q_seq_len), dtype=hidet.int32, device='cuda')
    pos_ids = torch.arange(0, config.q_seq_len, dtype=torch.int32, device='cuda').unsqueeze(0).repeat(config.batch_size, 1)
    pos_ids = hidet.from_torch(pos_ids)
    k_cache = hidet.randn(config.batch_size, model_config.n_layers, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=hidet.float32, device='cuda')
    v_cache = hidet.randn(config.batch_size, model_config.n_layers, model_config.n_heads, config.kv_seq_len, model_config.head_dim, dtype=hidet.float32, device='cuda')

    def bench():
        model(ids, pos_ids, k_cache, v_cache)
    
    return benchmark_func(bench, repeat=config.repeat, median=False)

def bench_gpt2_onnx(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 50257, (config.batch_size, config.q_seq_len)).to('cuda')
    attn_mask = torch.ones(config.batch_size, config.kv_seq_len + 1, dtype=torch.int32, device='cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim).to('cuda') 
        for _ in range(2)) for _ in range(model_config.n_layers))

    bench = lambda: model(ids, attn_mask, kv_cache)
    return benchmark_func(bench, repeat=config.repeat, median=False)

# currently do not have enough gpu memory to run this: (for exporting llama to onnx)
# optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation --framework pt --opset 16 --sequence_length 1024 --batch_size 1 --device cuda --fp16 llama-2-7b-optimum/

config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=512, batch_size=32)
#####################
model, model_config = get_model_gpt2_torch('gpt2')
model = model.to('cuda')
with torch.autocast('cuda'):
    times = bench_gpt2_torch(model, model_config, config)

print('torch-gpt2', sum(times) / len(times))
del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_gpt2_torch('gpt2')
model = model.to('cuda')
model = torch.compile(model) # for some reason max-autotune is slower than default
with torch.autocast('cuda'):
    times = bench_gpt2_torch(model, model_config, config)

print('torch-compile-gpt2', sum(times) / len(times))
del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_gpt2_hidet('gpt2')
times = bench_gpt2_hidet(model, model_config, config)
print('hidet gpt2:', times)

del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_gpt2_onnx('gpt2')
timse = bench_gpt2_onnx(model, model_config, config)
print('onnx gpt2:', times)
del model
torch.cuda.empty_cache()


# don't have enough memory for larger batch/kv_seq_len
config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=128, batch_size=1)
#####################
model, model_config = get_model_llama_torch()
times = bench_llama_torch(model, model_config, config)
print('torch-llama:', times)
del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_llama_torch()
model = torch.compile(model)
times = bench_llama_torch(model, model_config, config)
print('torch-compile-llama:', times)
del model
torch.cuda.empty_cache()

#####################
model, model_config = get_model_llama_hidet()
times = bench_llama_hidet(model, model_config, config)
print('hidet-llama:', times)
del model
