# %%
from dataclasses import dataclass
from hidet.utils.benchmark import benchmark_func
from hidet.testing.models.llama import get_compiled_model
from hidet.runtime.storage import current_memory_pool

@dataclass
class DecodingSampleConfig:
    q_seq_len: int # between 1, 32 (inclusive)
    kv_seq_len: int # between 512, 1024 (inclusive)
    batch_size: int # between 1, 32 (inclusive)
    repeat: int = 20
    warm_up: int = 5

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    head_dim: int


import torch
import hidet
import hidet.testing

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

def bench_gpt2_hidet(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 50257, (config.batch_size, config.q_seq_len), dtype=torch.int32, device='cuda')
    ids = hidet.from_torch(ids)
    pos_ids = torch.arange(0, config.q_seq_len, dtype=torch.int32, device='cuda').unsqueeze(0).repeat(config.batch_size, 1)
    pos_ids = hidet.from_torch(pos_ids)
    k_cache = hidet.randn((config.batch_size, model_config.n_layers, model_config.n_heads, config.kv_seq_len, model_config.head_dim), dtype=hidet.float32, device='cuda')
    v_cache = hidet.randn((config.batch_size, model_config.n_layers, model_config.n_heads, config.kv_seq_len, model_config.head_dim), dtype=hidet.float32, device='cuda')

    def bench():
        model(ids, pos_ids, k_cache, v_cache)
    
    return benchmark_func(bench, repeat=config.repeat, median=False, warmup=config.warm_up)


def get_model_llama_hidet(name: str = 'decapoda-research/llama-7b-hf'):
    model, config, tokenizer = get_compiled_model(name, device='cuda', opt=True, batch_size='batch_size', build_space=2)
    model_config = ModelConfig(n_layers=config.num_hidden_layers, n_heads=config.num_attention_heads, head_dim=config.hidden_size // config.num_key_value_heads)
    return model, model_config


def bench_llama_hidet(model, model_config: ModelConfig, config: DecodingSampleConfig):
    ids = torch.randint(0, 32000, (config.batch_size, config.q_seq_len), dtype=torch.int32, device='cuda')
    ids = hidet.from_torch(ids)
    device = 'cuda'
    position_ids = torch.arange(0, 1024, dtype=torch.int32, device=device).unsqueeze(0)
    position_ids = hidet.from_torch(position_ids)

    make_past = lambda: hidet.zeros(
        [config.batch_size, model_config.n_heads, 0, model_config.head_dim], device=device, dtype=hidet.float16
    )
    past_keys_values = [make_past() for _ in range(model_config.n_layers * 2)]

    def bench():
        model(ids, position_ids, *past_keys_values)
    
    return benchmark_func(bench, repeat=config.repeat, median=False, warmup=config.warm_up)


config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=512, batch_size=32)
#####################
hidet.option.parallel_build(False)
model, model_config = get_model_gpt2_hidet('gpt2')
times = bench_gpt2_hidet(model, model_config, config)
print('hidet gpt2:', sum(times) / len(times))
# hidet gpt2: 36.24217748641968


del model
torch.cuda.empty_cache()
current_memory_pool('cuda:0').clear()

#####################
model, model_config = get_model_llama_hidet()
times = bench_llama_hidet(model, model_config, config)
print('hidet-llama:', sum(times) / len(times))
del model

torch.cuda.empty_cache()
current_memory_pool('cuda:0').clear()

