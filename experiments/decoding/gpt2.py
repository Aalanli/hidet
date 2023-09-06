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
from transformers import GPT2Config, GPT2LMHeadModel, LlamaForCausalLM

from optimum.onnxruntime import ORTModelForCausalLM

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
    ids = torch.randint(0, 50257, (config.batch_size, config.q_seq_len)).to('cuda')
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


# config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=512, batch_size=32)
# model, model_config = get_model_gpt2_torch('gpt2')
# model = model.to('cuda')
# model = torch.compile(model, mode='max-autotune')
# with torch.autocast('cuda'):
#     times = bench_gpt2_torch(model, model_config, config)
#     print(sum(times) / len(times))

# model, model_config = get_model_gpt2_hidet('gpt2')
# bench_gpt2_hidet(model, model_config, config)

# model, model_config = get_model_gpt2_onnx('gpt2')
# bench_gpt2_onnx(model, model_config, config)
