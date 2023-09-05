# %%

from dataclasses import dataclass
from hidet.utils.benchmark import benchmark_func

@dataclass
class DecodingSampleConfig:
    q_seq_len: int # between 1, 32 (inclusive)
    kv_seq_len: int # between 512, 1024 (inclusive)
    batch_size: int # between 1, 32 (inclusive)

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    head_dim: int


import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaForCausalLM

def get_model(name: str):
    assert name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

    hf_gpt2 = GPT2LMHeadModel.from_pretrained(name)
    hf_gpt2.eval()
    hf_gpt2.to('cuda')
    hf_gpt2_config = GPT2Config.from_pretrained(name)
    model_config = ModelConfig(n_layers=hf_gpt2_config.n_layer, n_heads=hf_gpt2_config.n_head, head_dim=hf_gpt2_config.n_embd // hf_gpt2_config.n_head)
    return hf_gpt2, model_config


def bench_gpt2(model: GPT2LMHeadModel, model_config: ModelConfig, config: DecodingSampleConfig, repeat=20):
    ids = torch.randint(0, 50257, (config.batch_size, config.q_seq_len)).to('cuda')
    kv_cache = tuple(tuple(
        torch.randn(config.batch_size, model_config.n_heads, config.kv_seq_len, model_config.head_dim).to('cuda') 
        for _ in range(2)) for _ in range(model_config.n_layers))
    
    def bench():
        model(ids, use_cache=True, past_key_values=kv_cache)
    
    return benchmark_func(bench, repeat=repeat, median=False)


model, model_config = get_model('gpt2')
model = model.to(torch.float16)
total_times = []
with torch.autocast('cuda'):
    model = torch.compile(model)
    for i in range(1, 512, 64):
        config = DecodingSampleConfig(q_seq_len=32, kv_seq_len=i, batch_size=32)
        times = bench_gpt2(model, model_config, config)
        print(sum(times) / len(times))
        total_times.append(sum(times) / len(times))

from matplotlib import pyplot as plt

plt.plot(range(1, 512, 64), total_times)
plt.show()

