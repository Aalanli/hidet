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
from transformers import GPT2Config, GPT2LMHeadModel

from optimum.onnxruntime import ORTModelForCausalLM


